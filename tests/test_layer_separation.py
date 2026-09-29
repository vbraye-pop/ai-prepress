import numpy as np
import pytest
from PIL import ImageCms

import ai_prepress.features.layer_separation as layer_separation_module
from ai_prepress.alpha_refine import refine_alpha as _real_refine_alpha  # captured before the
# autouse fixture below ever gets a chance to stub the module attribute of the same name out
from ai_prepress.features.layer_separation import separate_layers
from ai_prepress.io import LoadedImage
from ai_prepress.layer_decompose import LayerSeparationResult

_SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _source_image(height=8, width=8) -> LoadedImage:
    # a non-uniform ramp, not a flat fill - makes "did compositing actually copy the source
    # pixels, not just some constant" a meaningful assertion
    ramp = np.linspace(0, 65535, width, dtype=np.uint16)
    rgb = np.stack([np.tile(ramp, (height, 1))] * 3, axis=-1)
    return LoadedImage(array=rgb, icc_profile=_SRGB, bit_depth=16)


def _canned_result(height, width) -> LayerSeparationResult:
    background = np.full((height, width, 3), 100, dtype=np.uint8)
    alpha = np.zeros((height, width), dtype=np.uint8)
    alpha[2:6, 2:6] = 255
    return LayerSeparationResult(background=background, layer_alphas=[alpha])


def _square_alpha(height, width, y0, y1, x0, x1) -> np.ndarray:
    alpha = np.zeros((height, width), dtype=np.uint8)
    alpha[y0:y1, x0:x1] = 255
    return alpha


@pytest.fixture(autouse=True)
def _default_enhancement_mocks(monkeypatch):
    """object_count and layer_naming are real remote calls, and alpha_refine's guided filter is
    real local compute that degenerates on tiny fixture-sized arrays (radius=64 on an 8x8 image)
    - default every test to a working, uninteresting mock for all three so tests that only care
    about the core compositing/contamination logic don't need to know about any of them. Tests
    that specifically exercise count/naming/refinement behavior override these within their own
    body (a later monkeypatch.setattr in the same test wins)."""
    monkeypatch.setattr(layer_separation_module.object_count, "count_objects", lambda image: 1)
    monkeypatch.setattr(
        layer_separation_module.layer_naming,
        "name_layers",
        lambda layers: [None] * len(layers),
    )
    monkeypatch.setattr(layer_separation_module.alpha_refine, "refine_alpha", lambda alpha, source_rgb, **kwargs: alpha)


def test_separate_layers_preserves_source_bit_depth_and_rgb_values(monkeypatch):
    source = _source_image()
    canned = _canned_result(8, 8)
    monkeypatch.setattr(
        layer_separation_module.layer_decompose, "decompose_layers", lambda image, endpoint=None, layers=None: canned
    )

    result = separate_layers(source)

    assert len(result.layers) == 1
    layer = result.layers[0].image
    assert layer.array.dtype == np.uint16
    assert layer.array.shape == (8, 8, 4)

    non_occluded = canned.layer_alphas[0] > 0
    # the layer's RGB in the non-occluded region must equal the SOURCE's own uint16 values
    # exactly, not the (never-supplied) model RGB the fake doesn't even provide
    assert np.array_equal(layer.array[..., :3][non_occluded], source.array[non_occluded])
    # alpha is scaled to the OUTPUT dtype's own range too, same as RGB - 255 (uint8 "opaque")
    # becomes 65535 (uint16 "opaque"), not a literal 255 sitting inside a uint16 array
    assert (layer.array[..., 3][non_occluded] == 65535).all()
    assert (layer.array[..., 3][~non_occluded] == 0).all()


def test_separate_layers_derives_bbox_from_the_alpha_mask(monkeypatch):
    canned = _canned_result(8, 8)
    monkeypatch.setattr(
        layer_separation_module.layer_decompose, "decompose_layers", lambda image, endpoint=None, layers=None: canned
    )

    result = separate_layers(_source_image())
    assert result.layers[0].bbox == (2, 2, 6, 6)


def test_bbox_ignores_low_level_alpha_noise_spread_across_the_whole_frame(monkeypatch):
    # a real deployment against Qwen-Image-Layered showed its alpha output isn't clean binary -
    # it carries widespread low (1-10) values across nearly the whole frame, not just the
    # object's own soft edge. A synthetic all-or-nothing fixture (like _canned_result's) can't
    # catch a threshold bug this shape - it has to actually be noisy to reproduce it.
    height, width = 20, 20
    rng = np.random.default_rng(0)
    alpha = rng.integers(1, 9, size=(height, width), dtype=np.uint8)  # noise everywhere
    alpha[8:12, 8:12] = 255  # the one real, fully-opaque object
    canned = LayerSeparationResult(background=np.full((height, width, 3), 100, dtype=np.uint8), layer_alphas=[alpha])
    monkeypatch.setattr(
        layer_separation_module.layer_decompose, "decompose_layers", lambda image, endpoint=None, layers=None: canned
    )

    result = separate_layers(_source_image(height, width))
    assert result.layers[0].bbox == (8, 8, 12, 12)


def test_separate_layers_background_is_8bit_and_untouched_by_source_bit_depth(monkeypatch):
    canned = _canned_result(8, 8)
    monkeypatch.setattr(
        layer_separation_module.layer_decompose, "decompose_layers", lambda image, endpoint=None, layers=None: canned
    )

    result = separate_layers(_source_image())
    assert result.background.array.dtype == np.uint8
    assert result.background.bit_depth == 8
    assert np.array_equal(result.background.array, canned.background)


def test_separate_layers_falls_back_to_srgb_when_source_has_no_profile(monkeypatch):
    canned = _canned_result(8, 8)
    monkeypatch.setattr(
        layer_separation_module.layer_decompose, "decompose_layers", lambda image, endpoint=None, layers=None: canned
    )

    source = _source_image()
    source.icc_profile = None
    result = separate_layers(source)
    assert result.background.icc_profile is not None
    assert result.layers[0].image.icc_profile is not None


def test_separate_layers_returns_empty_list_when_nothing_is_separable(monkeypatch):
    empty = LayerSeparationResult(background=np.full((8, 8, 3), 50, dtype=np.uint8), layer_alphas=[])
    monkeypatch.setattr(
        layer_separation_module.layer_decompose, "decompose_layers", lambda image, endpoint=None, layers=None: empty
    )

    result = separate_layers(_source_image())
    assert result.layers == []


# --- automatic layer count -----------------------------------------------------------------


@pytest.mark.parametrize(
    "object_count_value,expected_layers",
    [
        (0, 2),  # clamped up to MIN_LAYERS even for "nothing detected"
        (1, 2),
        (3, 4),
        (10, 8),  # clamped down to MAX_LAYERS
    ],
)
def test_separate_layers_computes_layers_from_object_count(monkeypatch, object_count_value, expected_layers):
    captured = {}
    canned = _canned_result(8, 8)

    def fake_decompose(image, endpoint=None, layers=None):
        captured["layers"] = layers
        return canned

    monkeypatch.setattr(layer_separation_module.object_count, "count_objects", lambda image: object_count_value)
    monkeypatch.setattr(layer_separation_module.layer_decompose, "decompose_layers", fake_decompose)

    separate_layers(_source_image())
    assert captured["layers"] == expected_layers


def test_separate_layers_falls_back_to_a_fixed_layer_count_if_object_count_fails(monkeypatch):
    captured = {}
    canned = _canned_result(8, 8)

    def fake_decompose(image, endpoint=None, layers=None):
        captured["layers"] = layers
        return canned

    def failing_count(image):
        raise RuntimeError("endpoint down")

    monkeypatch.setattr(layer_separation_module.object_count, "count_objects", failing_count)
    monkeypatch.setattr(layer_separation_module.layer_decompose, "decompose_layers", fake_decompose)

    separate_layers(_source_image())  # must not raise
    assert captured["layers"] == layer_separation_module.FALLBACK_LAYER_COUNT


# --- automatic layer naming -----------------------------------------------------------------


def test_separate_layers_attaches_suggested_names(monkeypatch):
    canned = _canned_result(8, 8)
    monkeypatch.setattr(
        layer_separation_module.layer_decompose, "decompose_layers", lambda image, endpoint=None, layers=None: canned
    )
    monkeypatch.setattr(layer_separation_module.layer_naming, "name_layers", lambda layers: ["Mug"])

    result = separate_layers(_source_image())
    assert result.layers[0].suggested_name == "Mug"


def test_separate_layers_suggested_name_is_none_if_naming_fails(monkeypatch):
    canned = _canned_result(8, 8)
    monkeypatch.setattr(
        layer_separation_module.layer_decompose, "decompose_layers", lambda image, endpoint=None, layers=None: canned
    )

    def failing_name(layers):
        raise RuntimeError("endpoint down")

    monkeypatch.setattr(layer_separation_module.layer_naming, "name_layers", failing_name)

    result = separate_layers(_source_image())  # must not raise
    assert result.layers[0].suggested_name is None


# --- guided-filter alpha refinement -----------------------------------------------------------


def test_separate_layers_refines_every_layers_alpha(monkeypatch):
    # bboxes kept well apart (> NAME_ADJACENCY_TOLERANCE) so the contamination pass has nothing
    # to flag here - this test is only about refinement getting called, not the contamination path
    canned = LayerSeparationResult(
        background=np.full((30, 30, 3), 100, dtype=np.uint8),
        layer_alphas=[_square_alpha(30, 30, 2, 8, 2, 8), _square_alpha(30, 30, 20, 26, 20, 26)],
    )
    monkeypatch.setattr(
        layer_separation_module.layer_decompose,
        "decompose_layers",
        lambda image, endpoint=None, layers=None, prompt=None: canned,
    )
    calls = []

    def spy_refine(alpha, source_rgb, **kwargs):
        calls.append(alpha)
        return alpha

    monkeypatch.setattr(layer_separation_module.alpha_refine, "refine_alpha", spy_refine)

    separate_layers(_source_image(30, 30))
    assert len(calls) == 2


def test_separate_layers_guided_filter_actually_changes_a_real_blurred_edge(monkeypatch):
    # the spy-call test above proves the wiring; this proves the wiring isn't calling into a
    # silently-swallowed-exception no-op (refine_alpha is wrapped in a bare except in
    # _composite_layer) by using the REAL guided filter on a fixture sized like
    # test_alpha_refine.py's own (large enough that radius=64 doesn't just degenerate)
    import cv2

    height, width = 200, 200
    edge_x = 100
    source_rgb = np.zeros((height, width, 3), dtype=np.uint16)
    source_rgb[:, edge_x:] = 65535
    sharp_alpha = np.zeros((height, width), dtype=np.float32)
    sharp_alpha[:, edge_x:] = 255
    coarse_alpha = cv2.GaussianBlur(sharp_alpha, (0, 0), sigmaX=8.0).astype(np.uint8)

    source = LoadedImage(array=source_rgb, icc_profile=_SRGB, bit_depth=16)
    canned = LayerSeparationResult(
        background=np.full((height, width, 3), 100, dtype=np.uint8), layer_alphas=[coarse_alpha]
    )
    monkeypatch.setattr(
        layer_separation_module.layer_decompose,
        "decompose_layers",
        lambda image, endpoint=None, layers=None, prompt=None: canned,
    )
    # overrides the autouse identity stub with the real implementation - "a later
    # monkeypatch.setattr in the same test wins", per the fixture's own docstring
    monkeypatch.setattr(layer_separation_module.alpha_refine, "refine_alpha", _real_refine_alpha)

    result = separate_layers(source)
    # the layer's own alpha channel is scaled to the SOURCE dtype's range (uint16 here, see
    # separate_layers' own bit-depth handling), not a literal 0-255 uint8 - normalize both back to
    # [0, 1] before comparing transition widths, same convention alpha_refine.py itself uses
    refined_alpha_unit = result.layers[0].image.array[..., 3].astype(np.float64) / 65535.0
    coarse_alpha_unit = coarse_alpha.astype(np.float64) / 255.0

    def _row_transition_width(row_unit: np.ndarray) -> int:
        below = np.where(row_unit < 0.1)[0]
        above = np.where(row_unit > 0.9)[0]
        if below.size == 0 or above.size == 0:
            return len(row_unit)
        return max(int(above.min()) - int(below.max()), 0)

    coarse_width = _row_transition_width(coarse_alpha_unit[100])
    refined_width = _row_transition_width(refined_alpha_unit[100])
    assert refined_width < coarse_width


# --- contamination detection + recursive repair -----------------------------------------------


def _adjacent_pair_canned(height=20, width=20) -> LayerSeparationResult:
    # two squares that share an edge - real Qwen output rarely lines up pixel-exact, but well
    # within NAME_ADJACENCY_TOLERANCE
    return LayerSeparationResult(
        background=np.full((height, width, 3), 100, dtype=np.uint8),
        layer_alphas=[
            _square_alpha(height, width, 2, 10, 2, 10),
            _square_alpha(height, width, 2, 10, 9, 17),
        ],
    )


def test_separate_layers_flags_near_duplicate_adjacent_names_as_contaminated(monkeypatch):
    calls = []

    def fake_decompose(image, endpoint=None, layers=None, prompt=None):
        calls.append(layers)
        if len(calls) == 1:
            return _adjacent_pair_canned()
        # the repair retry: only one usable sub-layer comes back, not an improvement, so the
        # original flagged layers must survive with their flag intact
        return LayerSeparationResult(
            background=np.zeros((8, 8, 3), dtype=np.uint8), layer_alphas=[_square_alpha(8, 8, 0, 8, 0, 8)]
        )

    monkeypatch.setattr(layer_separation_module.layer_decompose, "decompose_layers", fake_decompose)
    # "jeans" vs "jeans." - punctuation-only difference, the task's own near-duplicate example
    monkeypatch.setattr(layer_separation_module.layer_naming, "name_layers", lambda layers: ["jeans", "jeans."])

    result = separate_layers(_source_image(20, 20))

    assert len(result.layers) == 2
    assert all(layer.contamination_flag for layer in result.layers)


def test_separate_layers_does_not_flag_non_adjacent_layers_even_with_identical_names(monkeypatch):
    canned = LayerSeparationResult(
        background=np.full((30, 30, 3), 100, dtype=np.uint8),
        layer_alphas=[_square_alpha(30, 30, 2, 8, 2, 8), _square_alpha(30, 30, 20, 26, 20, 26)],
    )
    monkeypatch.setattr(
        layer_separation_module.layer_decompose,
        "decompose_layers",
        lambda image, endpoint=None, layers=None, prompt=None: canned,
    )
    monkeypatch.setattr(layer_separation_module.layer_naming, "name_layers", lambda layers: ["Mug", "Mug"])

    result = separate_layers(_source_image(30, 30))

    assert not any(layer.contamination_flag for layer in result.layers)


def test_separate_layers_repairs_a_contaminated_layer_with_a_successful_retry(monkeypatch):
    calls = []
    prompts = []
    retry_result = LayerSeparationResult(
        background=np.zeros((8, 8, 3), dtype=np.uint8),
        layer_alphas=[_square_alpha(8, 8, 0, 8, 0, 4), _square_alpha(8, 8, 0, 8, 4, 8)],
    )

    def fake_decompose(image, endpoint=None, layers=None, prompt=None):
        calls.append(layers)
        prompts.append(prompt)
        if len(calls) == 1:
            return _adjacent_pair_canned()
        return retry_result

    monkeypatch.setattr(layer_separation_module.layer_decompose, "decompose_layers", fake_decompose)
    monkeypatch.setattr(layer_separation_module.layer_naming, "name_layers", lambda layers: ["jeans", "jeans"])

    result = separate_layers(_source_image(20, 20))

    # both original flagged layers got repaired, each into 2 sub-layers
    assert len(result.layers) == 4
    assert not any(layer.contamination_flag for layer in result.layers)  # fresh sub-layers, unflagged
    assert calls[1] == layer_separation_module.RECURSIVE_REPAIR_LAYERS
    assert prompts[1] == "the jeans, including any parts hidden by another object"


def test_separate_layers_repair_falls_back_to_the_original_layer_on_a_raising_retry(monkeypatch):
    calls = []

    def fake_decompose(image, endpoint=None, layers=None, prompt=None):
        calls.append(layers)
        if len(calls) == 1:
            return _adjacent_pair_canned()
        raise RuntimeError("layer-separation endpoint down")

    monkeypatch.setattr(layer_separation_module.layer_decompose, "decompose_layers", fake_decompose)
    monkeypatch.setattr(layer_separation_module.layer_naming, "name_layers", lambda layers: ["jeans", "jeans"])

    result = separate_layers(_source_image(20, 20))  # must not raise

    assert len(result.layers) == 2  # original content kept, not lost
    assert all(layer.contamination_flag for layer in result.layers)


def test_separate_layers_caps_repair_attempts(monkeypatch):
    calls = []

    def fake_decompose(image, endpoint=None, layers=None, prompt=None):
        calls.append(layers)
        if len(calls) == 1:
            # three mutually adjacent layers, all named the same - all three get flagged
            return LayerSeparationResult(
                background=np.full((20, 20, 3), 100, dtype=np.uint8),
                layer_alphas=[
                    _square_alpha(20, 20, 2, 10, 0, 6),
                    _square_alpha(20, 20, 2, 10, 6, 12),
                    _square_alpha(20, 20, 2, 10, 12, 18),
                ],
            )
        raise RuntimeError("simulated repair failure")

    monkeypatch.setattr(layer_separation_module.layer_decompose, "decompose_layers", fake_decompose)
    monkeypatch.setattr(layer_separation_module.layer_naming, "name_layers", lambda layers: ["jeans"] * 3)

    result = separate_layers(_source_image(20, 20))  # must not raise despite the 3rd call raising

    # MAX_REPAIR_ATTEMPTS=2 - only 2 of the 3 flagged layers are ever retried (1 main decompose
    # call + 2 repair calls = 3 total, not 4)
    assert len(calls) == 3
    assert len(result.layers) == 3
