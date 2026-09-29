import numpy as np
import pytest
from PIL import ImageCms

import ai_prepress.features.layer_separation as layer_separation_module
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
    """object_count and layer_naming are real remote calls - default every test to a working,
    uninteresting mock for both so tests that only care about the core compositing/contamination
    logic don't need to know about either. Tests that specifically exercise count/naming behavior
    override these within their own body (a later monkeypatch.setattr in the same test wins).

    separate_layers() tries the per-instance pipeline FIRST (see features/layer_separation.py's
    module docstring) - defaulting list_candidate_objects to "no candidates" makes every one of
    the Qwen-path tests below exercise the REAL fallback, not a second mocked path nobody
    exercises. The other three per-instance clients are stubbed too for the same reason
    object_count/layer_naming are: so a test that doesn't care about them never needs to know they
    exist."""
    monkeypatch.setattr(layer_separation_module.object_count, "count_objects", lambda image: 1)
    monkeypatch.setattr(
        layer_separation_module.layer_naming,
        "name_layers",
        lambda layers: [None] * len(layers),
    )
    monkeypatch.setattr(layer_separation_module.layer_naming, "list_candidate_objects", lambda image, timeout=None: [])
    monkeypatch.setattr(layer_separation_module.object_detect, "detect_objects", lambda image, text_prompt, timeout=None: [])
    monkeypatch.setattr(layer_separation_module.object_segment, "segment_boxes", lambda image, boxes, timeout=None: [])
    monkeypatch.setattr(layer_separation_module.instance_matte, "matte_crops", lambda crops, timeout=None: [])
    monkeypatch.setattr(
        layer_separation_module.background_inpaint, "inpaint_background", lambda source_image, mask, timeout=None: source_image
    )


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


def test_separate_layers_ships_the_coarse_alpha_unmodified(monkeypatch):
    # locks the decision in layer_separation's own module docstring: alpha_refine measurably
    # regressed real hair-edge sharpness and is deliberately not called from _composite_layer -
    # a soft (non-binary) ramp, not a hard-edged square, so a guided filter silently getting
    # re-wired back in would visibly change these exact values, not just pass/fail on a shape check
    height, width = 8, 8
    ramp_alpha = np.linspace(0, 255, width, dtype=np.uint8)
    coarse_alpha = np.tile(ramp_alpha, (height, 1))
    canned = LayerSeparationResult(
        background=np.full((height, width, 3), 100, dtype=np.uint8), layer_alphas=[coarse_alpha]
    )
    monkeypatch.setattr(
        layer_separation_module.layer_decompose,
        "decompose_layers",
        lambda image, endpoint=None, layers=None, prompt=None: canned,
    )

    result = separate_layers(_source_image(height, width))

    alpha_unit = result.layers[0].image.array[..., 3].astype(np.float64) / 65535.0
    coarse_unit = coarse_alpha.astype(np.float64) / 255.0
    np.testing.assert_allclose(alpha_unit, coarse_unit, atol=1e-3)


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


# --- per-instance pipeline (primary path) -------------------------------------------------------


def _detection(label, box, score=0.8) -> dict:
    return {"label": label, "score": score, "box": box}


def test_build_detection_prompt_joins_names_with_periods():
    prompt = layer_separation_module._build_detection_prompt(["pitcher", "orange.", "  book  "])
    assert prompt == "pitcher. orange. book."


def test_dedupe_detections_merges_overlapping_boxes_regardless_of_label():
    book = _detection("book", [0, 0, 10, 10], score=0.9)
    brown_book = _detection("brown book", [1, 1, 10, 10], score=0.6)  # same real object, near-synonym
    orange = _detection("orange", [50, 50, 60, 60], score=0.8)  # unrelated, must survive

    kept = layer_separation_module._dedupe_detections([brown_book, book, orange])

    assert len(kept) == 2
    labels = {d["label"] for d in kept}
    assert labels == {"book", "orange"}  # the higher-score duplicate wins, not just the first seen


def test_separate_layers_uses_per_instance_pipeline_when_detections_found(monkeypatch):
    inpainted_plate = np.full((20, 20, 3), 42, dtype=np.uint8)
    mask = _square_alpha(20, 20, 2, 10, 2, 10)

    monkeypatch.setattr(layer_separation_module.layer_naming, "list_candidate_objects", lambda image, timeout=None: ["mug"])
    monkeypatch.setattr(
        layer_separation_module.object_detect,
        "detect_objects",
        lambda image, text_prompt, timeout=None: [_detection("mug", [2, 2, 10, 10])],
    )
    monkeypatch.setattr(layer_separation_module.object_segment, "segment_boxes", lambda image, boxes, timeout=None: [mask])
    monkeypatch.setattr(
        layer_separation_module.instance_matte,
        "matte_crops",
        lambda crops, timeout=None: [np.full(crops[0].shape[:2], 255, dtype=np.uint8)],  # agrees with SAM2 fully
    )
    monkeypatch.setattr(
        layer_separation_module.background_inpaint, "inpaint_background", lambda source_image, mask, timeout=None: inpainted_plate
    )

    result = separate_layers(_source_image(20, 20))

    assert result.separation_path == "per-instance"
    assert result.requested_layers == 1
    assert len(result.layers) == 1
    assert result.layers[0].suggested_name == "mug"
    assert result.layers[0].bbox == (2, 2, 10, 10)
    assert np.array_equal(result.background.array, inpainted_plate)


def test_separate_layers_falls_back_to_qwen_when_no_candidate_objects(monkeypatch):
    # the default autouse fixture already stubs list_candidate_objects -> [] - this test just
    # makes the resulting fallback explicit rather than incidental
    canned = _canned_result(8, 8)
    monkeypatch.setattr(
        layer_separation_module.layer_decompose, "decompose_layers", lambda image, endpoint=None, layers=None: canned
    )

    result = separate_layers(_source_image())
    assert result.separation_path == "qwen"


def test_separate_layers_falls_back_to_qwen_when_detections_empty_after_dedupe(monkeypatch):
    canned = _canned_result(8, 8)
    monkeypatch.setattr(
        layer_separation_module.layer_decompose, "decompose_layers", lambda image, endpoint=None, layers=None: canned
    )
    monkeypatch.setattr(layer_separation_module.layer_naming, "list_candidate_objects", lambda image, timeout=None: ["ghost"])
    monkeypatch.setattr(layer_separation_module.object_detect, "detect_objects", lambda image, text_prompt, timeout=None: [])

    result = separate_layers(_source_image())
    assert result.separation_path == "qwen"


def test_separate_layers_matte_sam_iou_guard_falls_back_to_sam2_mask_when_birefnet_disagrees(monkeypatch):
    # SAM2's own mask covers the whole 8x8 crop; BiRefNet's alpha for the same crop covers a tiny,
    # almost entirely non-overlapping corner - low agreement, so the composited layer must use
    # SAM2's mask (fully opaque), not BiRefNet's (mostly empty)
    mask = _square_alpha(20, 20, 2, 10, 2, 10)
    disagreeing_matte = np.zeros((8, 8), dtype=np.uint8)
    disagreeing_matte[0:2, 0:2] = 255

    monkeypatch.setattr(layer_separation_module.layer_naming, "list_candidate_objects", lambda image, timeout=None: ["mug"])
    monkeypatch.setattr(
        layer_separation_module.object_detect,
        "detect_objects",
        lambda image, text_prompt, timeout=None: [_detection("mug", [2, 2, 10, 10])],
    )
    monkeypatch.setattr(layer_separation_module.object_segment, "segment_boxes", lambda image, boxes, timeout=None: [mask])
    monkeypatch.setattr(layer_separation_module.instance_matte, "matte_crops", lambda crops, timeout=None: [disagreeing_matte])
    monkeypatch.setattr(
        layer_separation_module.background_inpaint, "inpaint_background", lambda source_image, mask, timeout=None: source_image
    )

    result = separate_layers(_source_image(20, 20))

    assert result.separation_path == "per-instance"
    alpha = result.layers[0].image.array[..., 3]
    # the full 8x8 crop reads opaque (SAM2's mask), not just the tiny corner BiRefNet proposed
    assert (alpha[2:10, 2:10] > 0).all()


def test_separate_layers_per_instance_survives_a_raising_matte_call(monkeypatch):
    mask = _square_alpha(20, 20, 2, 10, 2, 10)

    monkeypatch.setattr(layer_separation_module.layer_naming, "list_candidate_objects", lambda image, timeout=None: ["mug"])
    monkeypatch.setattr(
        layer_separation_module.object_detect,
        "detect_objects",
        lambda image, text_prompt, timeout=None: [_detection("mug", [2, 2, 10, 10])],
    )
    monkeypatch.setattr(layer_separation_module.object_segment, "segment_boxes", lambda image, boxes, timeout=None: [mask])

    def failing_matte(crops):
        raise RuntimeError("endpoint down")

    monkeypatch.setattr(layer_separation_module.instance_matte, "matte_crops", failing_matte)
    monkeypatch.setattr(
        layer_separation_module.background_inpaint, "inpaint_background", lambda source_image, mask, timeout=None: source_image
    )

    result = separate_layers(_source_image(20, 20))  # must not raise, must not silently fall back to Qwen

    assert result.separation_path == "per-instance"
    assert len(result.layers) == 1
    assert result.layers[0].bbox == (2, 2, 10, 10)


def test_separate_layers_leaves_an_honest_hole_at_an_occlusion_boundary(monkeypatch):
    # box A and box B overlap in columns [8, 12) - a real SAM2 mask for each instance already
    # excludes the part covered by the OTHER instance (that's what makes it a mask of a real,
    # partly-occluded object rather than its raw bounding box), simulated directly here rather
    # than re-deriving it, since this test is about what separate_layers DOES with that mask, not
    # about SAM2's own occlusion reasoning
    mask_a = _square_alpha(20, 20, 2, 12, 2, 8)  # pitcher: visible region only, stops before the overlap
    mask_b = _square_alpha(20, 20, 2, 12, 8, 18)  # orange: covers the overlap - it's in front

    monkeypatch.setattr(
        layer_separation_module.layer_naming, "list_candidate_objects", lambda image, timeout=None: ["pitcher", "orange"]
    )
    monkeypatch.setattr(
        layer_separation_module.object_detect,
        "detect_objects",
        lambda image, text_prompt, timeout=None: [_detection("pitcher", [2, 2, 12, 12]), _detection("orange", [8, 2, 18, 12])],
    )
    monkeypatch.setattr(layer_separation_module.object_segment, "segment_boxes", lambda image, boxes, timeout=None: [mask_a, mask_b])
    monkeypatch.setattr(
        layer_separation_module.instance_matte,
        "matte_crops",
        lambda crops, timeout=None: [np.full(crop.shape[:2], 255, dtype=np.uint8) for crop in crops],
    )
    monkeypatch.setattr(
        layer_separation_module.background_inpaint, "inpaint_background", lambda source_image, mask, timeout=None: source_image
    )

    result = separate_layers(_source_image(20, 20))

    pitcher = next(layer for layer in result.layers if layer.suggested_name == "pitcher")
    orange = next(layer for layer in result.layers if layer.suggested_name == "orange")
    pitcher_alpha = pitcher.image.array[..., 3]
    orange_alpha = orange.image.array[..., 3]

    # the occluded column (x=9, inside the overlap) is honestly empty on the pitcher's own layer -
    # nothing reconstructs what's hidden behind the orange - while the orange itself is opaque there
    assert (pitcher_alpha[2:12, 9] == 0).all()
    assert (orange_alpha[2:12, 9] > 0).all()


def test_separate_layers_background_inpaint_receives_the_union_of_instance_masks(monkeypatch):
    mask_a = _square_alpha(20, 20, 2, 10, 2, 10)
    mask_b = _square_alpha(20, 20, 2, 10, 12, 18)
    captured = {}

    def fake_inpaint(source_image, mask, timeout=None):
        captured["mask"] = mask
        return source_image

    monkeypatch.setattr(
        layer_separation_module.layer_naming, "list_candidate_objects", lambda image, timeout=None: ["pitcher", "orange"]
    )
    monkeypatch.setattr(
        layer_separation_module.object_detect,
        "detect_objects",
        lambda image, text_prompt, timeout=None: [_detection("pitcher", [2, 2, 10, 10]), _detection("orange", [12, 2, 18, 10])],
    )
    monkeypatch.setattr(layer_separation_module.object_segment, "segment_boxes", lambda image, boxes, timeout=None: [mask_a, mask_b])
    monkeypatch.setattr(
        layer_separation_module.instance_matte,
        "matte_crops",
        lambda crops, timeout=None: [np.full(crop.shape[:2], 255, dtype=np.uint8) for crop in crops],
    )
    monkeypatch.setattr(layer_separation_module.background_inpaint, "inpaint_background", fake_inpaint)

    separate_layers(_source_image(20, 20))

    expected_union = np.maximum(mask_a, mask_b)
    assert np.array_equal(captured["mask"], expected_union)
