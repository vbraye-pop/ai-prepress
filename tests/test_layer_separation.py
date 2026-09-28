import numpy as np
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


def test_separate_layers_preserves_source_bit_depth_and_rgb_values(monkeypatch):
    source = _source_image()
    canned = _canned_result(8, 8)
    monkeypatch.setattr(layer_separation_module.layer_decompose, "decompose_layers", lambda image, endpoint=None: canned)

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
    monkeypatch.setattr(layer_separation_module.layer_decompose, "decompose_layers", lambda image, endpoint=None: canned)

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
    monkeypatch.setattr(layer_separation_module.layer_decompose, "decompose_layers", lambda image, endpoint=None: canned)

    result = separate_layers(_source_image(height, width))
    assert result.layers[0].bbox == (8, 8, 12, 12)


def test_separate_layers_background_is_8bit_and_untouched_by_source_bit_depth(monkeypatch):
    canned = _canned_result(8, 8)
    monkeypatch.setattr(layer_separation_module.layer_decompose, "decompose_layers", lambda image, endpoint=None: canned)

    result = separate_layers(_source_image())
    assert result.background.array.dtype == np.uint8
    assert result.background.bit_depth == 8
    assert np.array_equal(result.background.array, canned.background)


def test_separate_layers_falls_back_to_srgb_when_source_has_no_profile(monkeypatch):
    canned = _canned_result(8, 8)
    monkeypatch.setattr(layer_separation_module.layer_decompose, "decompose_layers", lambda image, endpoint=None: canned)

    source = _source_image()
    source.icc_profile = None
    result = separate_layers(source)
    assert result.background.icc_profile is not None
    assert result.layers[0].image.icc_profile is not None


def test_separate_layers_returns_empty_list_when_nothing_is_separable(monkeypatch):
    empty = LayerSeparationResult(background=np.full((8, 8, 3), 50, dtype=np.uint8), layer_alphas=[])
    monkeypatch.setattr(layer_separation_module.layer_decompose, "decompose_layers", lambda image, endpoint=None: empty)

    result = separate_layers(_source_image())
    assert result.layers == []
