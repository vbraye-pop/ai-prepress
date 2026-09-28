import numpy as np
import pytest

import ai_prepress.features.retouch_faces as retouch_module
from ai_prepress.face_landmarks import FaceLandmarks
from ai_prepress.features.retouch_faces import (
    RetouchStrengths,
    _box_blur_1d,
    _gaussian_blur,
    rasterize_mask,
    retouch_faces,
)
from ai_prepress.io import LoadedImage


def test_box_blur_preserves_a_constant_field():
    arr = np.full((40, 40), 0.5)
    blurred = _box_blur_1d(arr, radius=5, axis=0)
    assert np.allclose(blurred, 0.5)


def test_box_blur_spreads_an_impulse_without_changing_total_energy():
    arr = np.zeros((41, 41))
    arr[20, 20] = 1.0
    blurred = _gaussian_blur(arr, sigma=4.0)
    assert blurred[20, 20] < 1.0  # spread out
    assert blurred.sum() == pytest.approx(1.0, abs=1e-6)  # energy conserved


def test_rasterize_mask_is_near_1_inside_and_0_outside_with_no_feather():
    polygon = np.array([[10, 10], [90, 10], [90, 90], [10, 90]])
    mask = rasterize_mask(polygon, shape=(100, 100), feather=0)
    assert mask[50, 50] == pytest.approx(1.0)
    assert mask[2, 2] == pytest.approx(0.0)


def test_rasterize_mask_cutout_removes_a_hole():
    outer = np.array([[10, 10], [90, 10], [90, 90], [10, 90]])
    hole = np.array([[40, 40], [60, 40], [60, 60], [40, 60]])
    mask = rasterize_mask(outer, shape=(100, 100), cutouts=[hole], feather=0)
    assert mask[50, 50] == pytest.approx(0.0)  # inside the hole
    assert mask[20, 20] == pytest.approx(1.0)  # inside the outer shape, outside the hole


def _fake_landmarks() -> FaceLandmarks:
    rng = np.random.default_rng(0)
    points = rng.uniform(50, 250, size=(478, 2)).astype(np.float32)
    return FaceLandmarks(points=points)


def _fake_image(size=300) -> LoadedImage:
    rng = np.random.default_rng(1)
    array = rng.uniform(0, 1, size=(size, size, 3)).astype(np.float64)
    return LoadedImage(array=array, icc_profile=None, bit_depth=64)


def test_retouch_faces_returns_none_when_no_face_detected(monkeypatch):
    monkeypatch.setattr(retouch_module, "detect_landmarks", lambda rgb: None)
    result = retouch_faces(_fake_image(), RetouchStrengths(dark_circles=1.0))
    assert result is None


def test_retouch_faces_is_a_no_op_at_zero_strength(monkeypatch):
    monkeypatch.setattr(retouch_module, "detect_landmarks", lambda rgb: _fake_landmarks())
    image = _fake_image()
    result = retouch_faces(image, RetouchStrengths())
    # LF + HF must reconstruct the original exactly when nothing edits LF - this is the
    # identity the whole "HF is never touched" guarantee depends on
    assert np.allclose(result.array, image.array, atol=1e-6)


def test_retouch_faces_leaves_pixels_far_from_any_region_unchanged(monkeypatch):
    monkeypatch.setattr(retouch_module, "detect_landmarks", lambda rgb: _fake_landmarks())
    image = _fake_image(size=300)
    result = retouch_faces(
        image, RetouchStrengths(dark_circles=1.0, even_skin=1.0, contouring=1.0)
    )
    # landmark points are clustered in [50, 250) - a far corner sits well outside every
    # region's mask. Not bit-exact: a Gaussian's tail is never truly zero, so a feathered mask
    # leaves a negligible (~1e-5) trace even at distance - real, expected, and invisible at
    # any real bit depth (1e-3 in unit-float space is under 1 part in 255 at 8-bit).
    assert np.allclose(result.array[0, 0], image.array[0, 0], atol=1e-3)


def test_retouch_faces_fills_in_a_missing_icc_profile(monkeypatch):
    monkeypatch.setattr(retouch_module, "detect_landmarks", lambda rgb: _fake_landmarks())
    image = _fake_image()
    assert image.icc_profile is None
    result = retouch_faces(image, RetouchStrengths(dark_circles=0.5))
    assert result.icc_profile is not None


def test_even_skin_texture_zero_matches_default_hf_retention(monkeypatch):
    monkeypatch.setattr(retouch_module, "detect_landmarks", lambda rgb: _fake_landmarks())
    image = _fake_image()
    with_texture_zero = retouch_faces(image, RetouchStrengths(even_skin=0.5, even_skin_texture=0.0))
    without_texture_field = retouch_faces(image, RetouchStrengths(even_skin=0.5))
    assert np.allclose(with_texture_zero.array, without_texture_field.array)


def test_negative_even_skin_texture_lowers_local_variance_more_than_positive(monkeypatch):
    monkeypatch.setattr(retouch_module, "detect_landmarks", lambda rgb: _fake_landmarks())
    image = _fake_image()
    landmarks = _fake_landmarks()
    from ai_prepress.face_landmarks import skin_region

    oval, _ = skin_region(landmarks)
    cy, cx = int(oval[:, 1].mean()), int(oval[:, 0].mean())
    patch = np.s_[cy - 5 : cy + 5, cx - 5 : cx + 5]

    # same Amount (smoothing strength) in both calls, only Texture differs - negative Texture
    # should leave less pixel-to-pixel variance (less "texture") in the masked region than
    # positive Texture, which mildly boosts the original high-frequency detail back in
    flattened = retouch_faces(image, RetouchStrengths(even_skin=0.5, even_skin_texture=-1.0))
    detailed = retouch_faces(image, RetouchStrengths(even_skin=0.5, even_skin_texture=1.0))
    assert flattened.array[patch].std() < detailed.array[patch].std()


def test_retouch_faces_brightens_the_under_eye_region(monkeypatch):
    monkeypatch.setattr(retouch_module, "detect_landmarks", lambda rgb: _fake_landmarks())
    image = _fake_image()
    result = retouch_faces(image, RetouchStrengths(dark_circles=1.0))
    landmarks = _fake_landmarks()
    from ai_prepress.face_landmarks import under_eye_band

    band = under_eye_band(landmarks, "right")
    cy, cx = int(band[:, 1].mean()), int(band[:, 0].mean())
    assert result.array[cy, cx].mean() > image.array[cy, cx].mean()
