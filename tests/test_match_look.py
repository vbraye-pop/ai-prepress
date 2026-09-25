import numpy as np
from PIL import ImageCms

from ai_prepress.checks import acceptance_report
from ai_prepress.features.match_look import match_look
from ai_prepress.io import LoadedImage

_SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _noisy_image(mean, std, seed) -> LoadedImage:
    rng = np.random.default_rng(seed)
    array = rng.normal(mean, std, size=(128, 128, 3))
    array = np.clip(array, 0, 65535).astype(np.uint16)
    return LoadedImage(array=array, icc_profile=_SRGB, bit_depth=16)


def test_result_statistics_move_toward_the_reference():
    dark = _noisy_image(mean=[8000, 8000, 8000], std=1500, seed=1)
    bright_warm = _noisy_image(mean=[45000, 35000, 20000], std=3000, seed=2)

    result = match_look(dark, bright_warm, method="mkl")

    before_mean = dark.array.reshape(-1, 3).mean(axis=0)
    after_mean = result.array.astype(np.float64).reshape(-1, 3).mean(axis=0)
    reference_mean = bright_warm.array.reshape(-1, 3).mean(axis=0)

    # after should land much closer to the reference than the original did
    assert np.abs(after_mean - reference_mean).sum() < np.abs(before_mean - reference_mean).sum()


def test_output_keeps_16bit_depth_and_a_profile():
    dark = _noisy_image(mean=[8000, 8000, 8000], std=1500, seed=1)
    bright = _noisy_image(mean=[45000, 35000, 20000], std=3000, seed=2)

    result = match_look(dark, bright)

    assert result.array.dtype == np.uint16
    assert result.icc_profile is not None
    report = acceptance_report(dark, result)
    assert not report.bit_depth_collapsed


def test_reinhard_method_also_runs():
    dark = _noisy_image(mean=[8000, 8000, 8000], std=1500, seed=1)
    bright = _noisy_image(mean=[45000, 35000, 20000], std=3000, seed=2)
    result = match_look(dark, bright, method="reinhard")
    assert result.array.shape == dark.array.shape
