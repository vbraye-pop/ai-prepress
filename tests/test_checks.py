import numpy as np
from PIL import ImageCms

from ai_prepress.checks import acceptance_report
from ai_prepress.io import LoadedImage

_SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _flat_image(value: int, size: int = 32) -> LoadedImage:
    array = np.full((size, size, 3), value, dtype=np.uint8)
    return LoadedImage(array=array, icc_profile=_SRGB, bit_depth=8)


def test_identical_images_have_near_zero_drift():
    image = _flat_image(128)
    report = acceptance_report(image, image)
    assert report.delta_e_mean < 0.5
    assert report.icc_profile_present


def test_visibly_different_images_report_real_drift():
    before = _flat_image(50)
    after = _flat_image(200)
    report = acceptance_report(before, after)
    assert report.delta_e_mean > 10


def test_missing_profile_is_flagged():
    image = LoadedImage(array=np.full((16, 16, 3), 100, dtype=np.uint8), icc_profile=None, bit_depth=8)
    report = acceptance_report(image, image)
    assert report.icc_profile_present is False


def test_8bit_collapse_flag_only_fires_on_wider_storage():
    ramp = np.linspace(0, 65535, 512, dtype=np.uint16)
    wide = np.stack([np.tile(ramp, (16, 1))] * 3, axis=-1)
    collapsed_16bit = (wide // 65535 * 65535).astype(np.uint16)  # all one value, still 16-bit storage

    normal_8bit = LoadedImage(array=np.full((16, 512, 3), 128, dtype=np.uint8), icc_profile=_SRGB, bit_depth=8)
    fake_16bit = LoadedImage(array=collapsed_16bit, icc_profile=_SRGB, bit_depth=16)

    assert acceptance_report(normal_8bit, normal_8bit).bit_depth_collapsed is False
    assert acceptance_report(fake_16bit, fake_16bit).bit_depth_collapsed is True
