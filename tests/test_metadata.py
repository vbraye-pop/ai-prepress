import numpy as np
import pytest
import tifffile
from PIL import Image

from ai_prepress.metadata import describe


def test_tiff_reports_real_dimensions_and_dpi(tmp_path):
    array = np.random.default_rng(0).integers(0, 65535, size=(20, 30, 3), dtype=np.uint16)
    path = tmp_path / "test.tiff"
    tifffile.imwrite(path, array, photometric="rgb", resolution=(300, 300), resolutionunit="INCH")

    info = describe(path)

    assert (info.width, info.height, info.channels) == (30, 20, 3)
    assert info.bit_depth == 16
    assert info.dpi == (300.0, 300.0)
    assert info.compression == "NONE"
    assert info.icc_profile_present is False


def test_flags_16bit_storage_that_looks_upsampled(tmp_path):
    ramp = np.linspace(0, 255, 50, dtype=np.uint16) * 257  # only 50 distinct 16-bit values
    array = np.stack([np.tile(ramp, (10, 1))] * 3, axis=-1)
    path = tmp_path / "fake16.tiff"
    tifffile.imwrite(path, array, photometric="rgb")

    info = describe(path)
    assert info.looks_upsampled_from_8bit is True


def test_png_reports_dpi_and_exif(tmp_path):
    array = np.random.default_rng(1).integers(0, 255, size=(15, 25, 3), dtype=np.uint8)
    path = tmp_path / "test.png"
    Image.fromarray(array).save(path, dpi=(150, 150))

    info = describe(path)

    assert (info.width, info.height, info.channels) == (25, 15, 3)
    assert info.bit_depth == 8
    assert info.dpi == pytest.approx((150.0, 150.0), abs=0.1)  # PNG stores DPI as a rational approximation
    assert info.looks_upsampled_from_8bit is False


def test_webp_dispatches_through_the_pil_path(tmp_path):
    array = np.random.default_rng(2).integers(0, 255, size=(15, 25, 3), dtype=np.uint8)
    path = tmp_path / "test.webp"
    Image.fromarray(array).save(path, lossless=True)

    info = describe(path)

    assert (info.width, info.height, info.channels) == (25, 15, 3)
    assert info.bit_depth == 8
