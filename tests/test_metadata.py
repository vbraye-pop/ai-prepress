from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from ai_prepress.io import save
from ai_prepress.metadata import describe

ADOBE_RGB_PROFILE = Path("/System/Library/ColorSync/Profiles/AdobeRGB1998.icc")


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
    assert info.icc_profile == {}
    assert info.colourspace_guess == "sRGB"  # the fallback assumption for an untagged file
    assert "ImageWidth" in info.raw_tags


def test_reads_the_real_embedded_profile_fields_not_a_guess(tmp_path):
    if not ADOBE_RGB_PROFILE.exists():
        pytest.skip("no wide-gamut ICC profile available on this machine")
    profile = ADOBE_RGB_PROFILE.read_bytes()
    array = np.random.default_rng(3).integers(0, 65535, size=(20, 30, 3), dtype=np.uint16)
    path = tmp_path / "tagged.tiff"
    save(array, path, icc_profile=profile)

    info = describe(path)

    # these come straight from the profile's own header/tags, not inferred
    assert info.icc_profile["description"] == "Adobe RGB (1998)"
    assert info.icc_profile["color_space"] == "RGB"
    assert info.icc_profile["rendering_intent"] in {
        "perceptual",
        "relative colorimetric",
        "saturation",
        "absolute colorimetric",
    }
    assert info.icc_profile["icc_version"] is not None
    assert info.icc_profile["copyright"]

    # the heuristic guess stays available too, alongside the real data - not replaced by it
    assert info.colourspace_guess == "Adobe RGB (1998)"

    # the raw tag dump never leaks the profile's own raw bytes back out as a giant string
    assert "InterColorProfile" not in info.raw_tags


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
