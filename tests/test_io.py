from pathlib import Path

import numpy as np
import pytest

from ai_prepress.io import load, save

ADOBE_RGB_PROFILE = Path("/System/Library/ColorSync/Profiles/AdobeRGB1998.icc")


def _wide_gamut_profile() -> bytes:
    if not ADOBE_RGB_PROFILE.exists():
        pytest.skip("no wide-gamut ICC profile available on this machine")
    return ADOBE_RGB_PROFILE.read_bytes()


def _linear_ramp_16bit() -> np.ndarray:
    ramp = np.linspace(0, 65535, 512, dtype=np.uint16)
    img = np.tile(ramp, (384, 1))
    return np.stack([img, img // 2, img // 3], axis=-1).astype(np.uint16)


def test_16bit_tiff_round_trips_bit_depth_and_profile(tmp_path):
    profile = _wide_gamut_profile()
    source = _linear_ramp_16bit()
    out = tmp_path / "roundtrip.tiff"

    save(source, out, icc_profile=profile)
    reloaded = load(out)

    assert reloaded.array.dtype == np.uint16
    assert reloaded.icc_profile == profile
    for channel in range(3):
        # anything <= 256 unique values on a 16-bit source means it got
        # collapsed to 8-bit somewhere in the round trip
        assert np.unique(reloaded.array[..., channel]).size > 256


def test_save_refuses_to_write_without_a_profile(tmp_path):
    source = _linear_ramp_16bit()
    with pytest.raises(ValueError, match="ICC profile"):
        save(source, tmp_path / "untagged.tiff")


def test_png_round_trip_is_8bit_only_by_design(tmp_path):
    profile = _wide_gamut_profile()
    source = (_linear_ramp_16bit() // 257).astype(np.uint8)  # 16-bit range down to 8-bit
    out = tmp_path / "roundtrip.png"

    save(source, out, icc_profile=profile)
    reloaded = load(out)

    assert reloaded.array.dtype == np.uint8
    assert reloaded.icc_profile == profile

    with pytest.raises(ValueError, match="only supports 8-bit"):
        save(_linear_ramp_16bit(), tmp_path / "should_fail.png", icc_profile=profile)
