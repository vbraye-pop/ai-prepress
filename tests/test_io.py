from pathlib import Path

import numpy as np
import pytest
import tifffile

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


def _linear_ramp_16bit_rgba() -> np.ndarray:
    rgb = _linear_ramp_16bit()
    alpha_ramp = np.linspace(0, 65535, rgb.shape[1], dtype=np.uint16)
    alpha = np.tile(alpha_ramp, (rgb.shape[0], 1))
    return np.dstack([rgb, alpha])


def test_16bit_rgba_tiff_round_trips_alpha_as_a_fourth_channel(tmp_path):
    profile = _wide_gamut_profile()
    source = _linear_ramp_16bit_rgba()
    out = tmp_path / "roundtrip_rgba.tiff"

    save(source, out, icc_profile=profile)
    reloaded = load(out)

    assert reloaded.array.shape[-1] == 4
    assert reloaded.array.dtype == np.uint16
    assert reloaded.icc_profile == profile
    for channel in range(4):
        assert np.unique(reloaded.array[..., channel]).size > 256

    # straight (unassociated) alpha, not premultiplied - see io.save's comment for why
    with tifffile.TiffFile(out) as tf:
        tag = tf.pages[0].tags.get("ExtraSamples")
        assert tag is not None
        assert tag.value[0].name == "UNASSALPHA"


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


def test_webp_round_trip_is_lossless_and_keeps_profile(tmp_path):
    profile = _wide_gamut_profile()
    source = (_linear_ramp_16bit() // 257).astype(np.uint8)
    out = tmp_path / "roundtrip.webp"

    save(source, out, icc_profile=profile)
    reloaded = load(out)

    assert reloaded.array.dtype == np.uint8
    assert reloaded.icc_profile == profile
    assert np.array_equal(reloaded.array, source)  # lossless, not the default WebP mode
