"""Load/save images without losing bit depth or the embedded ICC profile.

Every feature routes its file I/O through here instead of rolling its own.
The reason this file exists at all: a widely-used pipeline (ComfyUI) drops
ICC profiles and clamps to 8-bit at exactly the point where pixel data
crosses a plain numpy array - np.array(im) makes a fresh array with no
profile metadata, and Image.fromarray() back the other way does the same
in reverse. That's the precise bug this project was built to stop
reproducing, so it gets tested directly (see tests/test_io.py).

Pillow can't hold a 16-bit RGB array at all (fromarray raises on
(H, W, 3) uint16 - checked directly, it's not a documentation gap).
TIFF via tifffile is the only path that keeps full bit depth, so it's the
working format here. PNG/JPEG/WebP via Pillow are accepted for 8-bit-only
input/output.
"""

from __future__ import annotations

import io as _stdlib_io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageCms

ICC_TAG = 34675  # TIFFTAG_ICCPROFILE

TIFF_EXTS = {".tif", ".tiff"}
PIL_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class LoadedImage:
    array: np.ndarray
    icc_profile: bytes | None
    bit_depth: int


def load(path: str | Path) -> LoadedImage:
    path = Path(path)
    ext = path.suffix.lower()

    if ext in TIFF_EXTS:
        with tifffile.TiffFile(path) as tf:
            page = tf.pages[0]
            array = page.asarray()
            tag = page.tags.get(ICC_TAG)
            profile = bytes(tag.value) if tag is not None else None
        return LoadedImage(array=array, icc_profile=profile, bit_depth=array.dtype.itemsize * 8)

    if ext in PIL_EXTS:
        with Image.open(path) as im:
            profile = im.info.get("icc_profile")
            array = np.array(im)
        return LoadedImage(array=array, icc_profile=profile, bit_depth=8)

    raise ValueError(f"unsupported extension: {ext}")


def save(image: LoadedImage | np.ndarray, path: str | Path, icc_profile: bytes | None = None) -> None:
    """Write an image back out. `icc_profile` overrides the one on `image` if given.

    Raises if no profile is available at all - an output with no color-space
    tag is exactly the failure mode this module exists to prevent, so there's
    no silent "just save it anyway" path.
    """
    if isinstance(image, LoadedImage):
        array = image.array
        if icc_profile is None:
            icc_profile = image.icc_profile
    else:
        array = image

    if icc_profile is None:
        raise ValueError("no ICC profile to embed - pass one explicitly, even sRGB")

    path = Path(path)
    ext = path.suffix.lower()

    if ext in TIFF_EXTS:
        tifffile.imwrite(
            path,
            array,
            photometric="rgb",
            extratags=[(ICC_TAG, "B", len(icc_profile), icc_profile, True)],
        )
        return

    if ext in PIL_EXTS:
        if array.dtype != np.uint8:
            raise ValueError(f"{ext} only supports 8-bit output - save as .tiff to keep {array.dtype}")
        # WebP defaults to lossy in Pillow - unlike JPEG it has a real lossless mode,
        # so there's no reason to take the lossy path here
        extra = {"lossless": True} if ext == ".webp" else {}
        Image.fromarray(array).save(path, icc_profile=icc_profile, **extra)
        return

    raise ValueError(f"unsupported extension: {ext}")


def to_unit_float(array: np.ndarray) -> np.ndarray:
    """Scale any supported dtype into float64 in [0, 1]."""
    if array.dtype == np.uint8:
        return array.astype(np.float64) / 255.0
    if array.dtype == np.uint16:
        return array.astype(np.float64) / 65535.0
    if array.dtype in (np.float32, np.float64):
        return array.astype(np.float64)
    raise ValueError(f"unhandled dtype: {array.dtype}")


def from_unit_float(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if dtype == np.uint8:
        return np.clip(array * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if dtype == np.uint16:
        return np.clip(array * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
    if dtype in (np.float32, np.float64):
        return array.astype(dtype)
    raise ValueError(f"unhandled dtype: {dtype}")


def identify_colourspace(icc_profile: bytes | None) -> str:
    """Best-effort match of an embedded profile to a named colour-science RGB space.

    A substring check on the profile's description tag, not a real ICC parser -
    enough to tell sRGB from Adobe RGB apart, not meant for arbitrary custom
    profiles. Falls back to sRGB, the same assumption most untagged-file
    handling makes anyway.
    """
    if not icc_profile:
        return "sRGB"
    try:
        description = ImageCms.ImageCmsProfile(_stdlib_io.BytesIO(icc_profile)).profile.profile_description
    except Exception:
        return "sRGB"
    if "adobe rgb" in description.lower():
        return "Adobe RGB (1998)"
    return "sRGB"
