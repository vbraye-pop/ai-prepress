"""Read everything worth knowing about an image file before running anything on it.

Separate from io.load(), which only returns what the processing pipeline
needs (pixel array + profile, nothing more). This is for a human looking at
a file that just arrived and wanting the full picture: real dimensions,
real bit depth, whether it's tagged, whether it secretly started life as
8-bit and got upsampled somewhere upstream, DPI, compression, EXIF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile
from PIL import ExifTags, Image

from ai_prepress.io import ICC_TAG, PIL_EXTS, TIFF_EXTS, identify_colourspace


@dataclass
class ImageMetadata:
    path: str
    file_size_bytes: int
    width: int
    height: int
    channels: int
    dtype: str
    bit_depth: int
    unique_values_per_channel: list[int]
    looks_upsampled_from_8bit: bool
    icc_profile_present: bool
    colourspace_guess: str
    dpi: tuple[float, float] | None
    compression: str | None
    exif: dict[str, str] = field(default_factory=dict)


def describe(path: str | Path) -> ImageMetadata:
    path = Path(path)
    ext = path.suffix.lower()
    file_size = path.stat().st_size

    if ext in TIFF_EXTS:
        return _describe_tiff(path, file_size)
    if ext in PIL_EXTS:
        return _describe_pil(path, file_size)
    raise ValueError(f"unsupported extension: {ext}")


def _unique_counts(array: np.ndarray) -> list[int]:
    if array.ndim == 2:
        return [int(np.unique(array).size)]
    return [int(np.unique(array[..., c]).size) for c in range(min(array.shape[-1], 3))]


def _tiff_dpi(page) -> tuple[float, float] | None:
    x_res = page.tags.get("XResolution")
    y_res = page.tags.get("YResolution")
    if x_res is None or y_res is None:
        return None
    try:
        x = x_res.value[0] / x_res.value[1]
        y = y_res.value[0] / y_res.value[1]
    except (TypeError, ZeroDivisionError, IndexError):
        return None
    return (float(x), float(y))


def _describe_tiff(path: Path, file_size: int) -> ImageMetadata:
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        array = page.asarray()
        icc_tag = page.tags.get(ICC_TAG)
        profile = bytes(icc_tag.value) if icc_tag is not None else None
        dpi = _tiff_dpi(page)
        compression_tag = page.tags.get("Compression")
        compression = compression_tag.value.name if compression_tag is not None else None

    channels = array.shape[-1] if array.ndim == 3 else 1
    bit_depth = array.dtype.itemsize * 8
    unique_counts = _unique_counts(array)

    return ImageMetadata(
        path=str(path),
        file_size_bytes=file_size,
        width=array.shape[1],
        height=array.shape[0],
        channels=channels,
        dtype=str(array.dtype),
        bit_depth=bit_depth,
        unique_values_per_channel=unique_counts,
        looks_upsampled_from_8bit=bit_depth > 8 and all(c <= 256 for c in unique_counts),
        icc_profile_present=profile is not None,
        colourspace_guess=identify_colourspace(profile),
        dpi=dpi,
        compression=compression,
    )


def _describe_pil(path: Path, file_size: int) -> ImageMetadata:
    with Image.open(path) as im:
        profile = im.info.get("icc_profile")
        dpi = im.info.get("dpi")
        array = np.array(im)
        exif = {ExifTags.TAGS.get(tag_id, str(tag_id)): str(value) for tag_id, value in im.getexif().items()}

    channels = array.shape[-1] if array.ndim == 3 else 1

    return ImageMetadata(
        path=str(path),
        file_size_bytes=file_size,
        width=array.shape[1],
        height=array.shape[0],
        channels=channels,
        dtype=str(array.dtype),
        bit_depth=8,
        unique_values_per_channel=_unique_counts(array),
        looks_upsampled_from_8bit=False,  # PIL path only ever loads 8-bit anyway
        icc_profile_present=profile is not None,
        colourspace_guess=identify_colourspace(profile),
        dpi=(float(dpi[0]), float(dpi[1])) if dpi else None,
        compression=None,
        exif=exif,
    )
