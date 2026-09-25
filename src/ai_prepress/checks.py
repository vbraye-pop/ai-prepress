"""Acceptance checks for a before/after image pair.

Run on every feature's output before calling it done - see the project
README for why. Three checks: color drift (Delta-E 2000), whether the bit
depth actually survived, and whether the ICC profile is still attached.
"""

from __future__ import annotations

import io as bytesio
from dataclasses import dataclass

import colour
import numpy as np
from PIL import Image, ImageCms

from ai_prepress.io import LoadedImage, to_unit_float

_SRGB_PROFILE = ImageCms.createProfile("sRGB")
_LAB_PROFILE = ImageCms.createProfile("LAB")


@dataclass
class AcceptanceReport:
    delta_e_mean: float
    delta_e_max: float
    delta_e_p95: float
    bit_depth_collapsed: bool
    unique_values_per_channel: list[int]
    icc_profile_present: bool


def _to_lab(image: LoadedImage) -> np.ndarray:
    """Convert to CIE Lab through the image's own ICC profile.

    Delta-E needs Lab, not RGB - feeding it raw RGB numbers gives a value
    that looks plausible but means nothing, since "how different two colors
    look" depends on which color space the numbers are in. If there's no
    embedded profile, sRGB is assumed - the same assumption a lot of
    software makes silently, made explicit here instead.
    """
    unit = to_unit_float(image.array)[..., :3]
    as_8bit = np.clip(unit * 255.0 + 0.5, 0, 255).astype(np.uint8)

    if image.icc_profile:
        input_profile = ImageCms.ImageCmsProfile(bytesio.BytesIO(image.icc_profile))
    else:
        input_profile = _SRGB_PROFILE

    transform = ImageCms.buildTransform(input_profile, _LAB_PROFILE, "RGB", "LAB")
    lab_im = ImageCms.applyTransform(Image.fromarray(as_8bit, mode="RGB"), transform)
    lab = np.array(lab_im).astype(np.float64)
    # PIL packs Lab into 0-255 per channel; unpack to the real ranges.
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1] -= 128.0
    lab[..., 2] -= 128.0
    return lab


def acceptance_report(before: LoadedImage, after: LoadedImage) -> AcceptanceReport:
    lab_before = _to_lab(before)
    lab_after = _to_lab(after)
    delta_e = colour.delta_E(lab_before, lab_after, method="CIE 2000")

    array = after.array
    unique_counts = [int(np.unique(array[..., c]).size) for c in range(min(array.shape[-1], 3))]
    # only meaningful if the file is stored wider than 8 bits per channel -
    # a genuinely 8-bit source having <=256 values isn't a collapse, it's normal
    collapsed = array.dtype.itemsize > 1 and all(count <= 256 for count in unique_counts)

    return AcceptanceReport(
        delta_e_mean=float(np.mean(delta_e)),
        delta_e_max=float(np.max(delta_e)),
        delta_e_p95=float(np.percentile(delta_e, 95)),
        bit_depth_collapsed=collapsed,
        unique_values_per_channel=unique_counts,
        icc_profile_present=after.icc_profile is not None,
    )
