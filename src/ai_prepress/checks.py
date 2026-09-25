"""Acceptance checks for a before/after image pair.

Run on every feature's output before calling it done - see the project
README for why. Three checks: color drift (Delta-E 2000), whether the bit
depth actually survived, and whether the ICC profile is still attached.
"""

from __future__ import annotations

from dataclasses import dataclass

import colour
import numpy as np

from ai_prepress.io import LoadedImage, identify_colourspace, to_unit_float


@dataclass
class AcceptanceReport:
    delta_e_mean: float
    delta_e_max: float
    delta_e_p95: float
    bit_depth_collapsed: bool
    unique_values_per_channel: list[int]
    icc_profile_present: bool


def _to_lab(image: LoadedImage) -> np.ndarray:
    """Convert to CIE Lab through the image's own color space, staying in float the whole way.

    Delta-E needs Lab, not RGB - feeding it raw RGB numbers gives a value
    that looks plausible but means nothing, since "how different two colors
    look" depends on which color space the numbers are in. Going through
    Pillow's ImageCms for this would mean quantizing to 8-bit first (Pillow
    can't hold 16-bit RGB, same wall hit in io.py), which would make this
    check blind to exactly the sub-8-bit drift it exists to catch - so this
    goes through colour-science's own RGB->XYZ->Lab math instead, entirely
    in float. If there's no embedded profile, sRGB is assumed - the same
    assumption a lot of software makes silently, made explicit here instead.
    """
    rgb = to_unit_float(image.array)[..., :3]
    space = colour.RGB_COLOURSPACES[identify_colourspace(image.icc_profile)]
    xyz = colour.RGB_to_XYZ(rgb, space, apply_cctf_decoding=True)
    return colour.XYZ_to_Lab(xyz, space.whitepoint)


def acceptance_report(before: LoadedImage, after: LoadedImage) -> AcceptanceReport:
    lab_before = _to_lab(before)
    lab_after = _to_lab(after)
    delta_e = colour.delta_E(lab_before, lab_after, method="CIE 2000")

    array = after.array
    unique_counts = [int(np.unique(array[..., c]).size) for c in range(min(array.shape[-1], 3))]
    # only meaningful if the file is stored wider than 8 bits per channel -
    # a genuinely 8-bit source having <=256 values isn't a collapse, it's normal.
    # any channel collapsing counts, not just all three at once.
    collapsed = array.dtype.itemsize > 1 and any(count <= 256 for count in unique_counts)

    return AcceptanceReport(
        delta_e_mean=float(np.mean(delta_e)),
        delta_e_max=float(np.max(delta_e)),
        delta_e_p95=float(np.percentile(delta_e, 95)),
        bit_depth_collapsed=collapsed,
        unique_values_per_channel=unique_counts,
        icc_profile_present=after.icc_profile is not None,
    )
