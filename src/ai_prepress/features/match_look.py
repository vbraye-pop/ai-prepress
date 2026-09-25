"""Match Look: statistical color transfer, no model involved.

Capture One markets this feature as AI. Watching what it actually does to
an image shows it driving ordinary parametric adjustments (curves, color
balance) rather than a learned pixel transform - so this is built as
classical color transfer instead of pulling in a model for it. Reinhard et
al. 2001 and Monge-Kantorovich Linearization are both published,
uncopyrighted math; reimplemented directly here rather than depending on
the `color-matcher` package (GPL-3.0, and its own example code clamps
output to 8-bit before saving, which is exactly the thing this project
exists to stop doing).

Two usage patterns share this one function, they're not separate code
paths: point it at a flat/neutral reference before retouching to normalize
a batch without baking in a grade, or at a graded look reference after
retouching for delivery consistency.
"""

from __future__ import annotations

from typing import Literal

import colour
import numpy as np
from PIL import ImageCms

from ai_prepress.io import LoadedImage, from_unit_float, identify_colourspace, to_unit_float

_EPS = 1e-6

# Ruderman et al.'s LMS <-> lαβ matrices, used by the Reinhard method below.
_RGB_TO_LMS = np.array(
    [
        [0.3811, 0.5783, 0.0402],
        [0.1967, 0.7244, 0.0782],
        [0.0241, 0.1288, 0.8444],
    ]
)
_LMS_TO_RGB = np.linalg.inv(_RGB_TO_LMS)
_LMS_TO_LAB = np.array([[1, 1, 1], [1, 1, -2], [1, -1, 0]]) @ np.diag(
    [1 / np.sqrt(3), 1 / np.sqrt(6), 1 / np.sqrt(2)]
)
_LAB_TO_LMS = np.linalg.inv(_LMS_TO_LAB)


def _to_common_space(rgb: np.ndarray, space: str) -> np.ndarray:
    if space == "sRGB":
        return rgb
    return colour.RGB_to_RGB(
        rgb, colour.RGB_COLOURSPACES[space], colour.RGB_COLOURSPACES["sRGB"],
        apply_cctf_decoding=True, apply_cctf_encoding=True,
    )


def _from_common_space(rgb: np.ndarray, space: str) -> np.ndarray:
    if space == "sRGB":
        return rgb
    return colour.RGB_to_RGB(
        rgb, colour.RGB_COLOURSPACES["sRGB"], colour.RGB_COLOURSPACES[space],
        apply_cctf_decoding=True, apply_cctf_encoding=True,
    )


def _reinhard(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    src_lms = np.clip(source, 0, None) @ _RGB_TO_LMS.T + _EPS
    ref_lms = np.clip(reference, 0, None) @ _RGB_TO_LMS.T + _EPS
    src_lab = np.log10(src_lms) @ _LMS_TO_LAB.T
    ref_lab = np.log10(ref_lms) @ _LMS_TO_LAB.T

    src_mean, src_std = src_lab.mean(axis=(0, 1)), src_lab.std(axis=(0, 1))
    ref_mean, ref_std = ref_lab.mean(axis=(0, 1)), ref_lab.std(axis=(0, 1))

    result_lab = (src_lab - src_mean) * (ref_std / np.maximum(src_std, _EPS)) + ref_mean
    result_lms = 10 ** (result_lab @ _LAB_TO_LMS.T)
    return np.clip(result_lms @ _LMS_TO_RGB.T, 0, 1)


def _matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, 0, None)
    return vectors @ np.diag(np.sqrt(values)) @ vectors.T


def _mkl(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    src_pixels = source.reshape(-1, 3)
    ref_pixels = reference.reshape(-1, 3)

    src_mean, ref_mean = src_pixels.mean(axis=0), ref_pixels.mean(axis=0)
    src_cov = np.cov(src_pixels, rowvar=False) + _EPS * np.eye(3)
    ref_cov = np.cov(ref_pixels, rowvar=False) + _EPS * np.eye(3)

    src_sqrt = _matrix_sqrt(src_cov)
    src_sqrt_inv = np.linalg.inv(src_sqrt)
    inner = _matrix_sqrt(src_sqrt @ ref_cov @ src_sqrt)
    transform = src_sqrt_inv @ inner @ src_sqrt_inv

    transformed = (src_pixels - src_mean) @ transform.T + ref_mean
    return np.clip(transformed.reshape(source.shape), 0, 1)


def match_look(
    target: LoadedImage,
    reference: LoadedImage,
    method: Literal["mkl", "reinhard"] = "mkl",
) -> LoadedImage:
    """Apply reference's color statistics to target. Alpha channels aren't handled - RGB only for now."""
    target_space = identify_colourspace(target.icc_profile)
    reference_space = identify_colourspace(reference.icc_profile)

    target_rgb = _to_common_space(to_unit_float(target.array)[..., :3], target_space)
    reference_rgb = _to_common_space(to_unit_float(reference.array)[..., :3], reference_space)

    transfer = _mkl if method == "mkl" else _reinhard
    result_common = transfer(target_rgb, reference_rgb)
    result_rgb = _from_common_space(result_common, target_space)

    icc_profile = target.icc_profile
    if icc_profile is None:
        icc_profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

    return LoadedImage(
        array=from_unit_float(result_rgb, target.array.dtype),
        icc_profile=icc_profile,
        bit_depth=target.bit_depth,
    )
