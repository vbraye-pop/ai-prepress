"""Retouch Faces: low-frequency masked edits for Dark Circles, Even Skin, and Contouring.

Capture One's own copy describes all three as smooth, region-scoped tone/shading operations
(dark circles = soften+brighten, even skin = contrast reduction, contouring = shading). That
maps directly onto classical frequency separation: split the image into a low-frequency (LF,
tone/shading) layer and a high-frequency (HF, texture/pores/detail) layer, edit LF only inside
a region mask, recombine with the HF layer. Dark Circles and Contouring never touch HF at all -
that's what makes "waxy skin" impossible by construction for those two, rather than relying on a
generative model not to over-smooth (see the "retouch-faces-sota" project note). Even Skin's
Texture control is the one deliberate exception: Capture One's own panel has a signed Texture
slider alongside Amount, independently confirmed from their blog - it scales HF within the
even_skin mask rather than leaving it untouched, on purpose (see RetouchStrengths).

Blemish removal (the 4th sub-tool) isn't here - it's a genuinely different, discrete detect-and-
inpaint task, not a frequency-separation one, and needs a generative model this project doesn't
have wired up yet.

No scipy (this project deliberately doesn't depend on it, see colour-science's own optional-
scipy warning at import) and PIL's GaussianBlur flatly rejects float/'F'-mode images (confirmed
by hand: `Image.fromarray(x, mode='F').filter(GaussianBlur(...))` raises `ValueError: image has
wrong mode`) - routing 16-bit data through it would mean quantizing to 8-bit first, the exact
bug class this whole project exists to prevent. _gaussian_blur below is a from-scratch separable
box-blur approximation (3 passes, Kovesi's radius formula for a close Gaussian match) operating
directly on the same float64 arrays the rest of the pipeline uses - no image-library round trip
for pixel data at all. PIL IS used for mask rasterization (see rasterize_mask) - fine there,
since a mask is a shape boundary, not a value that needs preserving to 16-bit precision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageCms, ImageDraw

from ai_prepress.face_landmarks import (
    FaceLandmarks,
    cheek_region,
    detect_landmarks,
    skin_region,
    under_eye_band,
)
from ai_prepress.io import LoadedImage, from_unit_float, to_unit_float


def _box_blur_1d(arr: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """Exact box blur via a cumulative-sum window sum - O(N) regardless of radius, edge-padded
    (replicated, not zero) so the blur doesn't darken toward the image border."""
    if radius <= 0:
        return arr
    pad = [(0, 0)] * arr.ndim
    pad[axis] = (radius, radius)
    padded = np.pad(arr, pad, mode="edge")

    zero_shape = list(padded.shape)
    zero_shape[axis] = 1
    cumsum = np.concatenate([np.zeros(zero_shape, dtype=arr.dtype), np.cumsum(padded, axis=axis)], axis=axis)

    window = 2 * radius + 1
    n = arr.shape[axis]
    hi = np.take(cumsum, np.arange(window, window + n), axis=axis)
    lo = np.take(cumsum, np.arange(0, n), axis=axis)
    return (hi - lo) / window


def _box_radii(sigma: float, passes: int = 3) -> list[int]:
    """Kovesi's near-Gaussian box-blur radii: N box blurs of these widths sum to approximately
    a Gaussian of the given sigma. Standard technique (real-time graphics, not novel here)."""
    if sigma <= 0:
        return [0] * passes
    ideal_width = math.sqrt((12 * sigma * sigma / passes) + 1)
    wl = int(ideal_width)
    if wl % 2 == 0:
        wl -= 1
    wu = wl + 2
    m_ideal = (12 * sigma * sigma - passes * wl * wl - 4 * passes * wl - 3 * passes) / (-4 * wl - 4)
    m = round(m_ideal)
    return [(wl - 1) // 2 if i < m else (wu - 1) // 2 for i in range(passes)]


def _gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    result = arr
    for radius in _box_radii(sigma):
        result = _box_blur_1d(result, radius, axis=0)
        result = _box_blur_1d(result, radius, axis=1)
    return result


def rasterize_mask(
    polygon: np.ndarray, shape: tuple[int, int], cutouts: list[np.ndarray] | None = None, feather: float = 0.0
) -> np.ndarray:
    """Polygon(s) -> a (H, W) float mask in [0, 1]. Drawn via PIL at full resolution (mask
    shapes are fine through PIL - it's pixel VALUES that can't round-trip through it, see the
    module docstring). Cutouts are painted out afterward rather than via an even-odd fill rule,
    since PIL's ImageDraw.polygon doesn't support one. `feather` is a blur sigma in pixels
    applied to the hard-edged mask - a hard polygon edge on a brightened region looks like a
    sticker cutout once composited, feathering is not optional polish."""
    height, width = shape
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    draw.polygon([tuple(p) for p in polygon], fill=255)
    for cutout in cutouts or []:
        draw.polygon([tuple(p) for p in cutout], fill=0)

    mask = np.asarray(mask_img, dtype=np.float64) / 255.0
    if feather > 0:
        mask = _gaussian_blur(mask, feather)
    return mask


@dataclass
class RetouchStrengths:
    dark_circles: float = 0.0  # 0-1, brighten under-eye
    even_skin: float = 0.0  # 0-1 "Amount" - pull skin tone toward its own regional mean
    even_skin_texture: float = 0.0  # -1..1 "Texture" - HF retention within the even_skin mask.
    # 0 is today's default (HF untouched), matching Capture One's own Even Skin panel: negative
    # suppresses high-frequency detail beyond what Amount alone does (their own worked example,
    # "80 Amount / -70 Texture", is described as an editorial-polish look), positive mildly
    # boosts it back (capped well short of oversharpening - see _texture_multiplier).
    contouring: float = 0.0  # 0-1, darken cheek - the crudest of the three, see module docstring


def _feather_radius(landmarks: FaceLandmarks) -> float:
    eye_width = np.linalg.norm(landmarks.subset([33])[0] - landmarks.subset([133])[0])
    return max(1.0, eye_width * 0.12)


def _apply_dark_circles(lf: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
    lift = 0.16 * strength
    return lf + mask[..., None] * lift


def _masked_regional_blur(lf: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """A coarse blur of `lf` that only ever averages pixels the mask actually selects - plain
    `_gaussian_blur(lf, sigma)` blurs the WHOLE frame, so near a mask edge (a forehead close to
    the hairline, in testing) it pulls in blue sky and hair color across the boundary, not just
    more skin. That contamination was the real cause of a visible tonal step at the mask edge
    that didn't go away even at low strength - it wasn't a blend-amount bug, the target value
    itself was wrong. Normalized convolution (weight by the mask, blur both, divide) is the
    standard fix: it only ever mixes in the pixels the mask actually includes."""
    weighted = _gaussian_blur(lf * mask[..., None], sigma)
    weight = np.maximum(_gaussian_blur(mask, sigma), 1e-6)
    return weighted / weight[..., None]


def _apply_even_skin(lf: np.ndarray, mask: np.ndarray, strength: float, regional: np.ndarray) -> np.ndarray:
    """Blends toward `regional` (a masked coarse blur of the same LF layer, see
    _masked_regional_blur), not a single flat scalar mean - an even earlier version used one
    global mean for the whole skin region, which pulled the brightest area (sunlit forehead in
    testing) down the hardest. A locally-smoothed target preserves the face's own overall
    lighting gradient while still removing small-scale blotchiness, which is what "reduces
    contrast" should mean here - a second, coarser frequency band, not a constant."""
    return lf + mask[..., None] * strength * (regional - lf)


def _texture_multiplier(mask: np.ndarray, texture: float) -> np.ndarray:
    """A per-pixel HF multiplier for the even_skin region - 1.0 (unchanged) outside the mask,
    ramping toward `1 + texture` inside it. texture in [-1, 0) suppresses fine detail (more
    flattening than Amount alone gives); (0, 1] mildly restores/boosts it, capped at 1.3x so
    this stays "restore some texture" rather than an open-ended sharpen."""
    factor = 1.0 + np.clip(texture, -1.0, 0.3)
    return 1.0 + mask * (factor - 1.0)


def _apply_contouring(lf: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
    # plain darken, not a real lighting-aware shading model - Capture One's own copy describes
    # contouring as analyzing facial lighting to deepen existing shadows, which this doesn't do.
    # Flagged as the crudest of the three rather than implied to match that behavior.
    shadow = 0.12 * strength
    return lf - mask[..., None] * shadow


def retouch_faces(image: LoadedImage, strengths: RetouchStrengths) -> LoadedImage | None:
    """Returns None if no face is detected - a photo without one is a real case the caller
    should handle, not a hidden failure. LF/HF split runs once regardless of how many effects
    are active; each active effect edits the shared LF layer. HF stays untouched except where
    Even Skin's Texture control deliberately scales it (see RetouchStrengths.even_skin_texture)."""
    unit = to_unit_float(image.array)[..., :3]
    rgb_8bit = (np.clip(unit, 0.0, 1.0) * 255 + 0.5).astype(np.uint8)

    landmarks = detect_landmarks(rgb_8bit)
    if landmarks is None:
        return None

    height, width = unit.shape[:2]
    face_width = np.linalg.norm(landmarks.subset([234])[0] - landmarks.subset([454])[0])
    feather = _feather_radius(landmarks)

    lf_sigma = max(2.0, face_width * 0.05)
    lf = _gaussian_blur(unit, lf_sigma)
    hf = unit - lf
    hf_multiplier = np.ones((height, width), dtype=np.float64)

    if strengths.dark_circles > 0:
        for side in ("right", "left"):
            mask = rasterize_mask(under_eye_band(landmarks, side), (height, width), feather=feather)
            lf = _apply_dark_circles(lf, mask, strengths.dark_circles)

    if strengths.even_skin > 0:
        oval, cutouts = skin_region(landmarks)
        mask = rasterize_mask(oval, (height, width), cutouts=cutouts, feather=feather * 1.5)
        regional = _masked_regional_blur(lf, mask, sigma=lf_sigma * 4)
        lf = _apply_even_skin(lf, mask, strengths.even_skin, regional)
        if strengths.even_skin_texture != 0:
            hf_multiplier *= _texture_multiplier(mask, strengths.even_skin_texture)

    if strengths.contouring > 0:
        for side in ("right", "left"):
            mask = rasterize_mask(cheek_region(landmarks, side), (height, width), feather=feather)
            lf = _apply_contouring(lf, mask, strengths.contouring)

    result_unit = np.clip(lf + hf * hf_multiplier[..., None], 0.0, 1.0)
    icc_profile = image.icc_profile
    if icc_profile is None:
        # io.save() requires one - same fallback match_look.py uses for the same reason
        icc_profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

    return LoadedImage(
        array=from_unit_float(result_unit, image.array.dtype),
        icc_profile=icc_profile,
        bit_depth=image.bit_depth,
    )
