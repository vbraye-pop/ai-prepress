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

import colour
import numpy as np
from PIL import Image, ImageCms, ImageDraw

from ai_prepress.face_landmarks import (
    FaceLandmarks,
    cheek_region,
    detect_landmarks,
    eye_sclera_region,
    lip_region,
    mouth_interior_region,
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


def _box_extreme_1d(arr: np.ndarray, radius: int, axis: int, op: str) -> np.ndarray:
    """Exact min/max filter over a size-(2*radius+1) window along one axis - the raster
    primitive behind erode ('erode') and dilate ('dilate'). Grayscale erosion/dilation by a
    rectangle is separable (a rectangle is the Minkowski sum of a horizontal and a vertical
    segment), so one pass per axis is an EXACT result for a rectangular structuring element, not
    an approximation the way _gaussian_blur's multi-pass box trick approximates a Gaussian - the
    known limitation is isotropy, not exactness: a square structuring element grows a shape's
    corners by radius*sqrt(2) versus radius on an axis-aligned edge, immaterial on the organic
    polygon shapes this module produces. Implemented as a direct windowed reduction rather than
    an O(log radius) doubling scheme - this only runs once per Apply click, not per slider drag,
    so the simpler, more obviously-correct version was chosen over the faster one."""
    if radius <= 0:
        return arr
    pad = [(0, 0)] * arr.ndim
    pad[axis] = (radius, radius)
    padded = np.pad(arr, pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=2 * radius + 1, axis=axis)
    reducer = np.max if op == "dilate" else np.min
    return reducer(windows, axis=-1)


def _reshape_mask(mask01: np.ndarray, edge_px: float) -> np.ndarray:
    """edge_px > 0 dilates (grows the region), < 0 erodes (shrinks it), 0 is an exact no-op -
    that exactness matters, since RetouchStrengths()'s zero-strength no-op test depends on
    defaults reproducing today's output bit-for-bit."""
    radius = round(abs(edge_px))
    if radius == 0:
        return mask01
    op = "dilate" if edge_px > 0 else "erode"
    result = mask01
    for axis in (0, 1):
        result = _box_extreme_1d(result, radius, axis, op)
    return result


def rasterize_mask(
    polygon: np.ndarray,
    shape: tuple[int, int],
    cutouts: list[np.ndarray] | None = None,
    feather: float = 0.0,
    edge: float = 0.0,
) -> np.ndarray:
    """Polygon(s) -> a (H, W) float mask in [0, 1]. Drawn via PIL at full resolution (mask
    shapes are fine through PIL - it's pixel VALUES that can't round-trip through it, see the
    module docstring).

    Order matters here: outer fill -> edge (erode/dilate) -> cutouts -> feather. Reshaping the
    mask AFTER cutouts were already removed would let a dilate grow back into the eyes/lips a
    cutout was deliberately excluding - cutouts are always applied at full strength, unaffected
    by edge, which is why they're a separate raster subtracted afterward rather than being drawn
    directly into the same image the way the very first version of this function did.

    `feather` is a blur sigma in pixels applied to the hard mask edge - a hard polygon edge on a
    brightened region looks like a sticker cutout once composited, feathering is not optional
    polish. `edge` is a signed pixel offset for growing/shrinking the region boundary itself
    (see _reshape_mask) - Lightroom Classic's own "Reshape" mask tool exposes exactly this pair
    (their Feather/Edge sliders) as the cheap fix for "the AI mask is wrong" before a full paint
    brush, which this project doesn't have yet."""
    height, width = shape
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    draw.polygon([tuple(p) for p in polygon], fill=255)
    mask = np.asarray(mask_img, dtype=np.float64) / 255.0

    if edge != 0:
        mask = _reshape_mask(mask, edge)

    if cutouts:
        cutout_img = Image.new("L", (width, height), 0)
        cutout_draw = ImageDraw.Draw(cutout_img)
        for cutout in cutouts:
            cutout_draw.polygon([tuple(p) for p in cutout], fill=255)
        cutout_mask = np.asarray(cutout_img, dtype=np.float64) / 255.0
        mask = mask * (1.0 - cutout_mask)

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
    feather_amount: float = 1.0  # multiplier on the region-proportional feather radius below,
    # 1.0 reproduces today's exact fixed value, 0 is a hard edge, >1 softer - Lightroom Classic's
    # own mask "Reshape" step exposes this as a user-facing slider rather than a fixed constant.
    edge_amount: float = 0.0  # -1..1, negative erodes / positive dilates every active region's
    # mask boundary (one shared value, same pattern feather already uses rather than a field per
    # effect) - the cheap fix for "the AI-derived region is a bit off" without needing a brush.
    eye_whiten: float = 0.0  # 0-1, sclera only (iris excluded by mask geometry, not color math)
    teeth_whiten: float = 0.0  # 0-1, tooth-colored pixels only within the mouth-interior mask
    lip_enhance: float = 0.0  # 0-1, boosts the person's OWN lip color - not a recolor/lipstick
    # mode, which every source confirms is a different, beauty-app-oriented default this
    # commercial/prepress-positioned tool deliberately doesn't default to (see _apply_lip_enhance)


def _feather_radius(landmarks: FaceLandmarks) -> float:
    eye_width = np.linalg.norm(landmarks.subset([33])[0] - landmarks.subset([133])[0])
    return max(1.0, eye_width * 0.12)


def _edge_offset_px(landmarks: FaceLandmarks, amount: float) -> float:
    """Mirrors _feather_radius: proportional to the face's own scale (eye width), not image
    resolution or a raw pixel count, so the same slider value means the same relative reshape
    regardless of photo size. Capped at half an eye-width so the slider's extreme erodes a
    region visibly without being a one-click "the mask disappeared" trap."""
    eye_width = np.linalg.norm(landmarks.subset([33])[0] - landmarks.subset([133])[0])
    return eye_width * 0.5 * np.clip(amount, -1.0, 1.0)


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


# ---------------------------------------------------------------------------------------------
# Eyes, teeth, lips: direct HSL color grades within a mask, not frequency-separation. These
# aren't texture-preserving smoothing operations the way the skin tools above are - they're
# small, targeted color corrections, so they run on the final recombined RGB image (see
# retouch_faces' second per-face pass) rather than being entangled with the LF/HF machinery.
# Grounded in a research pass across professional retouching tutorials (PhotoshopCafe, Fstoppers,
# Retouching Academy, Lightroom/Kelby, Photoshop Essentials, PortraitPro/ON1/Aperty documentation)
# rather than guessed - see each function's docstring for what's sourced vs an approximation.
# ---------------------------------------------------------------------------------------------


def _rgb_to_hsl(rgb: np.ndarray) -> np.ndarray:
    return colour.RGB_to_HSL(np.clip(rgb, 0.0, 1.0))


def _hsl_to_rgb(hsl: np.ndarray) -> np.ndarray:
    return np.clip(colour.HSL_to_RGB(hsl), 0.0, 1.0)


def _apply_eye_whiten(rgb: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
    """Desaturates the yellow/red cast and applies a small, highlight-rolloff-capped lightness
    lift - not a flat brighten. Professional practice treats sclera discoloration as a color-cast
    correction first, brightness second, and keeps the lightness move small: over-brightening or
    over-desaturating into a flat, glowing white ("milk eyes"/"alien eyes") is the single
    most-documented failure mode across every source consulted, not a blown-out catchlight - and
    the catchlight is protected structurally regardless, since it sits on the iris and this mask
    excludes the iris by construction (see face_landmarks.eye_sclera_region). Saturation is
    reduced by up to 20%, never to zero, so blood vessel texture stays visible - full
    desaturation reads as fake in every source that discussed it."""
    hsl = _rgb_to_hsl(rgb)
    lightness = hsl[..., 2]
    hsl[..., 1] = hsl[..., 1] * (1.0 - 0.2 * strength)
    rolloff = np.clip(1.0 - lightness, 0.0, 1.0)  # tapers to 0 as pixels are already bright
    hsl[..., 2] = np.clip(lightness + 0.12 * strength * rolloff, 0.0, 1.0)
    whitened = _hsl_to_rgb(hsl)
    return rgb + mask[..., None] * (whitened - rgb)


def _tooth_submask(rgb: np.ndarray, geometric_mask: np.ndarray) -> np.ndarray:
    """Restricts the mouth-interior polygon to actual tooth-colored pixels - gums, the dark gap
    between teeth, and the tongue all sit inside the same geometric polygon, but every source on
    teeth retouching treats keeping them out as a masking problem, not a color-math one (manual
    tutorials brush-mask tooth surfaces by hand; Lightroom's AI "Teeth" mask is a real semantic
    segmentation, not the raw mouth-opening shape). Teeth read reliably brighter than gums/gaps/
    tongue within the same mouth, so a threshold relative to THIS region's own lightness
    distribution - not a fixed value, which wouldn't hold across different lighting or skin
    tones - isolates them."""
    lightness = _rgb_to_hsl(rgb)[..., 2]
    within = geometric_mask > 0.5
    if not within.any():
        return geometric_mask
    threshold = np.percentile(lightness[within], 55)
    tooth_like = (lightness > threshold).astype(np.float64)
    return geometric_mask * tooth_like


def _apply_teeth_whiten(rgb: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
    """-60 to -80% saturation reduction within the tooth mask (Kelby's Lightroom workflow: -62%;
    Photoshop Essentials' manual technique: -70/-80%), never to zero - fully desaturated teeth
    read gray and dead in every source that showed the failure mode - plus a modest lightness
    lift (+10-20% lightness, or +0.79 exposure stops in Kelby's numbers; both land in the same
    ballpark once converted). `mask` is expected to already be _tooth_submask-filtered, not the
    raw mouth-interior polygon - color math alone can't tell a gum from a tooth."""
    hsl = _rgb_to_hsl(rgb)
    hsl[..., 1] = hsl[..., 1] * (1.0 - 0.7 * strength)
    hsl[..., 2] = np.clip(hsl[..., 2] + 0.1 * strength, 0.0, 1.0)
    whitened = _hsl_to_rgb(hsl)
    return rgb + mask[..., None] * (whitened - rgb)


def _apply_lip_enhance(rgb: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
    """Enhances the person's own lip color rather than recoloring it - the confirmed default for
    commercial/portrait-oriented tools (ON1, Aperty, the manual Capture One/Lightroom workflow),
    as opposed to beauty/glamour apps (PortraitPro's Lipstick feature), which default to a color
    swatch instead. A swatch/recolor mode would be a genuinely different, separate code path
    (compositing toward a chosen target hue), not a variant of this function.

    Lightness is left untouched entirely - texture, the vermilion border, and the specular
    highlight live almost entirely in L, so leaving it alone preserves them without needing a
    frequency-separation pass the way skin smoothing does. Only chroma (saturation) is scaled up,
    by up to 25% at full strength, gated back toward no boost as pixels approach the lip's own
    highlight range: two independent sources (PortraitPro shipping Shine as a slider separate
    from its lipstick color, and a manual retouching tutorial that dodges highlights back in
    *after* its color step) show professional practice treats the specular as a second concern
    the color step must not touch, not something the color math handles on its own. The boost is
    multiplicative (`s * factor`), which already keeps near-zero-chroma highlight pixels near
    zero without a separate floor - amplifying almost nothing by any factor is still almost
    nothing."""
    hsl = _rgb_to_hsl(rgb)
    hue, saturation, lightness = hsl[..., 0], hsl[..., 1], hsl[..., 2]

    within = mask > 0.5
    highlight_threshold = np.percentile(lightness[within], 80) if within.any() else 1.0
    rolloff_width = 0.15
    gate = np.clip((highlight_threshold - lightness) / rolloff_width, 0.0, 1.0)

    boost = 1.0 + 0.25 * strength * gate
    new_hsl = np.stack([hue, np.clip(saturation * boost, 0.0, 1.0), lightness], axis=-1)
    enhanced = _hsl_to_rgb(new_hsl)
    return rgb + mask[..., None] * (enhanced - rgb)


def _apply_face(
    lf: np.ndarray,
    hf_multiplier: np.ndarray,
    landmarks: FaceLandmarks,
    strengths: RetouchStrengths,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """One face's worth of masked edits against the shared lf/hf_multiplier layers - factored
    out of retouch_faces so the multi-face loop there is just "call this once per face" rather
    than a second copy of the per-effect logic."""
    feather = _feather_radius(landmarks) * strengths.feather_amount
    edge = _edge_offset_px(landmarks, strengths.edge_amount)

    if strengths.dark_circles > 0:
        for side in ("right", "left"):
            mask = rasterize_mask(under_eye_band(landmarks, side), shape, feather=feather, edge=edge)
            lf = _apply_dark_circles(lf, mask, strengths.dark_circles)

    if strengths.even_skin > 0:
        oval, cutouts = skin_region(landmarks)
        mask = rasterize_mask(oval, shape, cutouts=cutouts, feather=feather * 1.5, edge=edge)
        face_width = np.linalg.norm(landmarks.subset([234])[0] - landmarks.subset([454])[0])
        lf_sigma = max(2.0, face_width * 0.05)
        regional = _masked_regional_blur(lf, mask, sigma=lf_sigma * 4)
        lf = _apply_even_skin(lf, mask, strengths.even_skin, regional)
        if strengths.even_skin_texture != 0:
            hf_multiplier = hf_multiplier * _texture_multiplier(mask, strengths.even_skin_texture)

    if strengths.contouring > 0:
        for side in ("right", "left"):
            mask = rasterize_mask(cheek_region(landmarks, side), shape, feather=feather, edge=edge)
            lf = _apply_contouring(lf, mask, strengths.contouring)

    return lf, hf_multiplier


def _apply_face_color(
    rgb: np.ndarray, landmarks: FaceLandmarks, strengths: RetouchStrengths, shape: tuple[int, int]
) -> np.ndarray:
    """Eyes/teeth/lips against the final recombined image - see the module note above these
    functions for why they run as a separate pass from the LF/HF skin tools rather than editing
    `lf`. Feather is scaled down relative to the skin tools' - the sclera, tooth, and lip regions
    are all much smaller than a cheek or the whole face oval, so the same absolute feather radius
    would blur away a meaningful fraction of a small region rather than just softening its edge."""
    feather = _feather_radius(landmarks) * strengths.feather_amount * 0.4
    edge = _edge_offset_px(landmarks, strengths.edge_amount)

    if strengths.eye_whiten > 0:
        for side in ("right", "left"):
            sclera, cutouts = eye_sclera_region(landmarks, side)
            mask = rasterize_mask(sclera, shape, cutouts=cutouts, feather=feather, edge=edge)
            rgb = _apply_eye_whiten(rgb, mask, strengths.eye_whiten)

    if strengths.teeth_whiten > 0:
        geometric_mask = rasterize_mask(mouth_interior_region(landmarks), shape, feather=feather, edge=edge)
        tooth_mask = _tooth_submask(rgb, geometric_mask)
        rgb = _apply_teeth_whiten(rgb, tooth_mask, strengths.teeth_whiten)

    if strengths.lip_enhance > 0:
        outer, cutouts = lip_region(landmarks)
        mask = rasterize_mask(outer, shape, cutouts=cutouts, feather=feather, edge=edge)
        rgb = _apply_lip_enhance(rgb, mask, strengths.lip_enhance)

    return rgb


def retouch_faces(
    image: LoadedImage,
    strengths: RetouchStrengths | dict[int, RetouchStrengths],
    *,
    max_faces: int = 32,
    landmarks: list[FaceLandmarks] | None = None,
) -> LoadedImage | None:
    """Returns None if no face is detected - a photo without one is a real case the caller
    should handle, not a hidden failure. A plain RetouchStrengths broadcasts to every detected
    face (Capture One's "all faces" mode, and the only mode possible before this project could
    detect more than one face); a dict[face_index, RetouchStrengths] edits only the listed faces
    - an out-of-range index is a real bug (a UI sending a stale index after re-detection), so it
    raises rather than being silently ignored. `landmarks=` lets a caller that already ran
    detect_landmarks (the API layer, to share one detection across a preview call and an apply
    call) skip re-running it here.

    LF/HF split runs once regardless of face or effect count - it's a property of the whole
    image's frequency content, not per-face state. Each face's masked edits are applied in
    left-to-right order (see face_landmarks._sort_left_to_right); two faces close together in a
    group photo compose sequentially over any overlapping pixels, the same order-dependent way
    Lightroom/Capture One's own stacked local-adjustment masks compose - existing behavior
    extended to N faces, not a new problem multi-face introduces. HF stays untouched except
    where Even Skin's Texture control deliberately scales it (RetouchStrengths.even_skin_texture)."""
    unit = to_unit_float(image.array)[..., :3]
    height, width = unit.shape[:2]

    if landmarks is None:
        rgb_8bit = (np.clip(unit, 0.0, 1.0) * 255 + 0.5).astype(np.uint8)
        landmarks = detect_landmarks(rgb_8bit, max_faces=max_faces)
    if not landmarks:
        return None

    if isinstance(strengths, RetouchStrengths):
        per_face = dict.fromkeys(range(len(landmarks)), strengths)
    else:
        bad_indices = [i for i in strengths if not (0 <= i < len(landmarks))]
        if bad_indices:
            raise ValueError(f"face index out of range (found {len(landmarks)} face(s)): {bad_indices}")
        per_face = strengths

    # one shared blur sigma for the whole image (see retouch_faces' docstring on why LF/HF
    # stays a single pass) - averaged across every detected face rather than privileging
    # whichever face happens to be leftmost, so a group photo with faces at different distances
    # from the camera gets a representative middle-ground radius, not an arbitrary one
    face_widths = [np.linalg.norm(f.subset([234])[0] - f.subset([454])[0]) for f in landmarks]
    lf_sigma = max(2.0, float(np.mean(face_widths)) * 0.05)
    lf = _gaussian_blur(unit, lf_sigma)
    hf = unit - lf
    hf_multiplier = np.ones((height, width), dtype=np.float64)

    for face_index, face_landmarks in enumerate(landmarks):
        face_strengths = per_face.get(face_index)
        if face_strengths is None:
            continue
        lf, hf_multiplier = _apply_face(lf, hf_multiplier, face_landmarks, face_strengths, (height, width))

    result_unit = np.clip(lf + hf * hf_multiplier[..., None], 0.0, 1.0)

    # eyes/teeth/lips run as a second pass, against the recombined image - see _apply_face_color
    for face_index, face_landmarks in enumerate(landmarks):
        face_strengths = per_face.get(face_index)
        if face_strengths is None:
            continue
        result_unit = _apply_face_color(result_unit, face_landmarks, face_strengths, (height, width))
    result_unit = np.clip(result_unit, 0.0, 1.0)

    icc_profile = image.icc_profile
    if icc_profile is None:
        # io.save() requires one - same fallback match_look.py uses for the same reason
        icc_profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

    return LoadedImage(
        array=from_unit_float(result_unit, image.array.dtype),
        icc_profile=icc_profile,
        bit_depth=image.bit_depth,
    )
