"""Layer separation: photo -> per-object RGBA layers + a reconstructed background plate.

Orchestrates ai_prepress.layer_decompose (the remote Qwen-Image-Layered client) - this is the
first features/ module that calls a remote model directly, since face_parsing.py's Modal
precedent was historically wired straight into api/main.py and is now retired from the UI (see
that module's docstring).

Bit-depth handling is the one piece of real logic here: the remote model's own RGB output is
8-bit diffusion-model output, discarded per-layer in favor of the ORIGINAL image's own
full-precision RGB (this project's whole identity is never silently collapsing bit depth, and a
higher-fidelity source already exists for every non-occluded layer pixel). Only the background
plate is genuinely novel content - reconstructed pixels with no source data behind them - so it
stays honestly 8-bit, a documented exception rather than an implicit gap.

Three more passes sit around the main decomposition call, all explicitly ENHANCEMENTS over a
working fallback, never hard dependencies of it - a bad day for any one of them must never take
down the core separation result:
- ai_prepress.object_count (SAM2) estimates how many distinct objects are in the photo BEFORE
  decomposing, so `layers` is sized per-photo instead of a fixed constant. If it fails for any
  reason, falls back to FALLBACK_LAYER_COUNT (today's old hardcoded default).
- ai_prepress.layer_naming (InternVL3.5-2B) suggests a short label per layer AFTER decomposing,
  to pre-fill the UI's rename field. If it fails for any reason, every layer's `suggested_name`
  is just None and the UI falls back to its existing generic "Layer N" placeholder.
- ai_prepress.alpha_refine (guided filter) sharpens every layer's coarse, Lanczos-soft alpha
  against the source RGB's own real edges, scoped to that layer's own coverage region so it never
  reaches across an occlusion boundary. A refinement failure just ships the coarser alpha.

A fourth pass, contamination detection + recursive repair, reuses the naming call above rather
than adding new remote calls of its own for detection: two spatially adjacent layers whose
suggested names are near-duplicates (or whose opaque content near their shared boundary reads as
the same material by a cheap color histogram) look like the same real object over-segmented into
two layers, not two distinct objects. A flagged layer gets ONE retry through Qwen-Image-Layered's
own documented recursive/iterative decomposition feature, asking it to split what the first pass
merged - see _repair_contaminated_layer for why the retry input is flattened onto an opaque
background rather than sent as RGBA, and for the fallback discipline when a retry doesn't help.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import ImageCms

from ai_prepress import alpha_refine, layer_decompose, layer_naming, object_count
from ai_prepress.io import LoadedImage, from_unit_float, to_unit_float

MIN_LAYERS = 2
MAX_LAYERS = 8
# first-cut bounds on the auto-computed layer count, correct against real photos like every
# other untested numeric constant this project has shipped and then measured
FALLBACK_LAYER_COUNT = 4  # today's old hardcoded default - now only reached if object-counting
# itself fails, not the normal path

NAME_ADJACENCY_TOLERANCE = 4  # px - real Qwen alpha masks rarely share an exact boundary pixel
# (soft edges, resize rounding), so bboxes within this small a gap still count as "touching" for
# the contamination check below
COLOR_HISTOGRAM_SIMILARITY_THRESHOLD = 0.9  # cv2.HISTCMP_CORREL, range [-1, 1] - an honest
# placeholder, not yet measured against a real duplicated-layer failure the way
# FALLBACK_LAYER_COUNT eventually was: no such case has been reproduced against a live
# deployment yet, so this is a conservative first guess, correct it once one has been
RECURSIVE_REPAIR_LAYERS = 3  # background + 2 foreground sub-layers. Asking for literally 2 (the
# smallest example this feature's own spec suggested) would only ever come back as background +
# ONE foreground layer - layer 0 is always the background plate, see
# deploy/layer_separation.py's documented bottom-to-top stacking convention - which can't split
# anything into more than one piece. 3 is the minimum total that can actually produce a real split
MIN_REPAIR_SUBLAYERS = 2  # fewer usable sub-layers than this means the retry didn't separate the
# contaminated content into distinct pieces - nothing to prefer over the original layer
MAX_REPAIR_ATTEMPTS = 2  # each repair is its own serial multi-minute remote call inside one
# already-slow synchronous request, and every layer in a contaminating PAIR gets flagged (both
# sides), so an unbounded loop could turn one request into several times its normal latency


@dataclass
class SeparatedLayer:
    image: LoadedImage  # (H, W, 4), dtype matches the source image's own bit depth
    bbox: tuple[int, int, int, int]  # (x0, y0, x1, y1), derived from the alpha mask's own extent
    suggested_name: str | None = None  # from layer_naming - None if naming failed or wasn't run
    contamination_flag: bool = False  # True when this layer looks like the same real object
    # duplicated across two adjacent layers - see _flag_contaminated_layers. Only ever set on a
    # layer going INTO the recursive repair pass; a successfully repaired layer is replaced
    # outright by fresh (unflagged) sub-layers rather than mutated in place


@dataclass
class SeparatedLayers:
    background: LoadedImage  # (H, W, 3), 8-bit - inherently model-generated, see module docstring
    layers: list[SeparatedLayer]
    requested_layers: int  # the actual N sent to the model - not len(layers)+1, which is what
    # came BACK; surfaced for transparency while MIN_LAYERS/MAX_LAYERS/the count formula still
    # need tuning against real photos, see _estimate_layer_count


BBOX_ALPHA_THRESHOLD = 127  # majority-opaque, not "any nonzero" - see below. Not underscore-
# prefixed since api/main.py's alpha_coverage stat reuses it for the same reason.

def _bbox_from_alpha(alpha: np.ndarray) -> tuple[int, int, int, int]:
    """A real deployment against Qwen-Image-Layered (not a synthetic test fixture) showed the
    model's own alpha output isn't clean binary - it carries widespread low-level noise (values
    1-10) spread across nearly the whole frame, not just the object's own soft edge. `alpha > 0`
    picked that noise up and blew the bbox out to cover almost the entire image on a real photo.
    A majority-opacity threshold (>127) tracks the object's actual visible extent instead, while
    compositing (separate_layers below) still uses the full soft alpha unclamped, since edge
    softness is wanted there - this threshold is only for "where is this layer's content," not
    for the alpha values a caller actually gets back."""
    coords = np.argwhere(alpha > BBOX_ALPHA_THRESHOLD)
    if coords.size == 0:
        return (0, 0, 0, 0)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    return (int(x0), int(y0), int(x1) + 1, int(y1) + 1)


def _estimate_layer_count(image: LoadedImage) -> int:
    try:
        estimated_objects = object_count.count_objects(image)
    except Exception:
        # object-counting is an enhancement over a fixed default, never a hard dependency of
        # separation itself - a down/erroring endpoint must not fail the whole request
        return FALLBACK_LAYER_COUNT
    return max(MIN_LAYERS, min(MAX_LAYERS, estimated_objects + 1))  # +1 for the background plate


def _composite_layer(
    alpha: np.ndarray,
    source_rgb: np.ndarray,
    source_rgb_unit: np.ndarray,
    dtype: np.dtype,
    icc_profile: bytes,
    bit_depth: int,
) -> SeparatedLayer:
    """Refine the coarse alpha against the source's own real edges before compositing - see
    ai_prepress.alpha_refine's module docstring for why (Lanczos-upsampled diffusion output is
    soft by construction, not because the real object edge is). Left unscoped (no explicit
    visible_region): alpha_refine's own default IS a layer's own coverage region, thresholded and
    dilated from this same coarse alpha - exactly the scoping this needs, not a placeholder being
    reused past its intent. Wrapped the same way every other enhancement in this module is: a
    guided-filter bug or a degenerate tiny crop must not fail an otherwise-working separation, it
    should just ship the coarser alpha instead."""
    try:
        alpha = alpha_refine.refine_alpha(alpha, source_rgb)
    except Exception:
        pass
    alpha_unit = to_unit_float(alpha)
    rgba_unit = np.dstack([source_rgb_unit, alpha_unit])
    layer_image = LoadedImage(
        array=from_unit_float(rgba_unit, dtype),
        icc_profile=icc_profile,
        bit_depth=bit_depth,
    )
    return SeparatedLayer(image=layer_image, bbox=_bbox_from_alpha(alpha))


def _bboxes_adjacent(a: tuple[int, int, int, int], b: tuple[int, int, int, int], tolerance: int) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    if ax1 <= ax0 or ay1 <= ay0 or bx1 <= bx0 or by1 <= by0:
        return False  # a degenerate (empty) bbox has nothing to be adjacent to
    return not (ax1 + tolerance < bx0 or bx1 + tolerance < ax0 or ay1 + tolerance < by0 or by1 + tolerance < ay0)


def _expand_bbox(bbox: tuple[int, int, int, int], tolerance: int, width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return (max(0, x0 - tolerance), max(0, y0 - tolerance), min(width, x1 + tolerance), min(height, y1 + tolerance))


def _intersect_bbox(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return (x0, y0, x1, y1)


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _names_are_near_duplicate(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    return _normalize_name(a) == _normalize_name(b)


def _masked_histogram(layer: SeparatedLayer, source_rgb_uint8: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray | None:
    x0, y0, x1, y1 = region
    crop_rgb = source_rgb_uint8[y0:y1, x0:x1]
    crop_alpha_uint8 = from_unit_float(to_unit_float(layer.image.array[y0:y1, x0:x1, 3]), np.uint8)
    mask = (crop_alpha_uint8 > BBOX_ALPHA_THRESHOLD).astype(np.uint8)
    if mask.sum() == 0:
        return None
    hist = cv2.calcHist([crop_rgb], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def _color_similarity(
    layer_a: SeparatedLayer, layer_b: SeparatedLayer, source_rgb_uint8: np.ndarray, region: tuple[int, int, int, int]
) -> float | None:
    # source RGB is identical everywhere for every layer (all of them composite against the SAME
    # full-precision source, see _composite_layer) - the per-layer alpha mask fed to
    # _masked_histogram is what actually distinguishes "this layer's own material" from "that
    # layer's own material" within the shared region, not the RGB array itself
    x0, y0, x1, y1 = region
    if x1 <= x0 or y1 <= y0:
        return None
    hist_a = _masked_histogram(layer_a, source_rgb_uint8, region)
    hist_b = _masked_histogram(layer_b, source_rgb_uint8, region)
    if hist_a is None or hist_b is None:
        return None
    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))


def _flag_contaminated_layers(layers: list[SeparatedLayer], source_rgb: np.ndarray) -> None:
    """Two adjacent layers that are really the same over-segmented object tend to show it two
    ways: InternVL names them near-identically (reusing the naming call already made above, not a
    second remote call), or - when naming didn't catch it, or failed outright - their own opaque
    pixels near the shared boundary read as the same material. Either signal is enough to flag
    both layers of the pair for the recursive repair pass below."""
    height, width = source_rgb.shape[:2]
    source_rgb_uint8 = from_unit_float(to_unit_float(source_rgb), np.uint8)
    for i in range(len(layers)):
        for j in range(i + 1, len(layers)):
            layer_a, layer_b = layers[i], layers[j]
            if not _bboxes_adjacent(layer_a.bbox, layer_b.bbox, NAME_ADJACENCY_TOLERANCE):
                continue
            contaminated = _names_are_near_duplicate(layer_a.suggested_name, layer_b.suggested_name)
            if not contaminated:
                region = _intersect_bbox(
                    _expand_bbox(layer_a.bbox, NAME_ADJACENCY_TOLERANCE, width, height),
                    _expand_bbox(layer_b.bbox, NAME_ADJACENCY_TOLERANCE, width, height),
                )
                similarity = _color_similarity(layer_a, layer_b, source_rgb_uint8, region)
                contaminated = similarity is not None and similarity >= COLOR_HISTOGRAM_SIMILARITY_THRESHOLD
            if contaminated:
                layer_a.contamination_flag = True
                layer_b.contamination_flag = True


def _flatten_onto_mid_gray(rgba: np.ndarray) -> np.ndarray:
    unit = to_unit_float(rgba)
    rgb, alpha = unit[..., :3], unit[..., 3:4]
    flattened_unit = rgb * alpha + 0.5 * (1 - alpha)
    return from_unit_float(flattened_unit, np.uint8)


def _repair_contaminated_layer(
    layer: SeparatedLayer,
    image: LoadedImage,
    endpoint: str | None,
    icc_profile: bytes,
) -> list[SeparatedLayer] | None:
    """Re-submit one flagged layer's own content to Qwen-Image-Layered via its documented
    recursive/iterative decomposition, asking it to split what the first pass merged into one
    object. Returns None whenever the retry doesn't clearly help - the caller then keeps the
    original layer as-is - rather than ever silently losing content: a worse-but-present layer
    beats a missing one. Raising is also fine here; the caller catches it with the same fallback.

    The crop is flattened onto an opaque mid-gray background, never sent as RGBA - a real, open,
    unanswered GitHub issue on the model's own repo (QwenLM/Qwen-Image-Layered #17) asks how
    alpha-channel input is handled and has had no answer for months, so this project treats
    transparent input as unsupported rather than assumed to work. Mid-gray rather than the
    surrounding source pixels' own average: an averaged background colour that happens to land
    close to part of the contaminated content's own palette (an orange composited onto an
    averaged table-brown, say) would quietly reintroduce the same low-contrast blending this
    repair pass exists to fix.
    """
    height, width = image.array.shape[:2]
    x0, y0, x1, y1 = layer.bbox
    if x1 <= x0 or y1 <= y0:
        return None  # a degenerate bbox has no content worth retrying

    cropped_rgba = layer.image.array[y0:y1, x0:x1]
    flattened_rgb = _flatten_onto_mid_gray(cropped_rgba)
    crop_image = LoadedImage(array=flattened_rgb, icc_profile=icc_profile, bit_depth=8)

    prompt = None
    if layer.suggested_name:
        # generic phrasing, not per-photo semantic knowledge this layer of code doesn't have -
        # reuses the one label layer_naming already produced for this layer. Matches the model's
        # own documented prompt behavior (github.com/QwenLM/Qwen-Image-Layered's README: "The
        # text prompt is intended to describe the overall content of the input image - including
        # elements that may be partially occluded")
        prompt = f"the {layer.suggested_name}, including any parts hidden by another object"

    retry = layer_decompose.decompose_layers(
        crop_image, endpoint=endpoint, layers=RECURSIVE_REPAIR_LAYERS, prompt=prompt
    )

    usable_alphas = [alpha for alpha in retry.layer_alphas if (alpha > BBOX_ALPHA_THRESHOLD).any()]
    if len(usable_alphas) < MIN_REPAIR_SUBLAYERS:
        # not an improvement - either the retry collapsed back into one blob, or came back mostly
        # empty, a known failure mode of asking for more layers than the content actually
        # supports (this project's own art.jpg/layers=8 measurement: 2 of 7 layers came back at
        # exactly 0.0 alpha coverage, not a finer real split)
        return None

    source_rgb = image.array[..., :3]
    source_rgb_unit = to_unit_float(source_rgb)
    repaired = []
    for crop_alpha in usable_alphas:
        full_alpha = np.zeros((height, width), dtype=np.uint8)
        full_alpha[y0:y1, x0:x1] = crop_alpha
        repaired.append(
            _composite_layer(full_alpha, source_rgb, source_rgb_unit, image.array.dtype, icc_profile, image.bit_depth)
        )
    return repaired


def separate_layers(image: LoadedImage, endpoint: str | None = None) -> SeparatedLayers:
    """Returns every remote-detected layer as a full-precision RGBA LoadedImage, plus the
    background plate. Never returns None - unlike retouch_faces' "no face found" case, a photo
    with nothing separable just comes back with an empty `layers` list, still a valid result."""
    layers_requested = _estimate_layer_count(image)
    decomposed = layer_decompose.decompose_layers(image, endpoint=endpoint, layers=layers_requested)

    icc_profile = image.icc_profile
    if icc_profile is None:
        # io.save() requires one - same fallback retouch_faces.py and match_look.py use
        icc_profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

    source_rgb = image.array[..., :3]
    source_rgb_unit = to_unit_float(source_rgb)

    layers = [
        _composite_layer(alpha, source_rgb, source_rgb_unit, image.array.dtype, icc_profile, image.bit_depth)
        for alpha in decomposed.layer_alphas
    ]

    try:
        suggested_names = layer_naming.name_layers([layer.image for layer in layers])
    except Exception:
        # naming is cosmetic - it only ever pre-fills a rename field the UI already has, so a
        # failure here must not take down an otherwise-successful separation
        suggested_names = [None] * len(layers)
    for layer, name in zip(layers, suggested_names):
        layer.suggested_name = name

    try:
        _flag_contaminated_layers(layers, source_rgb)
    except Exception:
        # contamination detection only gates the repair pass below - a false negative here just
        # means that pass never fires for this request, never a failed request
        pass

    repair_attempts = 0
    repaired_layers: list[SeparatedLayer] = []
    for layer in layers:
        if layer.contamination_flag and repair_attempts < MAX_REPAIR_ATTEMPTS:
            repair_attempts += 1
            try:
                replacement = _repair_contaminated_layer(layer, image, endpoint, icc_profile)
            except Exception:
                # the recursive repair call is a best-effort improvement over an already-flagged
                # layer, never a hard dependency - anything from a down endpoint to a malformed
                # retry response must fall back to the original layer, not lose it
                replacement = None
            repaired_layers.extend(replacement if replacement is not None else [layer])
        else:
            repaired_layers.append(layer)
    layers = repaired_layers

    background = LoadedImage(array=decomposed.background, icc_profile=icc_profile, bit_depth=8)
    return SeparatedLayers(background=background, layers=layers, requested_layers=layers_requested)
