"""Layer separation: photo -> per-object RGBA layers + a reconstructed background plate.

THE PRIMARY PATH (Phase B, this session) is per-instance: ai_prepress.layer_naming lists candidate
object names for the whole photo -> ai_prepress.object_detect (Grounding DINO) turns those names
into real per-object boxes -> ai_prepress.object_segment (box-prompted SAM2) turns each box into a
pixel-accurate mask -> ai_prepress.instance_matte (BiRefNet_HR) re-mattes each mask's own bbox crop
for a real high-resolution alpha -> ai_prepress.background_inpaint (LaMa) fills the union of every
instance mask to reconstruct the plate. See _separate_layers_per_instance for the full pipeline and
_mask_iou's call site for the one real failure mode found while building it (BiRefNet matting the
WRONG object inside an overlapping crop).

Composition decision with the older Qwen-Image-Layered path below (explicit, asked for by this
session's own spec, not left implicit): the per-instance pipeline is ALWAYS TRIED FIRST, and
Qwen-Image-Layered runs only as a FALLBACK when the per-instance result comes back implausible - no
candidate objects, no surviving detections, or every detected instance failing to segment/matte
into anything usable (see separate_layers() below for where that fallback actually happens). Reasons
this and not "run both, pick a winner" or "replace Qwen outright":
- Cost/latency is wildly asymmetric. The per-instance pipeline is four fast synchronous Modal
  calls, each well inside a 150s web-endpoint ceiling. Qwen-Image-Layered is a genuinely
  multi-minute diffusion call behind its own submit/poll split (see layer_decompose.py's own
  docstring for why). Running both on every request to "pick a winner" would pay Qwen's full cost
  on every photo, including the vast majority where the cheap path already works - indefensible
  given the cost gap.
- Discrete real-world objects (the common prepress case: products, people, furniture) are exactly
  what an open-vocabulary detector is built for, and the per-instance path gives crisper masks and
  HONEST OCCLUSION HOLES (an occluded layer's alpha is just 0 wherever another instance's own mask
  covers it - nothing reconstructs hidden content, because nothing in this path is asked to) where
  Qwen's holistic diffusion decomposition tends to hallucinate plausible-looking-but-wrong content
  instead. That's a real quality win for the common case, not just a cost one.
- A flat graphic, poster, or other photo with no clear discrete objects is exactly where Grounding
  DINO has nothing to latch onto - zero (or near-zero) real detections - and Qwen's holistic
  decomposition is a genuinely better fit there. Falling back lazily, only when the cheap path
  actually comes up empty, gets that photo the right treatment without ever paying Qwen's cost on
  a photo the cheap path already handled well.

Everything below this point that isn't part of the new pipeline - Qwen-Image-Layered itself, its
automatic layer-count estimate, its automatic naming pass, its contamination-detection and
recursive-repair passes - stays exactly as Phase A shipped it, now serving the fallback path
instead of the primary one. None of that logic changed; only which path calls it first did.

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

Two more passes sit around the main decomposition call, both explicitly ENHANCEMENTS over a
working fallback, never hard dependencies of it - a bad day for either one must never take down
the core separation result:
- ai_prepress.object_count (SAM2) estimates how many distinct objects are in the photo BEFORE
  decomposing, so `layers` is sized per-photo instead of a fixed constant. If it fails for any
  reason, falls back to FALLBACK_LAYER_COUNT (today's old hardcoded default).
- ai_prepress.layer_naming (InternVL3.5-2B) suggests a short label per layer AFTER decomposing,
  to pre-fill the UI's rename field. If it fails for any reason, every layer's `suggested_name`
  is just None and the UI falls back to its existing generic "Layer N" placeholder.

ai_prepress.alpha_refine (guided filter) exists and is independently tested, but is deliberately
NOT wired in here. It was briefly wired into _composite_layer and then measured against a real
Qwen-Image-Layered coarse alpha on an actual hair edge (portrait.png, the guided filter's own
DEFAULT_RADIUS/DEFAULT_EPS comment had flagged this exact validation as still outstanding): the
naive 10%-90%-crossing measurement got WORSE at every (radius, eps) combination tried, including
the shipped defaults and a trimap-style band limiting the filter to a narrow region around the
coarse edge (median 14px unrefined vs. 27px best-case refined, 148px worst-case). Row-level
inspection shows this isn't a wider true edge - it's real hair texture in the guide RGB reading as
edge signal throughout what should be a flat opaque interior, which a boundary-crossing metric
reads as a much wider transition than what actually changed. Either way, no setting was found that
both changed the coarse alpha meaningfully and didn't hurt this measurement, so there's no evidence
left to justify shipping it. Re-wiring this needs a genuinely different approach (or a different
real edge case to validate against), not a parameter tweak - see alpha_refine.py's own docstring.

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

from ai_prepress import (
    background_inpaint,
    instance_matte,
    layer_decompose,
    layer_naming,
    object_count,
    object_detect,
    object_segment,
)
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
    # need tuning against real photos, see _estimate_layer_count. In the per-instance path this is
    # the deduped detection count instead - the closest equivalent "how many objects this path
    # expected to separate," with len(layers) still possibly smaller if one of them failed to
    # segment or matte into anything usable
    separation_path: str  # "per-instance" or "qwen" - which pipeline actually produced this
    # result, see the module docstring's composition decision and separate_layers() below


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
    """Composite one layer's alpha against the source's own full-precision RGB.

    Does not call ai_prepress.alpha_refine - see this module's own docstring for the real-photo
    measurement that found it regresses hair-edge sharpness rather than improving it."""
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


def _separate_layers_qwen(image: LoadedImage, endpoint: str | None = None) -> SeparatedLayers:
    """The FALLBACK path - see module docstring for the composition decision. Returns every
    remote-detected layer as a full-precision RGBA LoadedImage, plus the background plate. Never
    returns None - unlike retouch_faces' "no face found" case, a photo with nothing separable just
    comes back with an empty `layers` list, still a valid result."""
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
    return SeparatedLayers(
        background=background, layers=layers, requested_layers=layers_requested, separation_path="qwen"
    )


# --- per-instance pipeline (PRIMARY path) -------------------------------------------------------


PER_INSTANCE_CLIENT_TIMEOUT = 120.0  # every new client's own default (60-120s) times the Modal
# request itself, not a cold container spin-up on top of it - confirmed against a real deploy: a
# request made ~10 minutes after the previous one hit object_detect's scaledown_window=120s (see
# deploy/object_detect.py), the container had scaled down, and its default 60s client timeout was
# too tight for a genuine cold start, silently falling this whole path back to Qwen. A generous
# shared timeout here costs nothing on the (common) warm-container case and buys real headroom on
# the (real, seen) cold one - cheaper than debugging a silent fallback again.

MATTE_SAM_IOU_THRESHOLD = 0.5  # first-cut guard, not yet measured against a real BiRefNet/SAM2
# disagreement the way FALLBACK_LAYER_COUNT eventually was (same honest-placeholder spirit as
# COLOR_HISTOGRAM_SIMILARITY_THRESHOLD above). Exists for a real failure mode found while building
# this: BiRefNet_HR mattes WHATEVER salient object is in the crop it's given, with no idea which
# instance the crop was meant for - two overlapping detection boxes (a book box that also contains
# part of an orange, say) can come back with BiRefNet confidently matting the WRONG object. Low
# agreement between BiRefNet's own alpha and the SAM2 mask that produced the crop in the first
# place is the signal that happened - falls back to the correctly-scoped (if harder-edged) SAM2
# mask rather than shipping a confidently-wrong alpha.

DETECTION_DEDUP_IOU_THRESHOLD = 0.6  # same honest-placeholder spirit - Grounding DINO applies no
# NMS across DIFFERENT text terms describing the same real object (confirmed directly against a
# real deploy: the candidate-name listing call can propose close synonyms like "book" and "brown
# book" for one physical book), so this project's own client does that suppression instead of
# trusting the model's per-term boxes to already be deduplicated


def _box_area(box: list[float]) -> float:
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _box_iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = _box_area(a) + _box_area(b) - intersection
    return intersection / union if union > 0 else 0.0


def _dedupe_detections(detections: list[dict], iou_threshold: float = DETECTION_DEDUP_IOU_THRESHOLD) -> list[dict]:
    """Greedy NMS across ALL detections regardless of label text - see DETECTION_DEDUP_IOU_THRESHOLD
    for why label-agnostic matters here. Highest-score box of a duplicate pair wins."""
    ordered = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept: list[dict] = []
    for candidate in ordered:
        if not any(_box_iou(candidate["box"], other["box"]) >= iou_threshold for other in kept):
            kept.append(candidate)
    return kept


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0  # a genuine shape mismatch is itself disagreement - never crash comparing them
    a_bin = a > BBOX_ALPHA_THRESHOLD
    b_bin = b > BBOX_ALPHA_THRESHOLD
    union = np.logical_or(a_bin, b_bin).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a_bin, b_bin).sum()) / float(union)


def _build_detection_prompt(names: list[str]) -> str:
    return ". ".join(name.strip().rstrip(".") for name in names if name.strip()) + "."


def _separate_layers_per_instance(image: LoadedImage) -> SeparatedLayers | None:
    """The PRIMARY path - see module docstring for the full pipeline shape and the composition
    decision. Returns None whenever the result would be implausible (no candidate objects, no
    surviving detections, or every detected instance failing to segment/matte into anything
    usable) - the caller (separate_layers, below) falls back to Qwen-Image-Layered in that case,
    never raises past this function for a "nothing here" outcome that isn't a real error.

    Occlusion is handled by omission, not reconstruction: nothing in this pipeline is ever asked
    to guess what's under another object, so an occluded instance's own mask (and therefore its
    final alpha) is just 0 wherever another instance's mask already covers that pixel - an honest
    hole, not fabricated content. This is the documented, correct behavior this session's own
    research called for, not a gap to close later.
    """
    try:
        candidate_names = layer_naming.list_candidate_objects(image, timeout=PER_INSTANCE_CLIENT_TIMEOUT)
    except Exception:
        return None
    if not candidate_names:
        return None

    try:
        detections = object_detect.detect_objects(
            image, _build_detection_prompt(candidate_names), timeout=PER_INSTANCE_CLIENT_TIMEOUT
        )
    except Exception:
        return None
    detections = _dedupe_detections(detections)
    if not detections:
        return None

    try:
        masks = object_segment.segment_boxes(
            image, [d["box"] for d in detections], timeout=PER_INSTANCE_CLIENT_TIMEOUT
        )
    except Exception:
        return None

    icc_profile = image.icc_profile
    if icc_profile is None:
        icc_profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

    source_rgb = image.array[..., :3]
    source_rgb_unit = to_unit_float(source_rgb)
    source_rgb_uint8 = from_unit_float(source_rgb_unit, np.uint8)
    height, width = source_rgb.shape[:2]

    instance_bboxes = [_bbox_from_alpha(mask) for mask in masks]
    crops: list[np.ndarray] = []
    crop_source_indices: list[int] = []
    for i, bbox in enumerate(instance_bboxes):
        x0, y0, x1, y1 = bbox
        if x1 <= x0 or y1 <= y0:
            continue  # SAM2 came back with nothing usable for this box - drop the instance rather
            # than fail the whole request over one bad detection
        crops.append(source_rgb_uint8[y0:y1, x0:x1])
        crop_source_indices.append(i)

    if not crops:
        return None

    try:
        matte_alphas: list[np.ndarray | None] = list(
            instance_matte.matte_crops(crops, timeout=PER_INSTANCE_CLIENT_TIMEOUT)
        )
    except Exception:
        # matting is this path's own quality pass over an already-real SAM2 mask, not a hard
        # dependency of it - see MATTE_SAM_IOU_THRESHOLD's fallback below for the same reasoning
        # applied per-instance, not just on a total endpoint failure
        matte_alphas = [None] * len(crops)

    layers: list[SeparatedLayer] = []
    for crop_index, source_index in enumerate(crop_source_indices):
        x0, y0, x1, y1 = instance_bboxes[source_index]
        sam_alpha_crop = masks[source_index][y0:y1, x0:x1]
        matte_alpha_crop = matte_alphas[crop_index]

        if matte_alpha_crop is not None and _mask_iou(matte_alpha_crop, sam_alpha_crop) >= MATTE_SAM_IOU_THRESHOLD:
            chosen_alpha_crop = matte_alpha_crop
        else:
            chosen_alpha_crop = sam_alpha_crop

        full_alpha = np.zeros((height, width), dtype=np.uint8)
        full_alpha[y0:y1, x0:x1] = chosen_alpha_crop
        layer = _composite_layer(
            full_alpha, source_rgb, source_rgb_unit, image.array.dtype, icc_profile, image.bit_depth
        )
        layer.suggested_name = detections[source_index]["label"]
        layers.append(layer)

    layers = [layer for layer in layers if layer.bbox != (0, 0, 0, 0)]
    if not layers:
        return None  # every detected instance failed to segment/matte into anything usable

    union_mask = np.zeros((height, width), dtype=np.uint8)
    for mask in masks:
        union_mask = np.maximum(union_mask, mask)

    try:
        background_rgb = background_inpaint.inpaint_background(
            source_rgb_uint8, union_mask, timeout=PER_INSTANCE_CLIENT_TIMEOUT
        )
    except Exception:
        # inpainting is an enhancement over the raw (holed) plate, never a hard dependency of an
        # otherwise-successful separation - matches object_count/layer_naming's own fallback
        # discipline elsewhere in this file
        background_rgb = source_rgb_uint8

    background = LoadedImage(array=background_rgb, icc_profile=icc_profile, bit_depth=8)
    return SeparatedLayers(
        background=background, layers=layers, requested_layers=len(detections), separation_path="per-instance"
    )


def separate_layers(image: LoadedImage, endpoint: str | None = None) -> SeparatedLayers:
    """Photo -> per-object RGBA layers + a reconstructed background plate.

    Tries the per-instance pipeline FIRST and falls back to Qwen-Image-Layered only when that
    comes back implausible - see the module docstring for the full reasoning behind this order.
    `endpoint` only ever overrides the Qwen-Image-Layered endpoint (the fallback path); the
    per-instance clients each resolve their own endpoint from their own env var, same as
    object_count/layer_naming already do elsewhere in this file."""
    per_instance = _separate_layers_per_instance(image)
    if per_instance is not None:
        return per_instance
    return _separate_layers_qwen(image, endpoint=endpoint)
