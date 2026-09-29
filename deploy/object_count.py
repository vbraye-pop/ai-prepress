"""Modal deployment for a cheap pre-flight object count, used to size Qwen-Image-Layered's
`layers` parameter automatically instead of a hardcoded constant (see features/layer_separation.py
and deploy/layer_separation.py).

Model: SAM2 (facebookresearch/sam2), Apache 2.0 - confirmed directly from the repo's own LICENSE
file, not assumed from the SAM/SAM2 name. SAM3 was checked first (this project's own earlier
planning notes named "SAM3 + BiRefNet" as the original segmentation backend) and rejected: it
ships under Meta's own custom "SAM License" with binding acceptable-use restrictions, the same
category of legal-review burden this project has avoided before (Gemma, BSL).

Uses `SAM2AutomaticMaskGenerator` - a genuine "segment everything, no text/object prompt needed"
mode, exactly what a pure object COUNT needs (no vocabulary, no concept-prompting). The tiny
checkpoint (`sam2.1_hiera_tiny.pt`) is used deliberately - this is a coarse pre-flight count, not
pixel-accurate segmentation, and needs to stay cheap relative to the ~2 minute main decomposition
call it precedes.

Real API (from the repo's own automatic_mask_generator.py, not guessed): the generator's own
constructor already supports `min_mask_region_area` and `stability_score_thresh`, and does its
own internal NMS - but that internal NMS only suppresses near-duplicate masks from the SAME
sampled point neighbourhood, not the cross-scale duplicates points_per_side>1 always produces
(a whole object AND a sub-region of it both surviving as separate masks). filter_masks() below is
a second, explicit pass over the generator's raw output list for exactly that gap - see its own
docstring. min_mask_region_area is computed as 2% of the input image's own pixel area (not a
fixed pixel count - a fixed 100px^2 was the first version of this file and badly over-segmented a
real ~3.4-megapixel test photo, object_count=17 for a still life with ~4 real objects, see count()
for the fix). stability_score_thresh 0.90 (the library's own default, 0.95, is tuned for precise
segmentation and would likely under-count real objects that have a genuinely fuzzy/soft edge).

Deploy: uv run --group deploy modal deploy deploy/object_count.py

Fast enough (SAM2's own per-point speed + a single shared image encoder pass) to be a PLAIN
synchronous web endpoint - unlike deploy/layer_separation.py's submit/result split, which exists
specifically because THAT call can exceed Modal's 150-second web-endpoint ceiling. No Cls.from_name
machinery needed here.

torch/sam2/PIL are only ever imported inside method bodies below, never at module level - same
reasoning as every other deploy script in this project (the local venv doesn't carry torch).
numpy is the deliberate exception: it's a main dependency (not a deploy-only one), and importing
it at module level is what lets filter_masks() below be imported and unit-tested locally without
ever touching torch/sam2.
"""

import io

import modal
import numpy as np
from fastapi import File, Response, UploadFile
from fastapi.responses import JSONResponse

CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
CHECKPOINT_PATH = "/opt/sam2/checkpoints/sam2.1_hiera_tiny.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

# A contained/duplicate pair scores >= these on pixel overlap of the ACTUAL boolean masks, not
# their bboxes - two adjacent objects (e.g. an orange sitting in front of a table) routinely have
# one bbox fully inside the other while their real masks barely touch, so bbox-only containment
# would wrongly merge them. 0.9 leaves real partial occlusion (an object over half hidden behind
# another) uncounted as a duplicate.
MASK_CONTAINMENT_THRESH = 0.9
MASK_IOU_THRESH = 0.9

# A mask whose bbox touches the image border on at least two distinct sides AND covers more than
# this fraction of the frame is treated as background fill (sky/grass/wall/table), not a discrete
# object. Two-sides-or-more (not one) is deliberate: a standing person in a portrait photo touches
# only the bottom edge and can still legitimately cover close to half the frame - requiring a
# second edge is what keeps that case from being cut alongside real backgrounds, which typically
# span two or more edges (a lawn spans left+right+bottom, a sky spans left+right+top).
BACKGROUND_MAX_AREA_FRACTION = 0.45
BORDER_TOUCH_MIN_EDGES = 2
# generator output can leave a few px of margin around a true edge-to-edge region (min_mask_region_
# area's own connected-component pass), so "touches the border" allows a small pixel tolerance
# instead of requiring an exact 0.
BORDER_TOUCH_TOLERANCE_PX = 2

app = modal.App("ai-prepress-object-count")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl")
    .pip_install("torch", "torchvision", "pillow", "numpy", "fastapi", "python-multipart")
    .run_commands(
        # cloned to /opt, not /root - sam2's own build_sam.py refuses to import if the current
        # working directory is the PARENT of where the repo lives (a real runtime check in its
        # source, not a docs warning), and Modal's containers default to running from /root
        "git clone --depth 1 https://github.com/facebookresearch/sam2.git /opt/sam2",
        "cd /opt/sam2 && pip install -e .",
        f"mkdir -p /opt/sam2/checkpoints && curl -L -o {CHECKPOINT_PATH} {CHECKPOINT_URL}",
    )
)


def _bbox_edges(bbox: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    return x, y, x + w, y + h


def _bbox_overlaps(bbox_a: list[float], bbox_b: list[float]) -> bool:
    ax1, ay1, ax2, ay2 = _bbox_edges(bbox_a)
    bx1, by1, bx2, by2 = _bbox_edges(bbox_b)
    return ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2


def _border_edges_touched(bbox: list[float], width: int, height: int, tolerance: int) -> int:
    x1, y1, x2, y2 = _bbox_edges(bbox)
    return (
        (x1 <= tolerance)
        + (y1 <= tolerance)
        + (x2 >= width - tolerance)
        + (y2 >= height - tolerance)
    )


def drop_background_masks(
    masks: list[dict],
    width: int,
    height: int,
    max_area_fraction: float = BACKGROUND_MAX_AREA_FRACTION,
    min_border_edges: int = BORDER_TOUCH_MIN_EDGES,
    border_tolerance: int = BORDER_TOUCH_TOLERANCE_PX,
) -> list[dict]:
    """Drop masks that read as background fill rather than a discrete object: touching two or
    more image edges AND covering more than max_area_fraction of the frame. Requiring the bbox
    field (not the segmentation array) keeps this pass cheap - it runs before the O(n^2)
    dedupe_overlapping_masks() below, so it also shrinks that pass's input."""
    total_area = width * height
    if total_area <= 0:
        return list(masks)
    kept = []
    for m in masks:
        edges_touched = _border_edges_touched(m["bbox"], width, height, border_tolerance)
        area_fraction = m["area"] / total_area
        if edges_touched >= min_border_edges and area_fraction > max_area_fraction:
            continue
        kept.append(m)
    return kept


def _pixel_containment_and_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[float, float]:
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    area_a = int(mask_a.sum())
    area_b = int(mask_b.sum())
    smaller = min(area_a, area_b)
    union = area_a + area_b - intersection
    containment = intersection / smaller if smaller > 0 else 0.0
    iou = intersection / union if union > 0 else 0.0
    return containment, iou


def dedupe_overlapping_masks(
    masks: list[dict],
    containment_thresh: float = MASK_CONTAINMENT_THRESH,
    iou_thresh: float = MASK_IOU_THRESH,
) -> list[dict]:
    """Merge masks that are almost certainly the same real object at two granularities (a whole
    object and a sub-region of it, both sampled by different grid points). Containment/IoU are
    measured on the actual `segmentation` boolean arrays, not on bboxes - two adjacent, unrelated
    objects (e.g. an orange sitting in front of a table) routinely have one bbox fully inside the
    other while their real masks barely overlap, so bbox-only containment would wrongly merge
    them. bbox overlap here is only a cheap pre-screen before the pixel comparison. Masks are
    processed largest-area first and a mask is dropped only when it overlaps an already-kept
    (therefore larger-or-equal) mask, so the survivor of any merged pair is the more complete
    one."""
    ordered = sorted(masks, key=lambda m: m["area"], reverse=True)
    kept = []
    for m in ordered:
        duplicate_of_kept = False
        for k in kept:
            if not _bbox_overlaps(m["bbox"], k["bbox"]):
                continue
            containment, iou = _pixel_containment_and_iou(m["segmentation"], k["segmentation"])
            if containment >= containment_thresh or iou >= iou_thresh:
                duplicate_of_kept = True
                break
        if not duplicate_of_kept:
            kept.append(m)
    return kept


def filter_masks(masks: list[dict], width: int, height: int) -> list[dict]:
    """The full post-filter pass over SAM2AutomaticMaskGenerator's raw output, applied before
    counting survivors. Background drop runs first since it shrinks the list dedupe has to
    compare pairwise."""
    masks = drop_background_masks(masks, width, height)
    masks = dedupe_overlapping_masks(masks)
    return masks


@app.function(image=image, gpu="T4", scaledown_window=300)
@modal.fastapi_endpoint(method="POST")
async def count(file: UploadFile = File(...)) -> Response:
    import torch
    from PIL import Image
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    contents = await file.read()
    source = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
    height, width = source.shape[0], source.shape[1]

    # A fixed pixel-area threshold (the first version of this endpoint used 100px^2) behaves
    # wildly inconsistently across photo resolutions - confirmed empirically against two real
    # photos, not assumed: a 4-real-object still life came back object_count=17 at the library's
    # default points_per_side=32, and a single-person photo with grass and a background crowd
    # came back 67. SAM2's automatic mode doesn't distinguish "discrete foreground object" from
    # "any locally coherent texture patch" (a chunk of lawn, a window pane, a distant figure are
    # all individually large and stable enough to survive an area filter alone) on its own -
    # points_per_side=8 plus filter_masks() below (background-fill drop, cross-scale dedupe) is
    # what brings that under control now; see filter_masks()'s own docstring for how.
    # MIN_LAYERS/MAX_LAYERS clamping in features/layer_separation.py remains the safety net for
    # whatever this count still gets wrong end to end.
    min_area = int(0.02 * height * width)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        model = build_sam2(MODEL_CFG, CHECKPOINT_PATH, device="cuda")
        generator = SAM2AutomaticMaskGenerator(
            model,
            points_per_side=8,
            min_mask_region_area=min_area,
            pred_iou_thresh=0.9,
            stability_score_thresh=0.90,
        )
        raw_masks = generator.generate(source)

    filtered_masks = filter_masks(raw_masks, width, height)

    return JSONResponse(
        {"object_count": len(filtered_masks), "raw_mask_count": len(raw_masks)}
    )
