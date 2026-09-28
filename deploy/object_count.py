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
constructor already supports `min_mask_region_area` and `stability_score_thresh` - filtering
happens INSIDE `.generate()`, not as a separate manual post-processing pass. min_mask_region_area
is computed as 1% of the input image's own pixel area (not a fixed pixel count - a fixed 100px^2
was the first version of this file and badly over-segmented a real ~3.4-megapixel test photo,
object_count=17 for a still life with ~4 real objects, see count() for the fix). stability_score_
thresh ~0.80 (the library's own default, 0.95, is tuned for precise segmentation and would likely
under-count real objects that have a genuinely fuzzy/soft edge).

Deploy: uv run --group deploy modal deploy deploy/object_count.py

Fast enough (SAM2's own per-point speed + a single shared image encoder pass) to be a PLAIN
synchronous web endpoint - unlike deploy/layer_separation.py's submit/result split, which exists
specifically because THAT call can exceed Modal's 150-second web-endpoint ceiling. No Cls.from_name
machinery needed here.

torch/sam2/PIL are only ever imported inside method bodies below, never at module level - same
reasoning as every other deploy script in this project (the local venv doesn't carry torch).
"""

import io

import modal
from fastapi import File, Response, UploadFile
from fastapi.responses import JSONResponse

CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
CHECKPOINT_PATH = "/opt/sam2/checkpoints/sam2.1_hiera_tiny.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

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


@app.function(image=image, gpu="T4", scaledown_window=300)
@modal.fastapi_endpoint(method="POST")
async def count(file: UploadFile = File(...)) -> Response:
    import numpy as np
    import torch
    from PIL import Image
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    contents = await file.read()
    source = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))

    # A fixed pixel-area threshold (the first version of this endpoint used 100px^2) behaves
    # wildly inconsistently across photo resolutions, and even a resolution-relative 1-2%
    # threshold wasn't enough on its own - confirmed empirically against two real photos, not
    # assumed: a 4-real-object still life came back object_count=17 at the library's default
    # points_per_side=32, and a single-person photo with grass and a background crowd came back
    # 67. SAM2's automatic mode doesn't distinguish "discrete foreground object" from "any
    # locally coherent texture patch" (a chunk of lawn, a window pane, a distant figure are all
    # individually large and stable enough to survive an area filter alone). A much sparser point
    # grid is the one lever that empirically moved the number at all (points_per_side=2 dropped
    # the still-life count to 8) - this remains a REAL, OPEN CALIBRATION LIMITATION, not a solved
    # problem: a 4-point grid risks under-sampling genuinely separate objects just as easily as a
    # dense grid over-counts texture. MIN_LAYERS/MAX_LAYERS clamping in
    # features/layer_separation.py is what actually keeps this safe end to end regardless of how
    # well-calibrated this raw count is - treat that clamp as the real safety net, this endpoint
    # as a best-effort signal to correct further against more real photos over time.
    min_area = int(0.02 * source.shape[0] * source.shape[1])

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        model = build_sam2(MODEL_CFG, CHECKPOINT_PATH, device="cuda")
        generator = SAM2AutomaticMaskGenerator(
            model,
            points_per_side=8,
            min_mask_region_area=min_area,
            pred_iou_thresh=0.9,
            stability_score_thresh=0.90,
        )
        masks = generator.generate(source)

    return JSONResponse({"object_count": len(masks)})
