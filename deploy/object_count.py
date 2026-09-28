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
happens INSIDE `.generate()`, not as a separate manual post-processing pass. Real documented
practice for object-counting specifically: min_mask_region_area ~100px, stability_score_thresh
~0.80 (the library's own default, 0.95, is tuned for precise segmentation and would likely
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
CHECKPOINT_PATH = "/root/sam2/checkpoints/sam2.1_hiera_tiny.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

app = modal.App("ai-prepress-object-count")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch", "torchvision", "pillow", "numpy", "fastapi", "python-multipart")
    .run_commands(
        "git clone --depth 1 https://github.com/facebookresearch/sam2.git /root/sam2",
        "cd /root/sam2 && pip install -e .",
        f"mkdir -p /root/sam2/checkpoints && curl -L -o {CHECKPOINT_PATH} {CHECKPOINT_URL}",
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

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        model = build_sam2(MODEL_CFG, CHECKPOINT_PATH, device="cuda")
        generator = SAM2AutomaticMaskGenerator(
            model,
            min_mask_region_area=100,
            # the library's own default (0.95) is tuned for precise segmentation - this is a
            # coarse object count, not a mask-quality task, so a looser threshold catches real
            # objects with a genuinely soft/fuzzy edge that a stricter one would drop
            stability_score_thresh=0.80,
        )
        masks = generator.generate(source)

    return JSONResponse({"object_count": len(masks)})
