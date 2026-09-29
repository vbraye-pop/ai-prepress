"""Modal deployment for box-prompted instance segmentation, the second stage of the per-instance
separation pipeline: deploy/object_detect.py finds candidate object boxes, this file turns each
box into a pixel-accurate alpha mask. See deploy/object_count.py's own docstring for the SAM2
license research (Apache 2.0, SAM3 rejected over its custom acceptable-use license) - not repeated
here.

A separate deploy from object_count.py on purpose, not a second endpoint on that app: that app is
a plain stateless function re-loading the model on every cold start, sized for a coarse pre-flight
count. This stage's output becomes the actual layer alpha, so it needs the larger
sam2.1_hiera_base_plus.pt checkpoint (not the tiny one) and a warm, already-constructed predictor
reused across the several box prompts a single photo needs - hence @app.cls + @modal.enter()
below, unlike object_count.py's plain @app.function.

Real inference API (from facebookresearch/sam2's own sam2_image_predictor.py, not guessed):

    predictor = SAM2ImagePredictor(build_sam2(cfg, checkpoint, device="cuda"))
    predictor.set_image(source_rgb_hwc_uint8)   # paid once per image
    masks, scores, _ = predictor.predict(box=box, multimask_output=True)   # cheap, per box
    best = masks[np.argmax(scores)]

Coordinate-space contract - the invariant sentence below is shared verbatim with
deploy/object_detect.py (its sibling in this pipeline stage), so a reader of either file sees the
same guarantee in the same words: boxes are absolute pixel XYXY floats in the coordinate space of
the image EXACTLY as uploaded, no server-side resizing. On this endpoint's own side, that holds
because set_image() is called once on that same unresized array, and predict()'s own default
(normalize_coords=True, not overridden here) reads box coordinates in the resolution set_image()
was given - so passing boxes through unmodified end to end is what keeps the contract true here.

Deploy: uv run --group deploy modal deploy deploy/object_segment.py

torch/sam2/PIL are only ever imported inside method bodies below, never at module level - same
reasoning as every other deploy script in this project (the local venv doesn't carry torch).
"""

import io
import json
import zipfile

import modal
from fastapi import File, Form, Response, UploadFile

CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
CHECKPOINT_PATH = "/opt/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

app = modal.App("ai-prepress-object-segment")

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
        # -f: a plain `curl -L` with no -f writes a 404 response body straight into the
        # checkpoint file and still exits 0, which would only surface later as an unintelligible
        # torch.load error on first inference - fail the image build immediately instead
        f"mkdir -p /opt/sam2/checkpoints && curl -fL -o {CHECKPOINT_PATH} {CHECKPOINT_URL}",
    )
)


@app.cls(image=image, gpu="T4", scaledown_window=300)
class ObjectSegmenter:
    @modal.enter()
    def load(self):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model = build_sam2(MODEL_CFG, CHECKPOINT_PATH, device="cuda")
        self.predictor = SAM2ImagePredictor(model)

    @modal.fastapi_endpoint(method="POST")
    async def segment(self, file: UploadFile = File(...), boxes: str = Form(...)) -> Response:
        import numpy as np
        import torch
        from PIL import Image

        contents = await file.read()
        source = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
        box_list = json.loads(boxes)

        masks_out = []
        scores_out = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # set_image (the expensive shared image-encoder pass) and every predict() below must
            # share one autocast scope - an embedding computed under bf16 and later consumed
            # outside it is a real dtype-mismatch source in sam2
            self.predictor.set_image(source)
            for box in box_list:
                masks, scores, _ = self.predictor.predict(
                    box=np.array(box, dtype=np.float32), multimask_output=True
                )
                best = int(np.argmax(scores))
                masks_out.append(masks[best])
                scores_out.append(float(scores[best]))

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            for i, mask in enumerate(masks_out):
                mask_8bit = (mask.astype(np.uint8) * 255)
                mask_buffer = io.BytesIO()
                Image.fromarray(mask_8bit, mode="L").save(mask_buffer, format="PNG")
                zf.writestr(f"mask_{i:02d}.png", mask_buffer.getvalue())
            zf.writestr("manifest.json", json.dumps({"scores": scores_out}))

        return Response(content=buffer.getvalue(), media_type="application/zip")
