"""Modal deployment for open-vocabulary object detection, the first stage of the new
per-instance separation pipeline (see features/layer_separation.py for the eventual caller).

Model: IDEA-Research/grounding-dino-base, Apache-2.0 (confirmed on its own HF model card).
Loaded through transformers' own AutoModelForZeroShotObjectDetection + AutoProcessor, not the raw
IDEA-Research/GroundingDINO GitHub repo - that repo needs a fragile custom CUDA kernel build,
while transformers' port is a pure-PyTorch/optional-kernel implementation with the same weights.
`disable_custom_kernels=True` is passed to from_pretrained deliberately: the optional kernel build
is known-broken (huggingface/transformers#35976, #35979) and without this flag a container falls
back to the working pure-PyTorch attention path anyway, just after a slow, warning-laden build
attempt on every cold start.

Coordinate-space contract - the invariant sentence below is shared verbatim with
deploy/object_segment.py (its sibling in this pipeline stage), so a reader of either file sees the
same guarantee in the same words: boxes are absolute pixel XYXY floats in the coordinate space of
the image EXACTLY as uploaded, no server-side resizing. The processor internally resizes for the
model's own input, but
post_process_grounded_object_detection's target_sizes=[source.size[::-1]] rescales its output back
to the original image before this endpoint ever returns it.

GroundingDinoConfig's max_text_len defaults to 256 tokens - a text_prompt with many candidate
labels silently truncates at that point rather than erroring, so very long prompts lose their
tail entries without warning.

Deploy: uv run --group deploy modal deploy deploy/object_detect.py

Plain synchronous endpoint, not @app.cls warm-model reuse the way a per-instance loop would want -
one call per photo, single forward pass, no benefit from keeping state across calls beyond the
model weights themselves (which @modal.enter() already caches per warm container).

torch/transformers/PIL are only ever imported inside the class methods below, never at module
level - this file gets parsed locally by `modal deploy` to register the app, and the local
ai-prepress venv deliberately doesn't carry torch as a dependency (nothing local needs it, only
the remote container does, per the pyproject.toml split between main deps and the deploy group).
"""

import io

import modal
from fastapi import File, Form, UploadFile
from fastapi.responses import JSONResponse

MODEL_ID = "IDEA-Research/grounding-dino-base"

app = modal.App("ai-prepress-object-detect")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch", "torchvision", "transformers>=4.40", "pillow", "numpy", "fastapi", "python-multipart"
    )
    .run_commands(
        "python -c \""
        "from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection; "
        f"AutoProcessor.from_pretrained('{MODEL_ID}'); "
        f"AutoModelForZeroShotObjectDetection.from_pretrained('{MODEL_ID}')"
        "\""
    )
)


@app.cls(image=image, gpu="T4", scaledown_window=120)
class ObjectDetector:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            MODEL_ID, disable_custom_kernels=True
        ).to(self.device)
        self.model.eval()

    @modal.fastapi_endpoint(method="POST")
    async def detect(
        self,
        file: UploadFile = File(...),
        text_prompt: str = Form(...),
        box_threshold: float = Form(0.4),
        text_threshold: float = Form(0.3),
    ) -> JSONResponse:
        import torch
        from PIL import Image

        contents = await file.read()
        source = Image.open(io.BytesIO(contents)).convert("RGB")

        # a single-item batch: `text` takes one full candidate-label string per image, not a
        # nested [[prompt]] shape - the transformers version this image resolves to raises inside
        # its own tokenizer on the nested form (TypeError: TextEncodeInput must be ...), confirmed
        # by reproducing it locally against the same model id before landing on this shape
        inputs = self.processor(images=source, text=[text_prompt], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)

        # target_sizes takes (height, width) - source.size is PIL's (width, height), hence the
        # reverse - this is what maps the model's internal resize back to the uploaded image's own
        # pixel space, honouring this file's coordinate-space contract above
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[source.size[::-1]],
        )[0]

        # transformers versions disagree on where the string labels live: older releases put them
        # directly in "labels", 4.51+ moved to "text_labels" and repurposed "labels" for class
        # indices - text_labels is preferred when present so this doesn't regress on the newer shape
        text_labels_out = results.get("text_labels")
        raw_labels = text_labels_out if text_labels_out is not None else results["labels"]

        detections = [
            {
                "label": str(label),
                "score": float(score),
                "box": [float(v) for v in box.tolist()],
            }
            for label, score, box in zip(raw_labels, results["scores"], results["boxes"])
        ]

        return JSONResponse({"detections": detections})
