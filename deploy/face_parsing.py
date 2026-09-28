"""Modal deployment for face-region parsing.

Model: jonathandinu/face-parsing (SegFormer-B5 fine-tuned on CelebAMask-HQ). Chosen after a
license sweep found CelebAMask-HQ/LaPa contaminate essentially every open-weight face-parsing
model in the field, including Microsoft's own MIT-licensed FaRL repo and the newest 2024/25
architectures - there is no fully commercially-clean drop-in as of this writing. This is a
deliberate non-commercial placeholder for R&D only: it proves the remote-model-call pattern and
unblocks Retouch Faces' region work now, and gets swapped (MediaPipe-derived regions, EasyPortrait
with a legal sign-off, or a paid Banuba license) before anything client-facing touches it.

Deploy: uv run --group deploy modal deploy deploy/face_parsing.py

torch/transformers/PIL are only ever imported inside the class methods below, never at module
level - this file gets parsed locally by `modal deploy` to register the app, and the local
ai-prepress venv deliberately doesn't carry torch as a dependency (nothing local needs it, only
the remote container does, per the pyproject.toml split between main deps and the deploy group).
"""

import io

import modal
from fastapi import File, Response, UploadFile

MODEL_ID = "jonathandinu/face-parsing"

# Same 19-class CelebAMask-HQ scheme the model was fine-tuned on, index order matches the
# model's own config.json id2label - used by ai_prepress.face_parsing to name pixel values back.
LABELS = [
    "background", "skin", "nose", "eye_g", "l_eye", "r_eye", "l_brow", "r_brow",
    "l_ear", "r_ear", "mouth", "u_lip", "l_lip", "hair", "hat", "ear_r", "neck_l",
    "neck", "cloth",
]

app = modal.App("ai-prepress-face-parsing")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch", "torchvision", "transformers>=4.40", "pillow", "numpy", "fastapi", "python-multipart"
    )
    .run_commands(
        "python -c \""
        "from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation; "
        f"SegformerImageProcessor.from_pretrained('{MODEL_ID}'); "
        f"SegformerForSemanticSegmentation.from_pretrained('{MODEL_ID}')"
        "\""
    )
)


@app.cls(image=image, gpu="T4", scaledown_window=120)
class FaceParser:
    @modal.enter()
    def load(self):
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
        self.model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID).to(self.device)
        self.model.eval()

    @modal.fastapi_endpoint(method="POST")
    async def parse(self, file: UploadFile = File(...)) -> Response:
        import torch
        from PIL import Image

        contents = await file.read()
        source = Image.open(io.BytesIO(contents)).convert("RGB")

        inputs = self.processor(images=source, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            logits = self.model(**inputs).logits

        # model's native output is roughly 1/4 resolution - upsample back to the input size
        # rather than leave the caller to guess the downscale factor
        upsampled = torch.nn.functional.interpolate(
            logits, size=source.size[::-1], mode="bilinear", align_corners=False
        )
        labels = upsampled.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()

        buffer = io.BytesIO()
        Image.fromarray(labels, mode="L").save(buffer, format="PNG")
        return Response(content=buffer.getvalue(), media_type="image/png")
