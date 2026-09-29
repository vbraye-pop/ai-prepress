"""Modal deployment for per-instance high-resolution alpha matting - the third stage of the
per-instance separation pipeline (ai_prepress.object_detect boxes -> ai_prepress.object_segment
masks -> this stage's own high-resolution alpha per cropped instance).

Model: ZhengPeng7/BiRefNet_HR-matting, MIT (confirmed directly on its own HuggingFace model card).
Loaded via `AutoModelForImageSegmentation.from_pretrained(model_id, trust_remote_code=True)`, the
same trust_remote_code bake pattern deploy/layer_naming.py already uses for InternVL3.5-2B. Real
top-level imports, confirmed by reading the model's own birefnet.py directly rather than the
training repo's full requirements.txt: torch, torchvision, transformers>=4.40, kornia, timm,
einops, pillow, numpy.

Real inference call, quoted from the model card's own example:

    transform = Resize((2048, 2048)) + ToTensor + Normalize(IMAGENET_MEAN, IMAGENET_STD)
    pred = model(transform(crop).unsqueeze(0).cuda().half())[-1].sigmoid()

`.half()` (fp16), not `.bfloat16()` - T4 is a Turing-generation GPU with no native bf16
tensor-core support, unlike the A100 deploy/layer_separation.py runs its own model on.

This endpoint applies that exact transform - a plain non-aspect-preserving squash to
(2048, 2048) - to whatever image it's given, then resizes the model's own output back down to
that same input's size before returning. Tested directly against a real non-square crop (a
150x700 forearm crop from portrait.png, aspect ratio ~1:4.7, cropped where the arm crosses a dark
jacket edge on one side and denim on the other) with both this plain squash and a
pad-to-square-then-crop-back alternative (black-pad the crop to 700x700 client-side before
upload, then crop the returned 700x700 alpha back down to the original 150x700 region): squash
won, on two separate pieces of real evidence, not assumed:

1. Effective resolution: pad-to-square gives the model only 150/700 = ~21% of the frame's own
   width to represent the arm in, since the rest is black padding - a real small mole on the
   forearm (visible in the source crop) came back as a distinct alpha hole in squash's output but
   was lost entirely in pad's - a concrete, visible loss of detail, not a hypothetical one.
2. 10%-90% transition width (same metric alpha_refine.py's own docstring uses, anchored to each
   row's own plateau levels): measured on two rows crossing the arm's edge, squash came back
   sharper or tied at every row (2px vs pad's 4px on one row, 3px vs 3px tied on the other) -
   never worse.

Both are consistent with the model being trained on this exact fixed non-aspect-preserving
resize in the first place, so a squashed object is in-distribution for it while a black
letterboxed one, with most of its frame unused, is not. ai_prepress.instance_matte's own client
sends every crop through unmodified as a result - no pad-to-square prep exists on either side of
this pipeline stage.

Per-crop wall-clock, measured against this real deployment on a warm container: a single 900x1560
crop took ~3.1s, a single 150x700 crop ~2.3s, and a 4-crop request (2 of each) took 7.9s total -
comfortably inside Modal's 150-second web-endpoint ceiling even at MAX_LAYERS-scale crop counts
(~8), unlike deploy/layer_naming.py's own measured 77s/6-crop close call on a heavier model.

CRITICAL: crops are matted SEQUENTIALLY, one model forward pass at a time, never stacked into one
batched tensor. A single 2048x2048 fp16 forward measures roughly 9.5GB on a T4, close to its 16GB
ceiling - batching even 2 crops risks OOM. This project hit a real T4 OOM this same session on a
different model from exactly that mistake (see deploy/layer_naming.py's own docstring); not
repeated here.

Deploy: uv run --group deploy modal deploy deploy/instance_matte.py

torch/transformers/PIL are only ever imported inside method bodies below, never at module level -
same reasoning as every other deploy script in this project (the local venv deliberately doesn't
carry torch, per pyproject.toml's split between main deps and the deploy group).
"""

import io
import json
import zipfile

import modal
from fastapi import File, Response, UploadFile

MODEL_ID = "ZhengPeng7/BiRefNet_HR-matting"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MATTE_SIZE = 2048

app = modal.App("ai-prepress-instance-matte")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "torchvision",
        "transformers>=4.40",
        "kornia",
        "timm",
        "einops",
        "pillow",
        "numpy",
        "fastapi",
        "python-multipart",
    )
    .run_commands(
        "python -c \""
        "from transformers import AutoModelForImageSegmentation; "
        f"AutoModelForImageSegmentation.from_pretrained('{MODEL_ID}', trust_remote_code=True)"
        "\""
    )
)


@app.cls(image=image, gpu="T4", scaledown_window=300)
class InstanceMatter:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForImageSegmentation

        self.model = (
            AutoModelForImageSegmentation.from_pretrained(MODEL_ID, trust_remote_code=True)
            .cuda()
            .half()
            .eval()
        )

    @modal.fastapi_endpoint(method="POST")
    async def matte(self, files: list[UploadFile] = File(...)) -> Response:
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from torchvision import transforms as T

        transform = T.Compose(
            [
                T.Resize((MATTE_SIZE, MATTE_SIZE)),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for index, upload in enumerate(files):
                contents = await upload.read()
                crop = Image.open(io.BytesIO(contents)).convert("RGB")
                width, height = crop.size

                # one crop through the model at a time - see module docstring's OOM warning
                with torch.inference_mode():
                    tensor = transform(crop).unsqueeze(0).cuda().half()
                    pred = self.model(tensor)[-1].sigmoid()
                    alpha = F.interpolate(pred, size=(height, width), mode="bilinear", align_corners=False)
                del tensor, pred
                torch.cuda.empty_cache()

                alpha_8bit = (alpha[0, 0].clamp(0.0, 1.0) * 255 + 0.5).to(torch.uint8).cpu().numpy()
                alpha_bytes = io.BytesIO()
                Image.fromarray(alpha_8bit, mode="L").save(alpha_bytes, format="PNG")
                zf.writestr(f"alpha_{index:02d}.png", alpha_bytes.getvalue())

            zf.writestr("manifest.json", json.dumps({"count": len(files)}))

        return Response(content=buffer.getvalue(), media_type="application/zip")
