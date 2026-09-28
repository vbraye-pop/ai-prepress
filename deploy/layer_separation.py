"""Modal deployment for photo -> RGBA layers + background decomposition.

Model: Qwen/Qwen-Image-Layered (Apache 2.0, confirmed directly on its own HuggingFace model
card - no license trap here, unlike deploy/face_parsing.py's model). Real inference API (from the
model's own README, not guessed):

    pipeline = QwenImageLayeredPipeline.from_pretrained("Qwen/Qwen-Image-Layered")
    output = pipeline(image=..., layers=4, resolution=640, ...)
    output_image = output.images[0]  # list of N RGBA PIL images, bottom-to-top stacking order

Two real-API facts that shape this file, discovered by reading the model card directly rather
than assumed from the earlier general research pass:

1. `layers` is a CALLER-SPECIFIED count, not auto-detected object count. DEFAULT_LAYER_COUNT below
   is a fixed constant for this phase (matches the model's own documented example) - exposing it as
   a user-facing control is real future work, not needed for the smallest useful slice.
2. The model has no separate "background" output - it returns N RGBA layers in bottom-to-top order
   (confirmed by the official repo's own PowerPoint-export instructions: "upload the layers in
   order - from the bottom layer to the top"). Layer 0 is therefore treated as the background plate
   here, by that documented stacking convention, not by an explicit background flag the API
   provides. This is the one part of the wire contract that needs empirical confirmation against a
   real photo once this is actually deployed (see the plan's verification section) - flagged, not
   silently assumed correct.

`resolution` is a fixed working bucket (640 or 1024, not the input's own size) - every output layer
gets resized back to the source image's dimensions before being sent over the wire, same
upsample-back-to-input-size pattern deploy/face_parsing.py already uses.

Deploy: uv run --group deploy modal deploy deploy/layer_separation.py

torch/diffusers/PIL are only ever imported inside method bodies below, never at module level -
this file gets parsed locally by `modal deploy` to register the app, and the local ai-prepress
venv deliberately doesn't carry torch (nothing local needs it, only the remote container does, per
pyproject.toml's split between main deps and the deploy group).

GPU tier: A100-80GB, not T4 (used by the much lighter SegFormer face-parsing model) - Qwen-Image's
own public size class is in the tens-of-billions-of-parameters range, comfortably exceeding a T4's
16GB even in bf16. Explicitly flagged as a starting point to correct against real measured
wall-clock/OOM behavior once Modal access exists, not asserted as final - a one-line edit to
`@app.cls(gpu=...)` either way.
"""

import io
import zipfile

import modal
from fastapi import File, Response, UploadFile
from fastapi.responses import JSONResponse

MODEL_ID = "Qwen/Qwen-Image-Layered"
DEFAULT_LAYER_COUNT = 4  # matches the model's own documented example - see module docstring
RESOLUTION_BUCKET = 640  # "recommended for this version" per the model's own README

app = modal.App("ai-prepress-layer-separation")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "git+https://github.com/huggingface/diffusers",  # QwenImageLayeredPipeline is too new
        # for a stable diffusers release as of this writing - the model's own README installs
        # from git directly, followed here rather than pinning a release that may not have it
        "transformers>=4.51.3",  # Qwen2.5-VL support, per the model's own README
        "accelerate",
        "pillow",
        "numpy",
        "fastapi",
        "python-multipart",
    )
    .run_commands(
        "python -c \""
        "from diffusers import QwenImageLayeredPipeline; "
        f"QwenImageLayeredPipeline.from_pretrained('{MODEL_ID}')"
        "\""
    )
)


@app.cls(image=image, gpu="A100-80GB", scaledown_window=600, timeout=600)
class LayerDecomposer:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import QwenImageLayeredPipeline

        self.pipeline = QwenImageLayeredPipeline.from_pretrained(MODEL_ID)
        self.pipeline = self.pipeline.to("cuda", torch.bfloat16)
        self.pipeline.set_progress_bar_config(disable=True)

    def decompose(self, image_bytes: bytes, layers: int = DEFAULT_LAYER_COUNT) -> bytes:
        """Not a web endpoint - called via .spawn() from the submit handler below, so it isn't
        subject to Modal web functions' 150-second HTTP request ceiling (this class's own
        `timeout=600` applies instead). Returns a zip: background.png (RGB, 8-bit, layer 0's own
        RGB - it's genuinely model-generated content with no higher-precision source behind it,
        an explicit documented exception to this project's bit-depth-preservation rule, see
        features/layer_separation.py) plus alpha_00.png..alpha_NN.png (single-channel 8-bit, one
        per remaining layer - RGB is discarded for these since the client recombines them with
        the ORIGINAL image's own full-precision RGB instead) plus manifest.json."""
        import json

        import torch
        from PIL import Image

        source = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        target_size = source.size

        generator = torch.Generator(device="cuda").manual_seed(777)
        with torch.inference_mode():
            output = self.pipeline(
                image=source,
                generator=generator,
                true_cfg_scale=4.0,
                negative_prompt=" ",
                num_inference_steps=50,
                num_images_per_prompt=1,
                layers=layers,
                resolution=RESOLUTION_BUCKET,
                cfg_normalize=True,
                use_en_prompt=True,
            )
        output_layers = output.images[0]  # bottom-to-top order, see module docstring

        background_rgba = output_layers[0].resize(target_size, Image.LANCZOS)
        background_rgb = background_rgba.convert("RGB")

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            bg_bytes = io.BytesIO()
            background_rgb.save(bg_bytes, format="PNG")
            zf.writestr("background.png", bg_bytes.getvalue())

            for index, layer_rgba in enumerate(output_layers[1:]):
                resized = layer_rgba.resize(target_size, Image.LANCZOS)
                alpha = resized.split()[-1]
                alpha_bytes = io.BytesIO()
                alpha.save(alpha_bytes, format="PNG")
                zf.writestr(f"alpha_{index:02d}.png", alpha_bytes.getvalue())

            zf.writestr("manifest.json", json.dumps({"layer_count": len(output_layers) - 1}))

        return buffer.getvalue()


@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
async def submit(file: UploadFile = File(...)) -> Response:
    contents = await file.read()
    call = LayerDecomposer().decompose.spawn(contents)
    return JSONResponse({"call_id": call.object_id})


@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
async def result(call_id: str) -> Response:
    function_call = modal.FunctionCall.from_id(call_id)
    try:
        zip_bytes = function_call.get(timeout=0)
    except TimeoutError:
        return JSONResponse({"status": "pending"})
    except modal.exception.OutputExpiredError:
        # results expire after 7 days - the client's own poll loop times out at 10 minutes, so
        # reaching this means a call_id from a much older session was polled, not a race
        return JSONResponse({"status": "expired"})
    return Response(content=zip_bytes, media_type="application/zip")
