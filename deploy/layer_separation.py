"""Modal deployment for photo -> RGBA layers + background decomposition.

Model: Qwen/Qwen-Image-Layered (Apache 2.0, confirmed directly on its own HuggingFace model
card - no license trap here, unlike deploy/face_parsing.py's model). Real inference API (from the
model's own README, not guessed):

    pipeline = QwenImageLayeredPipeline.from_pretrained("Qwen/Qwen-Image-Layered")
    output = pipeline(image=..., layers=4, resolution=640, ...)
    output_image = output.images[0]  # list of N RGBA PIL images, bottom-to-top stacking order

Two real-API facts that shape this file, discovered by reading the model card directly rather
than assumed from the earlier general research pass:

1. `layers` is a CALLER-SPECIFIED count - this model has no auto-detect mode of its own.
   DEFAULT_LAYER_COUNT below is the fallback when a caller doesn't override it (also the value
   `deploy/object_count.py` + `features/layer_separation.py` fall back to if the object-count
   pre-flight step itself fails) - the real per-photo count now comes from that separate model,
   not a fixed constant baked in here.
2. The model has no separate "background" output - it returns N RGBA layers in bottom-to-top order
   (confirmed by the official repo's own PowerPoint-export instructions: "upload the layers in
   order - from the bottom layer to the top"). Layer 0 is therefore treated as the background plate
   here, by that documented stacking convention, not by an explicit background flag the API
   provides. This is the one part of the wire contract that needs empirical confirmation against a
   real photo once this is actually deployed (see the plan's verification section) - flagged, not
   silently assumed correct.

`resolution` is a fixed working bucket (not the input's own size) - every output layer gets resized
back to the source image's dimensions before being sent over the wire, same upsample-back-to-input-
size pattern deploy/face_parsing.py already uses.

RESOLUTION_BUCKET is 640, not 1024, decided by A/B testing both against real photos rather than
taken on the model card's own "recommended for this version" line. Same seed, same photos
(art.jpg's pitcher/orange/book/table still life, portrait.png's person-on-grass natural photo),
same layers=4, only the bucket changed:

- Quality: 1024 did not produce a cleaner separation. On art.jpg it returned 3 layers same as 640,
  but one of the three came back at exactly 0.0 alpha coverage - a wasted, empty layer, not a
  finer split of the orange+table merge. Coverage of the two remaining layers also shifted
  (0.19/0.18 at 1024 vs 0.08/0.37/0.07 at 640), consistent with a different, not better, split.
- Edge sharpness: every layer is resized back to source resolution regardless of the working
  bucket, so a narrower bucket mechanically narrows the raw output ramp width by roughly
  (640/1024 = 0.625x) even with zero real quality change. Measured raw median ramp width at 1024
  came out to ~0.64-0.75x of 640's on both test photos - at or above that mechanical floor, so
  there is no evidence 1024 resolves the alpha edge any more finely than 640 once the resampling
  arithmetic is accounted for.
- Latency: cold start at 1024 took 587s against this class's own `timeout=600` - 13 seconds of
  headroom on a single measurement, with no retry budget. A second 1024 call 43s later took 490s.
  Two containers were alive at the time so it isn't confirmed which one served it, but neither
  reading comes close to 640's own measured ~430s cold / ~125s warm, and both sit near the 600s
  ceiling.

1024 was rejected on all three axes measured, not just latency.

`prompt` (optional, threaded through submit -> decompose.spawn -> the pipeline call, same shape
as `layers`) is the pipeline's own positive-caption override, paired with the always-present
`negative_prompt` in diffusers' usual convention. Confirmed against the model's own README
(github.com/QwenLM/Qwen-Image-Layered), not guessed: "The text prompt is intended to describe the
overall content of the input image - including elements that may be partially occluded", and the
README documents recursive decomposition as a real, supported capability ("any layer can itself be
further decomposed, enabling infinite decomposition") with no code example of its own - this file's
`prompt` plumbing plus features/layer_separation.py's recursive repair pass on a flagged layer are
what actually exercise it. Left out of the pipeline call entirely when not given, so the
unprompted path's automatic captioning (`use_en_prompt`) is unchanged from before this existed.
Never sent alongside a transparent (RGBA) input image - the model's own GitHub issue #17 asks how
alpha-channel input is handled and has had no answer for months, so this project treats it as
unsupported rather than assumed to work, and always flattens to opaque RGB first (see
features/layer_separation.py's repair path).

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
from fastapi import File, Form, Response, UploadFile
from fastapi.responses import JSONResponse

MODEL_ID = "Qwen/Qwen-Image-Layered"
DEFAULT_LAYER_COUNT = 4  # matches the model's own documented example - see module docstring
RESOLUTION_BUCKET = 640  # measured against 1024 on real photos - see module docstring

app = modal.App("ai-prepress-layer-separation")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")  # needed for pip's `git+https://...` install of diffusers below
    .pip_install(
        "torch",
        "torchvision",  # the pipeline's own Qwen2VLVideoProcessor requires it, discovered from a
        # real build failure (ImportError: Qwen2VLVideoProcessor requires the Torchvision
        # library), not in the model's own documented install instructions
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

    @modal.method()
    def decompose(self, image_bytes: bytes, layers: int = DEFAULT_LAYER_COUNT, prompt: str | None = None) -> bytes:
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
        pipeline_kwargs = dict(
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
        if prompt is not None:
            pipeline_kwargs["prompt"] = prompt
        with torch.inference_mode():
            output = self.pipeline(**pipeline_kwargs)
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
@modal.asgi_app()
def web():
    """A single ASGI app hosting both routes under one URL, rather than two separate
    @modal.fastapi_endpoint functions - discovered the hard way against a real deployment that
    each fastapi_endpoint-decorated function gets its OWN subdomain (...-submit.modal.run,
    ...-result.modal.run), not a shared base path with route dispatch. The client
    (ai_prepress.layer_decompose) is written around one base URL with /submit and /result
    sub-paths, so this is the fix that matches it rather than a client-side rewrite."""
    from fastapi import FastAPI

    web_app = FastAPI()

    @web_app.post("/submit")
    async def submit(
        file: UploadFile = File(...),
        layers: int = Form(DEFAULT_LAYER_COUNT),
        prompt: str | None = Form(None),
    ) -> Response:
        contents = await file.read()
        # a plain in-module `LayerDecomposer().decompose.spawn(...)` fails here with
        # `AttributeError: 'function' object has no attribute 'spawn'` - discovered against a
        # real deployment, not documented anywhere obvious. Calling a sibling class's method from
        # inside another already-running container of the same app needs an explicit lookup by
        # name (the same mechanism cross-app calls use), not direct instantiation.
        decomposer_cls = modal.Cls.from_name("ai-prepress-layer-separation", "LayerDecomposer")
        call = decomposer_cls().decompose.spawn(contents, layers=layers, prompt=prompt)
        return JSONResponse({"call_id": call.object_id})

    @web_app.get("/result")
    async def result(call_id: str) -> Response:
        function_call = modal.FunctionCall.from_id(call_id)
        try:
            zip_bytes = function_call.get(timeout=0)
        except TimeoutError:
            return JSONResponse({"status": "pending"})
        except modal.exception.OutputExpiredError:
            # results expire after 7 days - the client's own poll loop times out at 10 minutes,
            # so reaching this means a call_id from a much older session was polled, not a race
            return JSONResponse({"status": "expired"})
        return Response(content=zip_bytes, media_type="application/zip")

    return web_app
