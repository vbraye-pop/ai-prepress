"""Modal deployment for short semantic layer names, used to pre-fill Layer Separation's rename
field instead of the generic "Layer N" placeholder (see features/layer_separation.py).

Model: OpenGVLab/InternVL3_5-2B, Apache 2.0 (confirmed directly on its own HuggingFace model card,
backbone Qwen3 also Apache 2.0) - picked earlier this session after Moondream 3 (Preview) was
rejected for Business Source License 1.1. Task is narrow (label one isolated crop), so 2.3B params
is already generous, no need for anything bigger.

Real API, quoted from the model's own card, not guessed: `AutoModel.from_pretrained(...,
trust_remote_code=True)` + `model.chat(tokenizer, pixel_values, question, generation_config)`,
called once per layer rather than batched through `model.batch_chat` - a real deployed 6-layer
request confirmed the batched form OOMs a T4 (14.57GiB) with this image's plain eager attention
build (no flash-attn wheel installed), since every image's own dynamic_preprocess tiles get
concatenated into one attention pass. Per-image calls bound peak memory to one image's own tile
count regardless of how many layers a photo has.

Two real integration details, confirmed from the model card's own reference code, not assumed:
1. The model's own `load_image()` does `Image.open(...).convert('RGB')` as its first step - it
   does NOT handle RGBA. Every layer crop is flattened onto a flat background client-side (see
   ai_prepress/layer_naming.py) before it ever reaches this deployment, so this file has zero
   RGBA-handling logic of its own.
2. No official "short label" prompt template exists on the model's own card - the terse-label
   phrasing below is authored for this project, not copied from documentation, and should be
   treated as a first cut to tune against real output, not a settled prompt.

Deploy: uv run --group deploy modal deploy deploy/layer_naming.py

Fast enough (2.3B params, ~4.6GB weights in bf16, a batch of small single-object crops) to be a
plain synchronous web endpoint, same reasoning as deploy/object_count.py - no submit/result split
needed, unlike deploy/layer_separation.py's multi-minute diffusion call. Measured, not estimated:
~48s for a real 3-crop batch on a T4, comfortably under Modal's 150s web-endpoint ceiling but
meaningfully slower than "a few seconds" - worth remembering when this runs after a multi-minute
decompose call, since it adds real, non-trivial wall-clock on top of it.

torch/transformers/PIL are only ever imported inside the function body below, never at module
level - same reasoning as every other deploy script in this project.
"""

import io

import modal
from fastapi import File, Response, UploadFile
from fastapi.responses import JSONResponse

MODEL_ID = "OpenGVLab/InternVL3_5-2B"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PROMPT = (
    "<image>\nWhat kind of object is this? Answer with a short category name (1 to 3 words, "
    "like 'coffee mug' or 'denim jacket'), not any text or logo visible on it. Respond with "
    "only the name, no punctuation or explanation."
)
# a real deployed test surfaced the exact failure the "not any text or logo" clause fixes: the
# original prompt (just "name the object") made the model transcribe a sticker's own printed
# text verbatim ("Rly Thot Shot") instead of describing what kind of object it was ("sticker") -
# discovered from real output, not anticipated in advance

app = modal.App("ai-prepress-layer-naming")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "torchvision",
        # unpinned "transformers" pulled a version whose internal weight-tying refactor
        # (AttributeError: 'InternVLChatModel' object has no attribute 'all_tied_weights_keys')
        # broke InternVL's own trust_remote_code modeling code - a real build failure, not a
        # guess. InternVL3.5 (up to 8B) documents transformers>=4.52.1 as its own minimum;
        # pinned to a specific known-working release rather than an open-ended lower bound so a
        # future transformers release can't silently break this build again.
        "transformers==4.55.0",
        "accelerate",
        "einops",  # InternVL's own trust_remote_code modeling files import this directly
        "timm",  # same - part of its own remote code, not an incidental extra
        "sentencepiece",
        "pillow",
        "numpy",
        "fastapi",
        "python-multipart",
    )
    .run_commands(
        "python -c \""
        "from transformers import AutoModel, AutoTokenizer; "
        f"AutoModel.from_pretrained('{MODEL_ID}', trust_remote_code=True); "
        f"AutoTokenizer.from_pretrained('{MODEL_ID}', trust_remote_code=True, use_fast=False)"
        "\""
    )
)


def _build_transform(input_size):
    from torchvision import transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(pil_image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = pil_image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )
    target_aspect_ratio = _find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized = pil_image.resize((target_width, target_height))
    tiles_per_row = target_width // image_size
    processed = []
    for i in range(blocks):
        box = (
            (i % tiles_per_row) * image_size,
            (i // tiles_per_row) * image_size,
            ((i % tiles_per_row) + 1) * image_size,
            ((i // tiles_per_row) + 1) * image_size,
        )
        processed.append(resized.crop(box))
    if use_thumbnail and len(processed) != 1:
        processed.append(pil_image.resize((image_size, image_size)))
    return processed


def _load_pixel_values(pil_image, input_size=448, max_num=12):
    import torch

    transform = _build_transform(input_size)
    tiles = _dynamic_preprocess(pil_image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(tile) for tile in tiles])


@app.function(image=image, gpu="T4", scaledown_window=300)
@modal.fastapi_endpoint(method="POST")
async def name(files: list[UploadFile] = File(...)) -> Response:
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)

    generation_config = {"max_new_tokens": 16, "do_sample": False}

    # One image (and its own dynamic_preprocess tiles) per batch_chat call, not the whole
    # request concatenated into one - a real deployed 6-layer request confirmed this GPU OOMs on
    # a T4 (14.57GiB) with the plain torch/eager attention build this image installs (no
    # flash-attn wheel): even 2 images already logged retried allocator warnings before
    # succeeding, so this isn't a large-N-only edge case worth a higher batch cap, it's the batch
    # concatenation itself. Per-image calls bound peak memory to one image's own tile count
    # regardless of how many layers a photo has, at the cost of one more small generate() call
    # per layer rather than one big one - the model itself is already reloaded fresh per request
    # (see module docstring), so this adds no extra load time, only extra (cheap) generation calls.
    labels = []
    for upload in files:
        contents = await upload.read()
        # already flat RGB by the time it gets here - see module docstring point 1
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        pixel_values = _load_pixel_values(pil_image).to(torch.bfloat16).cuda()

        with torch.inference_mode():
            response = model.chat(tokenizer, pixel_values, PROMPT, generation_config)

        labels.append(response.strip().strip("."))
        del pixel_values
        torch.cuda.empty_cache()
    return JSONResponse({"labels": labels})
