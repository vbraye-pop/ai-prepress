"""Modal deployment for short semantic layer names, used to pre-fill Layer Separation's rename
field instead of the generic "Layer N" placeholder (see features/layer_separation.py).

Model: OpenGVLab/InternVL3_5-2B, Apache 2.0 (confirmed directly on its own HuggingFace model card,
backbone Qwen3 also Apache 2.0) - picked earlier this session after Moondream 3 (Preview) was
rejected for Business Source License 1.1. Task is narrow (label one isolated crop), so 2.3B params
is already generous, no need for anything bigger.

Real API, quoted from the model's own card, not guessed: `AutoModel.from_pretrained(...,
trust_remote_code=True)` + a `model.batch_chat(tokenizer, pixel_values, num_patches_list=...,
questions=..., generation_config=...)` call that labels every image in one request - used here to
label a whole photo's layers in a single call instead of one round trip per layer.

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
needed, unlike deploy/layer_separation.py's multi-minute diffusion call.

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
PROMPT = "<image>\nName the object in this image in 1 to 3 words. Respond with only the name, no punctuation or explanation."

app = modal.App("ai-prepress-layer-naming")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "torchvision",
        "transformers",
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

    per_image_pixel_values = []
    for upload in files:
        contents = await upload.read()
        # already flat RGB by the time it gets here - see module docstring point 1
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        per_image_pixel_values.append(_load_pixel_values(pil_image).to(torch.bfloat16).cuda())

    num_patches_list = [pv.size(0) for pv in per_image_pixel_values]
    pixel_values = torch.cat(per_image_pixel_values, dim=0)
    questions = [PROMPT] * len(files)
    generation_config = {"max_new_tokens": 16, "do_sample": False}

    with torch.inference_mode():
        responses = model.batch_chat(
            tokenizer,
            pixel_values,
            num_patches_list=num_patches_list,
            questions=questions,
            generation_config=generation_config,
        )

    labels = [response.strip().strip(".") for response in responses]
    return JSONResponse({"labels": labels})
