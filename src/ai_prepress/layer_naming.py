"""Client for the layer-naming Modal endpoint (see deploy/layer_naming.py).

Suggests a short label per layer to pre-fill Layer Separation's rename field - see
features.layer_separation for how a failure here degrades gracefully (naming is cosmetic, the
manual rename field it pre-fills already exists regardless, so this is never a hard dependency of
the core separation feature).

InternVL3.5-2B's own reference preprocessing does `Image.open(...).convert('RGB')` as its first
step - it does not handle RGBA. This module is the one place that flattening happens: each RGBA
layer is composited onto a flat white background (paste with the layer's own alpha as the mask,
the standard PIL idiom for this) before being sent over the wire - the deploy script itself never
needs to know about alpha at all.

One HTTP request still labels every layer of a photo, but the server names them one at a time
internally (see deploy/layer_naming.py for why: batching them into one model.batch_chat call
OOMs a T4 on a real multi-layer photo). Default timeout is 300s, not the 60s a single-shot batch
call would have justified - a real deployed 6-layer request took 77s sequential, and MAX_LAYERS
in features/layer_separation.py allows up to 8.

`list_candidate_objects` reuses this same endpoint (with the optional `prompt`/`max_new_tokens`
overrides deploy/layer_naming.py added for exactly this) to ask InternVL to enumerate every
distinct object in a WHOLE photo BEFORE any layers exist - the first stage of the per-instance
separation pipeline in features/layer_separation.py, which needs candidate object names to hand
Grounding DINO as its own open-vocabulary text prompt. `name_layers` above is untouched: every
field it sends keeps the server's old defaults, so a per-crop naming call behaves identically to
before this function existed.
"""

from __future__ import annotations

import io
import os

import httpx
from PIL import Image

from ai_prepress.io import LoadedImage, to_unit_float


def _flatten_to_white(image: LoadedImage) -> bytes:
    unit = to_unit_float(image.array)
    rgba_8bit = (unit.clip(0.0, 1.0) * 255 + 0.5).astype("uint8")
    rgba = Image.fromarray(rgba_8bit, mode="RGBA") if rgba_8bit.shape[-1] == 4 else Image.fromarray(rgba_8bit).convert("RGBA")

    flattened = Image.new("RGB", rgba.size, (255, 255, 255))
    flattened.paste(rgba, mask=rgba.split()[-1])

    buffer = io.BytesIO()
    flattened.save(buffer, format="PNG")
    return buffer.getvalue()


def name_layers(layers: list[LoadedImage], endpoint: str | None = None, timeout: float = 300.0) -> list[str]:
    if not layers:
        return []

    endpoint = endpoint or os.environ.get("AI_PREPRESS_LAYER_NAMING_URL", "")
    if not endpoint:
        raise ValueError(
            "no layer-naming endpoint configured - set AI_PREPRESS_LAYER_NAMING_URL "
            "or pass endpoint= explicitly"
        )

    files = [("files", (f"layer_{i}.png", _flatten_to_white(layer), "image/png")) for i, layer in enumerate(layers)]

    response = httpx.post(endpoint, files=files, timeout=timeout)
    response.raise_for_status()
    return response.json()["labels"]


OBJECT_LIST_PROMPT = (
    "<image>\nList every distinct physical object in this photo that could be individually cut "
    "out or retouched, comma-separated, short names only (1-3 words each, like 'coffee mug' or "
    "'orange'). Skip the background, wall, table surface, or floor unless one of them is itself a "
    "genuinely separate object. Respond with only the comma-separated list, no numbering, no "
    "explanation."
)
OBJECT_LIST_MAX_NEW_TOKENS = 64  # a single-crop label fits in the server's default 16 tokens, a
# whole-photo list of several short names doesn't - measured against a real deploy, see
# features.layer_separation for why this needed its own value rather than reusing name_layers'


def list_candidate_objects(image: LoadedImage, endpoint: str | None = None, timeout: float = 60.0) -> list[str]:
    """Asks InternVL to enumerate the distinct objects in the WHOLE photo (not a pre-separated
    layer) - the candidate names features.layer_separation hands to Grounding DINO. Returns an
    empty list on an empty/unparseable response, same "nothing to work with" shape as an empty
    `layer_alphas` elsewhere in this project - the caller decides what an empty list means."""
    endpoint = endpoint or os.environ.get("AI_PREPRESS_LAYER_NAMING_URL", "")
    if not endpoint:
        raise ValueError(
            "no layer-naming endpoint configured - set AI_PREPRESS_LAYER_NAMING_URL "
            "or pass endpoint= explicitly"
        )

    files = [("files", ("photo.png", _flatten_to_white(image), "image/png"))]
    data = {"prompt": OBJECT_LIST_PROMPT, "max_new_tokens": OBJECT_LIST_MAX_NEW_TOKENS}

    response = httpx.post(endpoint, files=files, data=data, timeout=timeout)
    response.raise_for_status()
    labels = response.json()["labels"]
    if not labels:
        return []
    return [name.strip() for name in labels[0].split(",") if name.strip()]
