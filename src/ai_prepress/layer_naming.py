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

One request labels every layer of a photo at once via the model's own batch_chat API, rather than
one round trip per layer.
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


def name_layers(layers: list[LoadedImage], endpoint: str | None = None, timeout: float = 60.0) -> list[str]:
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
