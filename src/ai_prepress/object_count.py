"""Client for the object-count Modal endpoint (see deploy/object_count.py).

A cheap pre-flight step used to size Qwen-Image-Layered's `layers` parameter automatically -
see features.layer_separation for how the returned count becomes a layer count (clamped, with
a fixed fallback if this call fails - object counting is an enhancement over a fixed default,
never a hard dependency of the core separation feature).

Plain synchronous httpx.post, not the submit/poll pattern layer_decompose.py needs - SAM2's
automatic-mask-generation mode is fast enough (a shared image encoder pass plus a lightweight
per-point decoder) to fit well inside Modal's 150-second web-endpoint ceiling, same shape as
face_parsing.py's client.
"""

from __future__ import annotations

import io
import os

import httpx
import numpy as np
from PIL import Image

from ai_prepress.io import LoadedImage, to_unit_float


def count_objects(image: LoadedImage, endpoint: str | None = None, timeout: float = 60.0) -> int:
    endpoint = endpoint or os.environ.get("AI_PREPRESS_OBJECT_COUNT_URL", "")
    if not endpoint:
        raise ValueError(
            "no object-count endpoint configured - set AI_PREPRESS_OBJECT_COUNT_URL "
            "or pass endpoint= explicitly"
        )

    unit = to_unit_float(image.array)[..., :3]
    rgb_8bit = (np.clip(unit, 0.0, 1.0) * 255 + 0.5).astype(np.uint8)
    request_buffer = io.BytesIO()
    Image.fromarray(rgb_8bit).save(request_buffer, format="PNG")

    response = httpx.post(
        endpoint,
        files={"file": ("image.png", request_buffer.getvalue(), "image/png")},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["object_count"]
