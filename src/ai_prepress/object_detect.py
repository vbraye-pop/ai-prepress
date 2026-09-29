"""Client for the object-detection Modal endpoint (see deploy/object_detect.py).

Grounding DINO open-vocabulary detection, the first stage of the new per-instance separation
pipeline - see deploy/object_detect.py's module docstring for the model and inference details.

Coordinate-space contract, shared verbatim with the endpoint's own docstring and with
ai_prepress/object_segment.py (its sibling in this pipeline stage): returned boxes are absolute
pixel XYXY floats in the coordinate space of `image` EXACTLY as passed in, no resizing on either
side of the wire.

Plain synchronous httpx.post, single forward pass per call - same shape as object_count.py's
client, not the submit/poll pattern layer_decompose.py needs.
"""

from __future__ import annotations

import io
import os

import httpx
import numpy as np
from PIL import Image

from ai_prepress.io import LoadedImage, to_unit_float


def detect_objects(
    image: LoadedImage,
    text_prompt: str,
    endpoint: str | None = None,
    box_threshold: float = 0.4,
    text_threshold: float = 0.3,
    timeout: float = 60.0,
) -> list[dict]:
    endpoint = endpoint or os.environ.get("AI_PREPRESS_OBJECT_DETECT_URL", "")
    if not endpoint:
        raise ValueError(
            "no object-detect endpoint configured - set AI_PREPRESS_OBJECT_DETECT_URL "
            "or pass endpoint= explicitly"
        )

    unit = to_unit_float(image.array)[..., :3]
    rgb_8bit = (np.clip(unit, 0.0, 1.0) * 255 + 0.5).astype(np.uint8)
    request_buffer = io.BytesIO()
    Image.fromarray(rgb_8bit).save(request_buffer, format="PNG")

    response = httpx.post(
        endpoint,
        files={"file": ("image.png", request_buffer.getvalue(), "image/png")},
        data={
            "text_prompt": text_prompt,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["detections"]
