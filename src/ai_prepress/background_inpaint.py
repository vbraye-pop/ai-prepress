"""Client for the background-inpaint Modal endpoint (see deploy/background_inpaint.py).

Plain synchronous httpx.post, same shape as ai_prepress.object_count and ai_prepress.face_parsing
- the server-side compute is capped to a bounded working resolution (MAX_INPAINT_EDGE) so a single
feed-forward pass fits comfortably inside Modal's 150s web-endpoint ceiling, no submit/poll split
needed.

Mask contract, enforced server-side (see deploy/background_inpaint.py's module docstring for why):
hard binary 0/255, single channel, same pixel dimensions as the source image, 255 = hole to fill.
A soft/feathered mask is rejected with a 400, not silently accepted - passing one here is a caller
bug, not a variant of the contract.
"""

from __future__ import annotations

import io
import os

import httpx
import numpy as np
from PIL import Image


def inpaint_background(
    source_image: np.ndarray,
    mask: np.ndarray,
    endpoint: str | None = None,
    timeout: float = 90.0,
) -> np.ndarray:
    """`source_image` is (H, W, 3) uint8 RGB. `mask` is (H, W) uint8, hard binary 0/255, 255 =
    hole to fill. Returns (H, W, 3) uint8 RGB - the source unchanged outside the mask, LaMa
    content (feathered at the boundary) inside it."""
    endpoint = endpoint or os.environ.get("AI_PREPRESS_BACKGROUND_INPAINT_URL", "")
    if not endpoint:
        raise ValueError(
            "no background-inpaint endpoint configured - set AI_PREPRESS_BACKGROUND_INPAINT_URL "
            "or pass endpoint= explicitly"
        )

    source_buffer = io.BytesIO()
    Image.fromarray(source_image).save(source_buffer, format="PNG")
    mask_buffer = io.BytesIO()
    Image.fromarray(mask).save(mask_buffer, format="PNG")

    response = httpx.post(
        endpoint,
        files={
            "file": ("source.png", source_buffer.getvalue(), "image/png"),
            "mask": ("mask.png", mask_buffer.getvalue(), "image/png"),
        },
        timeout=timeout,
    )
    response.raise_for_status()

    return np.array(Image.open(io.BytesIO(response.content)).convert("RGB"))
