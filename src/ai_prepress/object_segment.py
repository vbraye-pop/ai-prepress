"""Client for the object-segment Modal endpoint (see deploy/object_segment.py).

Second stage of the per-instance separation pipeline: turns each candidate box from
ai_prepress.object_detect into a pixel-accurate alpha mask via box-prompted SAM2.

Coordinate-space contract - the invariant sentence below is shared verbatim with
ai_prepress.object_detect (its sibling in this pipeline stage): boxes are absolute pixel XYXY
floats in the coordinate space of the image EXACTLY as passed in, no resizing on either side of
the wire.

Plain synchronous httpx.post, same shape as ai_prepress.object_count's client - a handful of box
prompts against one already-encoded image comfortably fits Modal's 150-second web-endpoint
ceiling. The endpoint returns a zip (one mask per box, plus a scores manifest), so unpacking it
mirrors ai_prepress.layer_decompose's own _unpack_zip helper rather than object_count's single
JSON response.
"""

from __future__ import annotations

import io
import json
import os
import zipfile

import httpx
import numpy as np
from PIL import Image

from ai_prepress.io import LoadedImage, to_unit_float


def segment_boxes(
    image: LoadedImage,
    boxes: list[list[float]],
    endpoint: str | None = None,
    timeout: float = 60.0,
) -> list[np.ndarray]:
    """Returns one (H, W) uint8 mask per input box (255=object, 0=background), same order as
    `boxes`. `boxes` are absolute pixel XYXY in `image`'s own coordinate space - see module
    docstring."""
    endpoint = endpoint or os.environ.get("AI_PREPRESS_OBJECT_SEGMENT_URL", "")
    if not endpoint:
        raise ValueError(
            "no object-segment endpoint configured - set AI_PREPRESS_OBJECT_SEGMENT_URL "
            "or pass endpoint= explicitly"
        )

    unit = to_unit_float(image.array)[..., :3]
    rgb_8bit = (np.clip(unit, 0.0, 1.0) * 255 + 0.5).astype(np.uint8)
    request_buffer = io.BytesIO()
    Image.fromarray(rgb_8bit).save(request_buffer, format="PNG")

    response = httpx.post(
        endpoint,
        files={"file": ("image.png", request_buffer.getvalue(), "image/png")},
        data={"boxes": json.dumps(boxes)},
        timeout=timeout,
    )
    response.raise_for_status()
    return _unpack_zip(response.content)


def _unpack_zip(zip_bytes: bytes) -> list[np.ndarray]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        mask_names = sorted(name for name in zf.namelist() if name.startswith("mask_"))
        return [np.array(Image.open(io.BytesIO(zf.read(name)))) for name in mask_names]
