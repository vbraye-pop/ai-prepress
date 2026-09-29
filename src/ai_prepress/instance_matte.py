"""Client for the instance-matte Modal endpoint (see deploy/instance_matte.py).

Third stage of the per-instance separation pipeline: turns each per-instance crop from
ai_prepress.object_segment (a mask's own bounding-box crop of the source photo) into a
high-resolution alpha matte via BiRefNet_HR-matting.

Crops are plain (H, W, 3) uint8 RGB arrays, not ai_prepress.io.LoadedImage - this stage never
touches a file's own bit depth or ICC profile, it only ever sees already-cropped 8-bit regions cut
out of a source photo by an earlier pipeline stage, the same "raw array in, raw array out" shape
features.layer_separation's own crop_rgb slices use internally.

No client-side image prep happens here (contrast ai_prepress.layer_naming's RGBA-to-white
flattening) - deploy/instance_matte.py's own squash-vs-pad test found the model's plain
non-aspect-preserving (2048, 2048) resize sharper than a pad-to-square alternative on a real
non-square crop, so every crop is sent through exactly as given (see that module's docstring for
the measurement). Sequential per-crop matting happens server-side, not here - see that module's
own OOM warning for why.
"""

from __future__ import annotations

import io
import os
import zipfile

import httpx
import numpy as np
from PIL import Image


def matte_crops(
    crops: list[np.ndarray], endpoint: str | None = None, timeout: float = 120.0
) -> list[np.ndarray]:
    """Returns one (H, W) uint8 alpha array per input crop, same order as `crops`."""
    if not crops:
        return []

    endpoint = endpoint or os.environ.get("AI_PREPRESS_INSTANCE_MATTE_URL", "")
    if not endpoint:
        raise ValueError(
            "no instance-matte endpoint configured - set AI_PREPRESS_INSTANCE_MATTE_URL "
            "or pass endpoint= explicitly"
        )

    files = [("files", (f"crop_{i:02d}.png", _encode_png(crop), "image/png")) for i, crop in enumerate(crops)]

    response = httpx.post(endpoint, files=files, timeout=timeout)
    response.raise_for_status()
    return _unpack_zip(response.content)


def _encode_png(crop: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(crop).save(buffer, format="PNG")
    return buffer.getvalue()


def _unpack_zip(zip_bytes: bytes) -> list[np.ndarray]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        alpha_names = sorted(name for name in zf.namelist() if name.startswith("alpha_"))
        return [np.array(Image.open(io.BytesIO(zf.read(name)))) for name in alpha_names]
