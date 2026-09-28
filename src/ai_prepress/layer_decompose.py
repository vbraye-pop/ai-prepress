"""Client for the layer-separation Modal endpoint (see deploy/layer_separation.py).

Calls plain HTTP endpoints rather than the Modal SDK directly, same principle as
ai_prepress.face_parsing - swapping the backend later is a URL change, not a rewrite.

The remote model has no synchronous web endpoint for the decomposition itself: Modal web
functions carry a hard 150-second HTTP request ceiling, and a cold-started multi-layer diffusion
call can plausibly exceed that. deploy/layer_separation.py instead exposes two fast endpoints
(`submit` spawns the work and returns immediately, `result` is a cheap non-blocking status check)
wrapping one long-running internal Modal method. This module hides that spawn-and-poll mechanics
behind a single ordinary blocking call - everything above it (features/, api/main.py, the UI) sees
the same shape as any other remote call in this project.
"""

from __future__ import annotations

import io
import os
import time
import zipfile
from dataclasses import dataclass

import httpx
import numpy as np
from PIL import Image

from ai_prepress.io import LoadedImage, to_unit_float


@dataclass
class LayerSeparationResult:
    background: np.ndarray  # (H, W, 3) uint8 - genuinely model-generated, no higher-precision source exists
    layer_alphas: list[np.ndarray]  # (H, W) uint8 each - compositing against source RGB happens in features/


def decompose_layers(
    image: LoadedImage,
    endpoint: str | None = None,
    timeout: float = 600.0,
    poll_interval: float = 2.0,
    layers: int | None = None,
) -> LayerSeparationResult:
    """`layers=None` leaves the decision to the server's own DEFAULT_LAYER_COUNT - callers that
    want the real per-photo count (features.layer_separation, via object_count) pass it
    explicitly. Kept optional, not required, so this function stays usable standalone (direct
    calls, tests) without needing a whole object-counting pipeline first."""
    endpoint = endpoint or os.environ.get("AI_PREPRESS_LAYER_SEPARATION_URL", "")
    if not endpoint:
        raise ValueError(
            "no layer-separation endpoint configured - set AI_PREPRESS_LAYER_SEPARATION_URL "
            "or pass endpoint= explicitly"
        )
    endpoint = endpoint.rstrip("/")

    unit = to_unit_float(image.array)[..., :3]
    rgb_8bit = (np.clip(unit, 0.0, 1.0) * 255 + 0.5).astype(np.uint8)
    request_buffer = io.BytesIO()
    Image.fromarray(rgb_8bit).save(request_buffer, format="PNG")

    submit_data = {"layers": layers} if layers is not None else {}
    submit_response = httpx.post(
        f"{endpoint}/submit",
        files={"file": ("image.png", request_buffer.getvalue(), "image/png")},
        data=submit_data,
        timeout=30.0,
    )
    submit_response.raise_for_status()
    call_id = submit_response.json()["call_id"]

    deadline = time.monotonic() + timeout
    while True:
        try:
            result_response = httpx.get(f"{endpoint}/result", params={"call_id": call_id}, timeout=30.0)
            result_response.raise_for_status()
        except httpx.TransportError:
            # a single flaky poll over a multi-minute operation shouldn't abort the whole call -
            # confirmed against a real deployment: an isolated httpx.ReadTimeout on one /result
            # poll killed an otherwise-successful run before this retry was added. Still bounded
            # by the overall `timeout` below, so a genuinely dead endpoint still gives up.
            if time.monotonic() >= deadline:
                raise
            time.sleep(poll_interval)
            continue

        if result_response.headers.get("content-type", "").startswith("application/zip"):
            return _unpack_zip(result_response.content)

        status = result_response.json().get("status")
        if status == "expired":
            raise RuntimeError(f"layer-separation call {call_id} expired before it was collected")

        if time.monotonic() >= deadline:
            raise TimeoutError(f"layer-separation call {call_id} did not complete within {timeout}s")
        time.sleep(poll_interval)


def _unpack_zip(zip_bytes: bytes) -> LayerSeparationResult:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        background = np.array(Image.open(io.BytesIO(zf.read("background.png"))).convert("RGB"))
        alpha_names = sorted(name for name in zf.namelist() if name.startswith("alpha_"))
        layer_alphas = [np.array(Image.open(io.BytesIO(zf.read(name)))) for name in alpha_names]
    return LayerSeparationResult(background=background, layer_alphas=layer_alphas)
