"""Client for the face-region-parsing Modal endpoint (see deploy/face_parsing.py).

Calls a plain HTTP endpoint rather than importing the Modal SDK directly, so swapping the
backend later - a different model, a different platform, a local model - is a URL change, not a
rewrite. Same principle as the rest of this project's model-tier calls.

The remote model only ever sees 8-bit RGB (it's a standard vision transformer, it can't use more
precision than that regardless of what's sent), so the source image is deliberately downcast to
8-bit before sending - not a silent loss, the model just never had 16-bit to lose in the first
place.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass

import httpx
import numpy as np
from PIL import Image

from ai_prepress.io import LoadedImage, to_unit_float

# Must match deploy/face_parsing.py's LABELS exactly - kept as a separate copy rather than a
# shared import since the deploy script and the local package don't share a dependency tree
# (deploy/ needs modal+transformers+torch, src/ deliberately doesn't).
LABELS = [
    "background", "skin", "nose", "eye_g", "l_eye", "r_eye", "l_brow", "r_brow",
    "l_ear", "r_ear", "mouth", "u_lip", "l_lip", "hair", "hat", "ear_r", "neck_l",
    "neck", "cloth",
]


@dataclass
class FaceParsingResult:
    labels: np.ndarray  # (H, W) uint8, each value indexes into LABELS
    label_names: list[str]

    def mask_for(self, *names: str) -> np.ndarray:
        """Boolean (H, W) mask, True wherever the pixel belongs to any of the given labels."""
        indices = [self.label_names.index(name) for name in names]
        return np.isin(self.labels, indices)


def parse_face(image: LoadedImage, endpoint: str | None = None, timeout: float = 60.0) -> FaceParsingResult:
    endpoint = endpoint or os.environ.get("AI_PREPRESS_FACE_PARSING_URL", "")
    if not endpoint:
        raise ValueError(
            "no face-parsing endpoint configured - set AI_PREPRESS_FACE_PARSING_URL "
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

    labels = np.array(Image.open(io.BytesIO(response.content)))
    return FaceParsingResult(labels=labels, label_names=LABELS)
