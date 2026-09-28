"""Face bounding-box detection, used to crop a portrait before sending it to the face-parsing
model (see face_parsing.py). The parser is fine-tuned on CelebAMask-HQ, which is tightly-cropped
face-filling-the-frame data - feeding it a full environmental photo puts the face at a few percent
of the 512x512 budget the model actually sees, which is what was producing the bad segmentations
(background misclassified as face parts, blocky/crude masks). Cropping first fixes that with the
same model and the same weights, no retraining involved.

mediapipe is pinned below 1.0 in pyproject.toml - 1.0.x hard-crashes on macOS/arm64 for any Tasks
graph with a detector calculator, see the pin comment there.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import numpy as np

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_full_range/float16/latest/blaze_face_full_range.tflite"
)
_MODEL_PATH = Path.home() / ".cache" / "ai-prepress" / "blaze_face_full_range.tflite"

# full-range (not short-range) variant - short-range is tuned for selfie-distance faces filling
# most of the frame, full-range handles the smaller in-frame faces typical of environmental photos
_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


def _model_path() -> Path:
    if not _MODEL_PATH.exists():
        response = httpx.get(_MODEL_URL, timeout=30.0, follow_redirects=True)
        response.raise_for_status()
        _MODEL_PATH.write_bytes(response.content)
    return _MODEL_PATH


def detect_face_box(rgb: np.ndarray) -> tuple[int, int, int, int] | None:
    """Returns (x0, y0, x1, y1) of the highest-confidence detected face, or None if none found."""
    import mediapipe as mp
    from mediapipe.tasks.python import core, vision

    base_options = core.base_options.BaseOptions(model_asset_path=str(_model_path()))
    detector = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(base_options=base_options))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = detector.detect(mp_image)

    if not result.detections:
        return None

    best = max(result.detections, key=lambda d: d.categories[0].score)
    bb = best.bounding_box
    return bb.origin_x, bb.origin_y, bb.origin_x + bb.width, bb.origin_y + bb.height


def square_crop_around(
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    margin: float = 1.4,
    top_bias: float = 0.15,
) -> tuple[int, int, int, int]:
    """Expands a detected face box into a square crop with margin, shifted up slightly to leave
    room for hair/forehead above the box (BlazeFace boxes hug eyes-nose-mouth-chin, not hair).

    margin=1.4 was picked empirically, not from a rule of thumb: on a real test photo, 1.4x gave
    the best eye/brow/nose/lip detection (the classes that matter for downstream retouching work)
    - looser margins (2x-3x) kept more hair/torso context but shrank the face's share of the
    model's fixed 512x512 input budget enough to lose eye and brow pixels almost entirely.

    Clamped to image bounds by shifting (not shrinking) the box, so the output is always square -
    a squashed non-square crop would go through the model's own resize-to-512x512 distorted.
    """
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    size = max(w, h)
    cx = x0 + w / 2
    cy = y0 + h / 2 - top_bias * size * margin

    crop_size = margin * size
    cx0, cy0 = cx - crop_size / 2, cy - crop_size / 2
    cx1, cy1 = cx0 + crop_size, cy0 + crop_size

    width, height = image_size
    if cx0 < 0:
        cx1 -= cx0
        cx0 = 0
    if cy0 < 0:
        cy1 -= cy0
        cy0 = 0
    if cx1 > width:
        cx0 -= cx1 - width
        cx1 = width
    if cy1 > height:
        cy0 -= cy1 - height
        cy1 = height

    return (
        max(0, int(cx0)),
        max(0, int(cy0)),
        min(width, int(cx1)),
        min(height, int(cy1)),
    )
