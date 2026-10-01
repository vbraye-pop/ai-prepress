"""AI Crop: deterministic crop geometry driven by one ML-derived bounding box.

Matches the project brief's Feature 3 spec directly: Capture One's own AI Crop is mostly
deterministic aspect-ratio/margin/alignment math, with exactly one piece of ML feeding it a
Subject or Face bounding box - not a learned end-to-end crop. Everything in compute_crop below
is plain geometry; the only model involvement is finding the box in the first place.

Two bbox sources, both reusing infrastructure this project already has deployed rather than
standing up anything new:
- "subject": ai_prepress.instance_matte run on the WHOLE photo, not a pre-cropped region.
  BiRefNet_HR-matting is a general salient-object matting model - it doesn't need a detection or
  segmentation stage first the way Layer Separation's per-instance pipeline does, so calling it
  directly on a full frame is the cheapest possible remote call for a crop bbox (one image, one
  model, no Grounding DINO/SAM2 stage). This is exactly the original brief's "reuse the BiRefNet
  mask - its bounding box IS the Subject detection" idea, just pointed at the model this project
  actually ended up deploying for masking (BiRefNet_HR) instead of plain BiRefNet.
- "face": ai_prepress.face_landmarks, entirely local, no remote call at all - reuses the exact
  FACE_OVAL landmark group /api/face-regions already computes a bbox from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import ImageCms

from ai_prepress import face_landmarks, instance_matte
from ai_prepress.io import LoadedImage, to_unit_float

# majority-opacity, not "any nonzero" - same convention and same reason as
# features.layer_separation's own BBOX_ALPHA_THRESHOLD: a real deployment's alpha output carries
# low-level noise spread across nearly the whole frame, not just the object's own soft edge, and
# a low threshold picks that noise up as if it were part of the subject
BBOX_ALPHA_THRESHOLD = 127


class NoSubjectFoundError(Exception):
    """Raised when the requested bbox source found nothing to crop around."""


@dataclass
class CropRect:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def _to_rgb_8bit(image: LoadedImage) -> np.ndarray:
    unit = to_unit_float(image.array)[..., :3]
    return (np.clip(unit, 0.0, 1.0) * 255 + 0.5).astype(np.uint8)


def _bbox_from_alpha(alpha: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(alpha > BBOX_ALPHA_THRESHOLD)
    if coords.size == 0:
        return None
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    return (int(x0), int(y0), int(x1) + 1, int(y1) + 1)


def subject_bbox(image: LoadedImage, endpoint: str | None = None) -> tuple[int, int, int, int]:
    """The main salient subject's bbox, via BiRefNet_HR run on the whole photo at once."""
    [alpha] = instance_matte.matte_crops([_to_rgb_8bit(image)], endpoint=endpoint)
    bbox = _bbox_from_alpha(alpha)
    if bbox is None:
        raise NoSubjectFoundError("no salient subject found in this image")
    return bbox


def face_bbox(image: LoadedImage, face_index: int = 0) -> tuple[int, int, int, int]:
    """The chosen face's bbox (left-to-right order, same as /api/face-regions), via local
    MediaPipe landmarks - no remote call."""
    faces = face_landmarks.detect_landmarks(_to_rgb_8bit(image), max_faces=face_index + 1)
    if len(faces) <= face_index:
        raise NoSubjectFoundError(f"no face at index {face_index} - {len(faces)} face(s) found")
    oval = faces[face_index].subset(face_landmarks.FACE_OVAL)
    x0, y0 = oval.min(axis=0)
    x1, y1 = oval.max(axis=0)
    return (int(x0), int(y0), int(x1), int(y1))


def compute_crop(
    image_size: tuple[int, int],
    bbox: tuple[int, int, int, int],
    aspect_ratio: float,
    margin: float = 0.15,
) -> CropRect:
    """Grow `bbox` by `margin` (a fraction of the bbox's own larger side, added on every edge),
    then expand it to `aspect_ratio` while keeping it centered on the bbox's own center, clamped
    to fit inside `image_size` (width, height).

    Clamping can shift the crop off-center - a subject near one edge of the frame can't have a
    crop that's both centered on it and entirely inside the frame. That's the correct, honest
    tradeoff: a crop that cuts into the subject to stay centered would be worse than one that's
    off-center but keeps the whole detected region in frame.
    """
    img_w, img_h = image_size
    x0, y0, x1, y1 = bbox
    bbox_w, bbox_h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

    pad = margin * max(bbox_w, bbox_h)
    padded_w, padded_h = bbox_w + 2 * pad, bbox_h + 2 * pad

    if padded_w / padded_h > aspect_ratio:
        crop_w = padded_w
        crop_h = crop_w / aspect_ratio
    else:
        crop_h = padded_h
        crop_w = crop_h * aspect_ratio

    # can't ask for a crop bigger than the source image itself along either axis
    scale = min(1.0, img_w / crop_w, img_h / crop_h)
    crop_w, crop_h = crop_w * scale, crop_h * scale

    left = min(max(cx - crop_w / 2, 0.0), img_w - crop_w)
    top = min(max(cy - crop_h / 2, 0.0), img_h - crop_h)

    rx0, ry0 = round(left), round(top)
    rx1 = min(img_w, rx0 + round(crop_w))
    ry1 = min(img_h, ry0 + round(crop_h))
    return CropRect(x0=rx0, y0=ry0, x1=rx1, y1=ry1)


def apply_crop(image: LoadedImage, rect: CropRect) -> LoadedImage:
    icc_profile = image.icc_profile
    if icc_profile is None:
        # io.save() requires one - same fallback match_look.py/retouch_faces.py/
        # layer_separation.py all use for an untagged source
        icc_profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    return LoadedImage(
        array=image.array[rect.y0 : rect.y1, rect.x0 : rect.x1],
        icc_profile=icc_profile,
        bit_depth=image.bit_depth,
    )
