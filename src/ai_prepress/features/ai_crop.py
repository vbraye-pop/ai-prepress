"""AI Crop: deterministic crop geometry driven by one ML-derived bounding box.

Matches the project brief's Feature 3 spec directly: Capture One's own AI Crop is mostly
deterministic aspect-ratio/margin/alignment math, with exactly one piece of ML feeding it a
Subject, Face, or Auto bounding box - not a learned end-to-end crop. Everything in compute_crop
below is plain geometry; the only model involvement is finding the box (and, for Auto, deciding
which kind of box to look for) in the first place.

Detection is split from geometry on purpose: detect_subject/detect_faces/detect_auto are the only
functions that touch a model (remote for Subject, local for Face), and they run exactly ONCE per
photo. compute_crop is pure arithmetic with no model involvement at all, cheap enough to run on
every mouse-drag - see ai_prepress.api.main's /api/ai-crop/detect + /api/ai-crop/apply split and
ui/app.js's JS port of compute_crop for why that split exists: a user adjusting aspect ratio or
margin, or dragging the crop rectangle by hand, should never re-trigger a remote BiRefNet call.

Three bbox sources:
- "subject": ai_prepress.instance_matte (BiRefNet_HR-matting) run on the WHOLE photo, not a
  pre-cropped region - it's a general salient-object matting model, no detection/segmentation
  stage needed first, so this is the cheapest possible remote call for a crop bbox.
- "face": ai_prepress.face_landmarks, entirely local, no remote call. Returns EVERY detected face
  as its own candidate (not just one) - see Detection below - so a caller (the UI) can let the
  person pick which face to crop around instead of guessing a blind index.
- "auto": face first, subject as fallback - same enhancement-over-fallback discipline
  features.layer_separation already uses for object_count/layer_naming elsewhere in this project.
  A photo either has a face worth centering on or it doesn't; when it doesn't, Subject is the
  right fallback, not a hard failure.

Centering: each candidate carries both its bbox AND a `center`, which for Subject mode is the
alpha mask's own pixel-mass centroid, not the bbox's geometric midpoint. For an asymmetric subject
(an L-shaped object, a person with one arm raised) the mass centroid sits where the content
actually is; saliency-based auto-crop tools generally weight by actual detected content rather
than its bounding box for exactly this reason. compute_crop defaults to the bbox's own geometric
center when no explicit center is given, so this is additive, not a forced behavior change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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


@dataclass
class Candidate:
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]  # mass-weighted when available, else the bbox's own midpoint
    label: str | None = None  # "Face 1", "Face 2"... for face candidates; None for subject


@dataclass
class Detection:
    mode_used: str  # "subject" | "face" - which one actually ran (relevant for "auto")
    candidates: list[Candidate] = field(default_factory=list)
    primary_index: int = 0  # which candidate to default-select - largest face, or the only subject


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


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    x0, y0, x1, y1 = bbox
    return max(0, x1 - x0) * max(0, y1 - y0)


def _alpha_centroid(alpha: np.ndarray) -> tuple[float, float] | None:
    """Mass-weighted center of the alpha's opaque region, not its bbox's geometric center - see
    module docstring for why this is a meaningfully different (and often better) crop anchor."""
    mask = alpha > BBOX_ALPHA_THRESHOLD
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return (float(xs.mean()), float(ys.mean()))


def detect_subject(image: LoadedImage, endpoint: str | None = None) -> Detection:
    """The main salient subject, via BiRefNet_HR run on the whole photo at once. Always exactly
    one candidate - BiRefNet mattes the single most salient object, it doesn't enumerate several."""
    [alpha] = instance_matte.matte_crops([_to_rgb_8bit(image)], endpoint=endpoint)
    bbox = _bbox_from_alpha(alpha)
    if bbox is None:
        raise NoSubjectFoundError("no salient subject found in this image")
    center = _alpha_centroid(alpha) or _bbox_center(bbox)
    return Detection(mode_used="subject", candidates=[Candidate(bbox=bbox, center=center)], primary_index=0)


def _all_faces_candidate(candidates: list[Candidate]) -> Candidate:
    """A combined candidate spanning every detected face - the union of their bboxes, centered
    on their AREA-WEIGHTED centroid rather than a plain average of face centers. This is the same
    approach thumbor's own smart-crop takes for multiple detected focal points (confirmed against
    its own docs, not guessed): a small background face shouldn't pull a group crop's center as
    much as a large foreground one."""
    total_area = sum(_bbox_area(c.bbox) for c in candidates) or 1
    union_bbox = (
        min(c.bbox[0] for c in candidates),
        min(c.bbox[1] for c in candidates),
        max(c.bbox[2] for c in candidates),
        max(c.bbox[3] for c in candidates),
    )
    weighted_cx = sum(c.center[0] * _bbox_area(c.bbox) for c in candidates) / total_area
    weighted_cy = sum(c.center[1] * _bbox_area(c.bbox) for c in candidates) / total_area
    return Candidate(bbox=union_bbox, center=(weighted_cx, weighted_cy), label="All faces")


def detect_faces(image: LoadedImage, max_faces: int = 32) -> Detection:
    """Every detected face as its own candidate (left-to-right order, same as /api/face-regions),
    via local MediaPipe landmarks - no remote call. When more than one face is found, an extra
    "All faces" candidate is appended (see _all_faces_candidate) for group shots.

    `primary_index` defaults to the largest-area candidate. For a single face that's just the
    face itself; for several, the "All faces" union bbox is by construction at least as large as
    any one face, so it becomes primary automatically - framing the whole group by default, with
    any individual face still one click away, rather than an arbitrary "biggest face wins" guess.
    """
    faces = face_landmarks.detect_landmarks(_to_rgb_8bit(image), max_faces=max_faces)
    if not faces:
        raise NoSubjectFoundError("no face found in this image")

    candidates = []
    for index, face in enumerate(faces):
        oval = face.subset(face_landmarks.FACE_OVAL)
        x0, y0 = oval.min(axis=0)
        x1, y1 = oval.max(axis=0)
        bbox = (int(x0), int(y0), int(x1), int(y1))
        center = (float(oval[:, 0].mean()), float(oval[:, 1].mean()))
        candidates.append(Candidate(bbox=bbox, center=center, label=f"Face {index + 1}"))

    if len(candidates) > 1:
        candidates.append(_all_faces_candidate(candidates))

    primary_index = max(range(len(candidates)), key=lambda i: _bbox_area(candidates[i].bbox))
    return Detection(mode_used="face", candidates=candidates, primary_index=primary_index)


def detect_auto(image: LoadedImage, instance_matte_endpoint: str | None = None, max_faces: int = 32) -> Detection:
    """Face first, Subject as fallback - a photo either has a face worth centering on or it
    doesn't, and when it doesn't, Subject is the sane answer rather than a hard failure. Same
    enhancement-over-fallback shape as object_count/layer_naming in features.layer_separation."""
    try:
        return detect_faces(image, max_faces=max_faces)
    except NoSubjectFoundError:
        return detect_subject(image, endpoint=instance_matte_endpoint)


def compute_crop(
    image_size: tuple[int, int],
    bbox: tuple[int, int, int, int],
    aspect_ratio: float,
    margin: float = 0.15,
    center: tuple[float, float] | None = None,
) -> CropRect:
    """Grow `bbox` by `margin` (a fraction of the bbox's own larger side, added on every edge),
    then expand it to `aspect_ratio` while keeping it centered on `center` (defaults to the
    bbox's own geometric center when not given - see Detection/Candidate for when a mass-weighted
    center is available instead), clamped to fit inside `image_size` (width, height).

    Clamping can shift the crop off-center - a subject near one edge of the frame can't have a
    crop that's both centered on it and entirely inside the frame. That's the correct, honest
    tradeoff: a crop that cuts into the subject to stay centered would be worse than one that's
    off-center but keeps the whole detected region in frame.
    """
    img_w, img_h = image_size
    x0, y0, x1, y1 = bbox
    bbox_w, bbox_h = x1 - x0, y1 - y0
    cx, cy = center if center is not None else _bbox_center(bbox)

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
