import numpy as np
import pytest
from PIL import ImageCms

import ai_prepress.features.ai_crop as ai_crop_module
from ai_prepress.face_landmarks import FACE_OVAL, FaceLandmarks
from ai_prepress.features.ai_crop import (
    NoSubjectFoundError,
    apply_crop,
    compute_crop,
    face_bbox,
    subject_bbox,
)
from ai_prepress.io import LoadedImage

_SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _image(height=200, width=300) -> LoadedImage:
    array = np.zeros((height, width, 3), dtype=np.uint8)
    return LoadedImage(array=array, icc_profile=_SRGB, bit_depth=8)


# --- compute_crop: pure geometry, no mocking needed --------------------------------------------


def test_compute_crop_centers_on_bbox_with_margin():
    rect = compute_crop((1000, 1000), bbox=(400, 400, 600, 600), aspect_ratio=1.0, margin=0.5)
    # bbox is 200x200, centered at (500, 500); margin=0.5 pads 100px every side -> 400x400 square
    assert rect.width == rect.height == 400
    assert (rect.x0 + rect.x1) / 2 == pytest.approx(500, abs=1)
    assert (rect.y0 + rect.y1) / 2 == pytest.approx(500, abs=1)


def test_compute_crop_matches_requested_aspect_ratio():
    rect = compute_crop((2000, 2000), bbox=(900, 900, 1100, 1100), aspect_ratio=16 / 9, margin=0.2)
    assert rect.width / rect.height == pytest.approx(16 / 9, rel=0.02)


def test_compute_crop_clamps_to_image_bounds_near_an_edge():
    # subject sits right in the top-left corner - a centered crop would run off both edges
    rect = compute_crop((1000, 1000), bbox=(0, 0, 100, 100), aspect_ratio=1.0, margin=0.3)
    assert rect.x0 >= 0 and rect.y0 >= 0
    assert rect.x1 <= 1000 and rect.y1 <= 1000
    # clamping shifts the crop off-center rather than cutting into the detected region
    assert rect.x0 <= 0 and rect.x1 >= 100
    assert rect.y0 <= 0 and rect.y1 >= 100


def test_compute_crop_never_exceeds_the_source_image():
    # an enormous margin would naively want a crop far bigger than the source image itself
    rect = compute_crop((500, 400), bbox=(200, 150, 300, 250), aspect_ratio=1.0, margin=50.0)
    assert rect.width <= 500
    assert rect.height <= 400


def test_apply_crop_slices_array_and_keeps_profile():
    image = _image(200, 300)
    image.array[50:150, 80:180] = 255
    rect = ai_crop_module.CropRect(x0=80, y0=50, x1=180, y1=150)

    cropped = apply_crop(image, rect)

    assert cropped.array.shape == (100, 100, 3)
    assert (cropped.array == 255).all()
    assert cropped.icc_profile == image.icc_profile


def test_apply_crop_falls_back_to_srgb_for_an_untagged_source():
    image = LoadedImage(array=np.zeros((200, 300, 3), dtype=np.uint8), icc_profile=None, bit_depth=8)
    rect = ai_crop_module.CropRect(x0=0, y0=0, x1=50, y1=50)

    cropped = apply_crop(image, rect)

    assert cropped.icc_profile is not None


# --- subject_bbox: mocked instance_matte, same style as test_layer_separation.py ----------------


def test_subject_bbox_returns_bbox_from_alpha(monkeypatch):
    alpha = np.zeros((200, 300), dtype=np.uint8)
    alpha[40:160, 60:240] = 255
    monkeypatch.setattr(ai_crop_module.instance_matte, "matte_crops", lambda crops, endpoint=None: [alpha])

    bbox = subject_bbox(_image(200, 300))
    assert bbox == (60, 40, 240, 160)


def test_subject_bbox_raises_when_nothing_salient(monkeypatch):
    empty_alpha = np.zeros((200, 300), dtype=np.uint8)
    monkeypatch.setattr(ai_crop_module.instance_matte, "matte_crops", lambda crops, endpoint=None: [empty_alpha])

    with pytest.raises(NoSubjectFoundError):
        subject_bbox(_image(200, 300))


# --- face_bbox: mocked face_landmarks, no MediaPipe model download in tests ---------------------


def _fake_face(width=300, height=200) -> FaceLandmarks:
    points = np.zeros((478, 2), dtype=np.float32)
    # park the face-oval indices at a known rectangle so the resulting bbox is predictable
    oval_box = np.array([100.0, 50.0, 200.0, 150.0])  # x0, y0, x1, y1
    for idx in FACE_OVAL:
        points[idx] = [oval_box[0], oval_box[1]]
    points[FACE_OVAL[0]] = [oval_box[2], oval_box[3]]  # ensure both corners are represented
    return FaceLandmarks(points=points)


def test_face_bbox_returns_oval_bbox(monkeypatch):
    monkeypatch.setattr(
        ai_crop_module.face_landmarks, "detect_landmarks", lambda rgb, max_faces=1: [_fake_face()]
    )
    bbox = face_bbox(_image(200, 300))
    assert bbox == (100, 50, 200, 150)


def test_face_bbox_raises_when_no_face_found(monkeypatch):
    monkeypatch.setattr(ai_crop_module.face_landmarks, "detect_landmarks", lambda rgb, max_faces=1: [])
    with pytest.raises(NoSubjectFoundError):
        face_bbox(_image())


def test_face_bbox_honors_face_index(monkeypatch):
    first, second = _fake_face(), _fake_face()
    second.points = second.points + 500  # a clearly different face
    monkeypatch.setattr(
        ai_crop_module.face_landmarks, "detect_landmarks", lambda rgb, max_faces=2: [first, second]
    )
    bbox = face_bbox(_image(), face_index=1)
    assert bbox != (100, 50, 200, 150)
