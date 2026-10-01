import numpy as np
import pytest
from PIL import ImageCms

import ai_prepress.features.ai_crop as ai_crop_module
from ai_prepress.face_landmarks import FACE_OVAL, FaceLandmarks
from ai_prepress.features.ai_crop import (
    NoSubjectFoundError,
    apply_crop,
    compute_crop,
    detect_auto,
    detect_faces,
    detect_subject,
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


def test_compute_crop_honors_an_explicit_center_override():
    # same bbox, but told to center on its left edge instead of its own midpoint
    rect = compute_crop((1000, 1000), bbox=(400, 400, 600, 600), aspect_ratio=1.0, margin=0.0, center=(400, 500))
    assert (rect.x0 + rect.x1) / 2 == pytest.approx(400, abs=1)


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


# --- detect_subject: mocked instance_matte, same style as test_layer_separation.py --------------


def test_detect_subject_returns_one_candidate_from_alpha(monkeypatch):
    alpha = np.zeros((200, 300), dtype=np.uint8)
    alpha[40:160, 60:240] = 255
    monkeypatch.setattr(ai_crop_module.instance_matte, "matte_crops", lambda crops, endpoint=None: [alpha])

    detection = detect_subject(_image(200, 300))

    assert detection.mode_used == "subject"
    assert len(detection.candidates) == 1
    assert detection.candidates[0].bbox == (60, 40, 240, 160)
    # centroid of a solid rectangle is its own geometric center
    assert detection.candidates[0].center == pytest.approx((149.5, 99.5), abs=1)


def test_detect_subject_centroid_tracks_asymmetric_mass_not_bbox_center(monkeypatch):
    # an L-shaped mask: a big block in the top-left plus a thin sliver reaching into the
    # bottom-right corner - the bbox's own geometric center sits in the empty notch, but the
    # actual pixel mass is weighted heavily toward the top-left block
    alpha = np.zeros((200, 200), dtype=np.uint8)
    alpha[0:100, 0:100] = 255  # big block
    alpha[190:200, 190:200] = 255  # tiny far corner sliver, pulls the bbox out to 200x200
    monkeypatch.setattr(ai_crop_module.instance_matte, "matte_crops", lambda crops, endpoint=None: [alpha])

    detection = detect_subject(_image(200, 200))
    candidate = detection.candidates[0]

    bbox_center_x = (candidate.bbox[0] + candidate.bbox[2]) / 2
    # the real centroid is pulled toward the big block, well left of the bbox's own midpoint
    assert candidate.center[0] < bbox_center_x - 20


def test_detect_subject_raises_when_nothing_salient(monkeypatch):
    empty_alpha = np.zeros((200, 300), dtype=np.uint8)
    monkeypatch.setattr(ai_crop_module.instance_matte, "matte_crops", lambda crops, endpoint=None: [empty_alpha])

    with pytest.raises(NoSubjectFoundError):
        detect_subject(_image(200, 300))


# --- detect_faces: mocked face_landmarks, no MediaPipe model download in tests ------------------


def _fake_face(oval_box=(100.0, 50.0, 200.0, 150.0)) -> FaceLandmarks:
    points = np.zeros((478, 2), dtype=np.float32)
    x0, y0, x1, y1 = oval_box
    for idx in FACE_OVAL:
        points[idx] = [x0, y0]
    points[FACE_OVAL[0]] = [x1, y1]  # ensure both corners are represented
    return FaceLandmarks(points=points)


def test_detect_faces_returns_one_candidate_per_face_plus_an_all_faces_union(monkeypatch):
    # two non-overlapping faces, so their union is strictly larger than either one alone - an
    # earlier version of this fixture had the second face's bbox fully contain the first,
    # making the union exactly tie the second face's own area and picking the wrong primary
    left, right = _fake_face((100, 50, 150, 100)), _fake_face((200, 200, 500, 500))
    monkeypatch.setattr(ai_crop_module.face_landmarks, "detect_landmarks", lambda rgb, max_faces=32: [left, right])

    detection = detect_faces(_image())

    assert detection.mode_used == "face"
    assert len(detection.candidates) == 3
    assert detection.candidates[0].label == "Face 1"
    assert detection.candidates[1].label == "Face 2"
    assert detection.candidates[2].label == "All faces"
    # the union bbox spans both faces
    assert detection.candidates[2].bbox == (100, 50, 500, 500)
    # strictly larger than either individual face, so it wins the area-based primary selection
    assert detection.primary_index == 2


def test_detect_faces_single_face_has_no_all_faces_candidate(monkeypatch):
    monkeypatch.setattr(ai_crop_module.face_landmarks, "detect_landmarks", lambda rgb, max_faces=32: [_fake_face()])

    detection = detect_faces(_image())

    assert len(detection.candidates) == 1
    assert detection.primary_index == 0


def test_all_faces_centroid_is_weighted_toward_the_larger_face(monkeypatch):
    # a small face far to the left, a much larger face far to the right - an unweighted average
    # of the two faces' own centers would land exactly between them; the area-weighted centroid
    # should sit closer to the big one instead
    small = _fake_face((0, 0, 20, 20))
    big = _fake_face((900, 0, 1000, 100))
    monkeypatch.setattr(ai_crop_module.face_landmarks, "detect_landmarks", lambda rgb, max_faces=32: [small, big])

    detection = detect_faces(_image(200, 1000))
    small_candidate, big_candidate, all_faces = detection.candidates
    unweighted_midpoint_x = (small_candidate.center[0] + big_candidate.center[0]) / 2
    assert all_faces.center[0] > unweighted_midpoint_x


def test_detect_faces_raises_when_no_face_found(monkeypatch):
    monkeypatch.setattr(ai_crop_module.face_landmarks, "detect_landmarks", lambda rgb, max_faces=32: [])
    with pytest.raises(NoSubjectFoundError):
        detect_faces(_image())


# --- detect_auto: face first, subject fallback --------------------------------------------------


def test_detect_auto_uses_face_when_present(monkeypatch):
    monkeypatch.setattr(ai_crop_module.face_landmarks, "detect_landmarks", lambda rgb, max_faces=32: [_fake_face()])

    def fail_if_called(crops, endpoint=None):
        raise AssertionError("subject detection should not run when a face was found")

    monkeypatch.setattr(ai_crop_module.instance_matte, "matte_crops", fail_if_called)

    detection = detect_auto(_image())
    assert detection.mode_used == "face"


def test_detect_auto_falls_back_to_subject_when_no_face(monkeypatch):
    monkeypatch.setattr(ai_crop_module.face_landmarks, "detect_landmarks", lambda rgb, max_faces=32: [])
    alpha = np.zeros((200, 300), dtype=np.uint8)
    alpha[10:20, 10:20] = 255
    monkeypatch.setattr(ai_crop_module.instance_matte, "matte_crops", lambda crops, endpoint=None: [alpha])

    detection = detect_auto(_image(200, 300))
    assert detection.mode_used == "subject"


def test_detect_auto_raises_when_neither_source_finds_anything(monkeypatch):
    monkeypatch.setattr(ai_crop_module.face_landmarks, "detect_landmarks", lambda rgb, max_faces=32: [])
    empty_alpha = np.zeros((200, 300), dtype=np.uint8)
    monkeypatch.setattr(ai_crop_module.instance_matte, "matte_crops", lambda crops, endpoint=None: [empty_alpha])

    with pytest.raises(NoSubjectFoundError):
        detect_auto(_image(200, 300))
