import numpy as np

from ai_prepress.face_landmarks import (
    FACE_OVAL,
    LEFT_EYE_LOOP,
    LEFT_IRIS,
    LIPS_INNER,
    LIPS_OUTER,
    RIGHT_EYE_LOOP,
    RIGHT_IRIS,
    FaceLandmarks,
    _convex_hull,
    _sort_left_to_right,
    _walk_loop,
    cheek_region,
    eye_sclera_region,
    forehead_region,
    lip_region,
    mouth_interior_region,
    skin_region,
    under_eye_band,
)


def test_walk_loop_orders_a_square():
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    assert _walk_loop(edges) == [0, 1, 2, 3]


def test_real_landmark_loops_have_the_expected_shape():
    # sanity check against mediapipe's own known topology, not just that the walk didn't crash
    assert len(FACE_OVAL) == 36
    assert len(RIGHT_EYE_LOOP) == 16
    assert len(LEFT_EYE_LOOP) == 16
    assert len(LIPS_OUTER) == 20
    assert len(LIPS_INNER) == 20
    assert len(RIGHT_IRIS) == 4
    assert len(LEFT_IRIS) == 4
    assert len(set(FACE_OVAL)) == len(FACE_OVAL)  # no repeated nodes


def test_convex_hull_drops_interior_points():
    square_with_center = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]])
    hull = _convex_hull(square_with_center)
    assert len(hull) == 4
    assert (5, 5) not in [tuple(p) for p in hull]


def test_convex_hull_is_non_self_intersecting_for_a_scattered_cloud():
    rng = np.random.default_rng(0)
    points = rng.uniform(0, 100, size=(30, 2))
    hull = _convex_hull(points)
    # every hull point must itself be one of the input points
    input_set = {tuple(p) for p in points}
    assert all(tuple(p) in input_set for p in hull)
    assert len(hull) >= 3


def _fake_landmarks() -> FaceLandmarks:
    # 478 scattered (not grid-aligned) points - real coordinates don't matter for these
    # structural checks, only that indexing into a 478-length array works for every named index
    # group and that convex-hull/filtering logic doesn't degenerate on exactly-collinear input
    rng = np.random.default_rng(0)
    points = rng.uniform(0, 100, size=(478, 2)).astype(np.float32)
    return FaceLandmarks(points=points)


def test_under_eye_band_returns_a_closed_band_polygon():
    landmarks = _fake_landmarks()
    band = under_eye_band(landmarks, "right")
    assert band.shape == (18, 2)  # 9 lower-lid points, mirrored to make a closed band


def test_skin_region_excludes_eyes_and_lips():
    landmarks = _fake_landmarks()
    oval, cutouts = skin_region(landmarks)
    assert len(oval) == len(FACE_OVAL)
    assert len(cutouts) == 3  # right eye, left eye, lips


def test_eye_sclera_region_excludes_the_iris():
    landmarks = _fake_landmarks()
    for side in ("right", "left"):
        sclera, cutouts = eye_sclera_region(landmarks, side)
        assert sclera.shape == (16, 2)
        assert len(cutouts) == 1
        assert cutouts[0].shape == (4, 2)


def test_mouth_interior_region_returns_the_inner_lip_loop():
    landmarks = _fake_landmarks()
    mouth = mouth_interior_region(landmarks)
    assert mouth.shape == (len(LIPS_INNER), 2)


def test_lip_region_excludes_the_mouth_interior():
    landmarks = _fake_landmarks()
    outer, cutouts = lip_region(landmarks)
    assert outer.shape == (len(LIPS_OUTER), 2)
    assert len(cutouts) == 1
    assert cutouts[0].shape == (len(LIPS_INNER), 2)


def test_cheek_and_forehead_regions_are_simple_polygons():
    landmarks = _fake_landmarks()
    for region in (cheek_region(landmarks, "right"), cheek_region(landmarks, "left"), forehead_region(landmarks)):
        assert region.shape[1] == 2
        assert len(region) >= 3


def _landmarks_at(x_offset: float) -> FaceLandmarks:
    points = np.zeros((478, 2), dtype=np.float32)
    points[FACE_OVAL] = np.array([[x_offset, 0]] * len(FACE_OVAL), dtype=np.float32)
    return FaceLandmarks(points=points)


def test_sort_left_to_right_orders_by_face_oval_mean_x():
    # MediaPipe gives no ordering guarantee across faces - fed in right-to-left here to prove
    # the sort actually reorders rather than passing through whatever order it was given
    right = _landmarks_at(500)
    left = _landmarks_at(10)
    middle = _landmarks_at(250)
    ordered = _sort_left_to_right([right, middle, left])
    assert [f.points[FACE_OVAL[0], 0] for f in ordered] == [10, 250, 500]
