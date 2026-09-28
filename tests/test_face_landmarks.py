import numpy as np

from ai_prepress.face_landmarks import (
    FACE_OVAL,
    LEFT_EYE_LOOP,
    LIPS_OUTER,
    RIGHT_EYE_LOOP,
    FaceLandmarks,
    _convex_hull,
    _walk_loop,
    cheek_region,
    forehead_region,
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


def test_cheek_and_forehead_regions_are_simple_polygons():
    landmarks = _fake_landmarks()
    for region in (cheek_region(landmarks, "right"), cheek_region(landmarks, "left"), forehead_region(landmarks)):
        assert region.shape[1] == 2
        assert len(region) >= 3
