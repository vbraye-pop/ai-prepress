"""Face landmark detection and retouching-region polygons.

Supersedes ai_prepress.face_parsing (CelebAMask-HQ semantic segmentation, 19 fixed classes)
as the region foundation for Retouch Faces. The classes that scheme offers - skin, eyes,
brows, nose, lips, ears, hair - don't include an under-eye band, a cheek/contour zone, or a
forehead, because CelebAMask-HQ's own annotators never labeled those as a class. No amount of
model quality fixes a class that doesn't exist in the training taxonomy. A dense landmark mesh
sidesteps this entirely: regions are geometry derived from point positions, not a fixed lookup
table. See the "face-parsing-vs-landmarks-pivot" project note for the fuller reasoning.

Also Apache 2.0 (MediaPipe's own license) rather than CelebAMask-HQ-trained, which carries a
non-commercial research license regardless of the model code's own license - see
face_parsing.py's module docstring for that history.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
_MODEL_PATH = Path.home() / ".cache" / "ai-prepress" / "face_landmarker.task"
_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


def _model_path() -> Path:
    if not _MODEL_PATH.exists():
        response = httpx.get(_MODEL_URL, timeout=30.0, follow_redirects=True)
        response.raise_for_status()
        _MODEL_PATH.write_bytes(response.content)
    return _MODEL_PATH


@dataclass
class FaceLandmarks:
    points: np.ndarray  # (478, 2) float32, absolute pixel coordinates, image space

    def subset(self, indices: list[int]) -> np.ndarray:
        return self.points[indices]


def _sort_left_to_right(faces: list[FaceLandmarks]) -> list[FaceLandmarks]:
    """MediaPipe gives no documented ordering guarantee across faces (internal detection/
    confidence order, not spatial) - sorting once here means face index 0 always means
    "leftmost face in the photo," deterministically, rather than trusting raw model order for
    something a UI needs to label stably ("Face 1" / "Face 2...") across repeated calls."""
    return sorted(faces, key=lambda f: f.subset(FACE_OVAL)[:, 0].mean())


def detect_landmarks(rgb: np.ndarray, max_faces: int = 1) -> list[FaceLandmarks]:
    """Runs locally - MediaPipe's own CPU path is fast enough that this doesn't need a Modal
    endpoint the way the SegFormer semantic parser did. Has its own built-in face detector, so
    no separate crop step is needed first (unlike face_parsing.parse_portrait). Returns a list,
    sorted left-to-right (see _sort_left_to_right), empty if no face is found - a photo without
    one is a real case callers should handle, not something to special-case with None."""
    import mediapipe as mp
    from mediapipe.tasks.python import core, vision

    base_options = core.base_options.BaseOptions(model_asset_path=str(_model_path()))
    landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(base_options=base_options, num_faces=max_faces)
    )
    height, width = rgb.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return []

    faces = [
        FaceLandmarks(points=np.array([[p.x * width, p.y * height] for p in lm], dtype=np.float32))
        for lm in result.face_landmarks
    ]
    return _sort_left_to_right(faces)


# ---------------------------------------------------------------------------------------------
# Real landmark index groups, sourced from mediapipe/python/solutions/face_mesh_connections.py
# (google-ai-edge/mediapipe, Apache 2.0) as of 2026-09-28, not guessed or recalled from memory.
# That file gives unordered *edges* (e.g. FACEMESH_FACE_OVAL), not a walk order a polygon fill
# can use directly - _walk_loop turns a closed-loop edge set into an ordered point sequence.
# ---------------------------------------------------------------------------------------------

_FACE_OVAL_EDGES = [
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389), (389, 356),
    (356, 454), (454, 323), (323, 361), (361, 288), (288, 397), (397, 365), (365, 379),
    (379, 378), (378, 400), (400, 377), (377, 152), (152, 148), (148, 176), (176, 149),
    (149, 150), (150, 136), (136, 172), (172, 58), (58, 132), (132, 93), (93, 234),
    (234, 127), (127, 162), (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
]
_RIGHT_EYE_LOWER_EDGES = [
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133),
]
_LEFT_EYE_LOWER_EDGES = [
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381), (381, 382),
    (382, 362),
]
_RIGHT_EYE_LOOP_EDGES = _RIGHT_EYE_LOWER_EDGES + [
    (33, 246), (246, 161), (161, 160), (160, 159), (159, 158), (158, 157), (157, 173),
    (173, 133),
]
_LEFT_EYE_LOOP_EDGES = _LEFT_EYE_LOWER_EDGES + [
    (263, 466), (466, 388), (388, 387), (387, 386), (386, 385), (385, 384), (384, 398),
    (398, 362),
]
_LIPS_OUTER_EDGES = [
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321),
    (321, 375), (375, 291), (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267),
    (267, 269), (269, 270), (270, 409), (409, 291),
]
_RIGHT_EYEBROW_NODES = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
_LEFT_EYEBROW_NODES = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]

# not a mediapipe-named group - "cheek surface" point cloud sourced from a cheat-sheet of these
# landmark indices (hackernoon.com/mediapipe-face-mesh-landmark-indices-cheat-sheet),
# cross-checked against the face oval list above (matched exactly) before trusting its other
# groupings. Not an ordered boundary loop, just points scattered across the region - _convex_hull
# below turns them into a fillable simple polygon; cheek_region additionally filters out any
# that reach above the eye (see its docstring).
_RIGHT_CHEEK_POINTS = [36, 205, 206, 207, 187, 123, 116, 117, 118, 119, 100, 47]
_LEFT_CHEEK_POINTS = [266, 425, 426, 427, 411, 352, 345, 346, 347, 348, 329, 277]
GLABELLA = [9, 151]  # between the brows - forehead's inner/lower boundary


def _walk_loop(edges: list[tuple[int, int]]) -> list[int]:
    """Orders a closed loop's edges into a point sequence, starting from the first edge's first
    node. Only correct for a true simple cycle (every node has exactly 2 neighbors) - the edge
    lists above that feed this are all single closed loops, not general graphs."""
    adjacency: dict[int, list[int]] = {}
    for a, b in edges:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    start = edges[0][0]
    loop = [start]
    visited_edges = set()
    current = start
    while True:
        next_node = next(
            (n for n in adjacency[current] if tuple(sorted((current, n))) not in visited_edges),
            None,
        )
        if next_node is None or next_node == start:
            break
        visited_edges.add(tuple(sorted((current, next_node))))
        loop.append(next_node)
        current = next_node
    return loop


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain algorithm - pure numpy, no scipy (this project deliberately
    doesn't depend on it, see colour-science's own optional-scipy warning at import). Needed
    because these landmark index groups are scattered points across a region, not an ordered
    boundary loop the way FACE_OVAL/eyes/lips are - an angle-around-centroid sort was tried
    first and produced self-intersecting star shapes, because several of these points aren't
    actually star-convex from their own centroid. A true hull can't self-intersect."""
    pts = sorted(map(tuple, points))
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


FACE_OVAL = _walk_loop(_FACE_OVAL_EDGES)
RIGHT_EYE_LOOP = _walk_loop(_RIGHT_EYE_LOOP_EDGES)
LEFT_EYE_LOOP = _walk_loop(_LEFT_EYE_LOOP_EDGES)
LIPS_OUTER = _walk_loop(_LIPS_OUTER_EDGES)
RIGHT_EYE_LOWER = _walk_loop(_RIGHT_EYE_LOWER_EDGES)
LEFT_EYE_LOWER = _walk_loop(_LEFT_EYE_LOWER_EDGES)


def under_eye_band(landmarks: FaceLandmarks, side: str, depth_fraction: float = 0.45) -> np.ndarray:
    """A polygon tracing each eye's lower lid, offset downward by a fraction of that eye's own
    width. Dark circles sit in this band. depth_fraction is an empirical starting point (not
    from a cited anatomical source, unlike the landmark indices above) - flagged here rather
    than presented as more rigorous than it is; tune against real photos before this feeds an
    actual Dark Circles edit."""
    lower_idx = RIGHT_EYE_LOWER if side == "right" else LEFT_EYE_LOWER
    upper = landmarks.subset(lower_idx)
    width = np.linalg.norm(upper[0] - upper[-1])
    offset = np.array([0.0, width * depth_fraction])
    lower = upper + offset
    return np.vstack([upper, lower[::-1]])


def cheek_region(landmarks: FaceLandmarks, side: str) -> np.ndarray:
    """The cheat-sheet's cheek point cloud (see the module-level citation) reaches up as far as
    the eye corner, which produces a hull that crowds into/overlaps the eye once filled - a
    convex hull is only ever as good as its input points, and these span too far vertically for
    "cheek" specifically. Filtered here to points strictly below that eye's own lower lid,
    computed from this photo's actual geometry rather than a fixed pixel margin, so it holds up
    across different face sizes/distances."""
    if side == "right":
        points_idx, eye_lower_idx = _RIGHT_CHEEK_POINTS, RIGHT_EYE_LOWER
    else:
        points_idx, eye_lower_idx = _LEFT_CHEEK_POINTS, LEFT_EYE_LOWER

    points = landmarks.subset(points_idx)
    eye_pts = landmarks.subset(eye_lower_idx)
    eye_width = np.linalg.norm(eye_pts[0] - eye_pts[-1])
    # first attempt (a plain y > eye bottom filter) left the topmost cheek point about 1px
    # below the eye - technically "below" but touching it once rendered. A real margin fixes it.
    margin = eye_width * 0.2
    below_eye = points[points[:, 1] > eye_pts[:, 1].max() + margin]
    return _convex_hull(below_eye if len(below_eye) >= 3 else points)


def forehead_region(landmarks: FaceLandmarks) -> np.ndarray:
    """Hull of the face oval's own points that sit above the eyebrows, plus the eyebrows
    themselves - the eyebrow line is a real anatomical boundary (unlike forehead's other three
    sides, which fade into hairline/temple with no sharp edge), so it's used directly as the
    lower bound. An earlier version took FACE_OVAL[:8]/[-8:] assuming that gave "the top of the
    oval" - it doesn't, the 36 oval points aren't evenly spaced by angle, and that slice reached
    down past the eyebrows on each side, producing a hull that visibly crossed through them.
    Filtering by actual y-position (above the eyebrows' own lowest point) is correct regardless
    of point spacing."""
    brow_idx = _RIGHT_EYEBROW_NODES + _LEFT_EYEBROW_NODES
    brow_pts = landmarks.subset(brow_idx)
    oval_pts = landmarks.subset(FACE_OVAL)
    above_brows = oval_pts[oval_pts[:, 1] < brow_pts[:, 1].min()]
    combined = np.vstack([above_brows, brow_pts, landmarks.subset(GLABELLA)])
    return _convex_hull(combined)


def skin_region(landmarks: FaceLandmarks) -> tuple[np.ndarray, list[np.ndarray]]:
    """The face oval polygon, plus the cutout polygons (eyes, lips) that should be excluded
    from any skin-toned edit - the shape an LF/HF skin operation (Even Skin, or Blemish's
    inpaint-exclusion mask) should be confined to. Caller subtracts the cutouts from the oval
    when rasterizing (this module returns polygons, not a rasterized mask, so callers can
    render at whatever resolution they need)."""
    oval = landmarks.subset(FACE_OVAL)
    cutouts = [
        landmarks.subset(RIGHT_EYE_LOOP),
        landmarks.subset(LEFT_EYE_LOOP),
        landmarks.subset(LIPS_OUTER),
    ]
    return oval, cutouts
