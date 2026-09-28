"""Local HTTP API wrapping the core package.

Everything runs on one machine for v1, but nothing talks to
ai_prepress.features directly except this app - that's what lets a future
Photoshop plugin or a local-inference mode become just another client of
the same API, instead of a rewrite.
"""

from __future__ import annotations

import dataclasses
import io
import tempfile
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from ai_prepress import io as core_io
from ai_prepress.checks import acceptance_report
from ai_prepress.face_landmarks import (
    FACE_OVAL,
    FaceLandmarks,
    cheek_region,
    detect_landmarks,
    forehead_region,
    skin_region,
    under_eye_band,
)
from ai_prepress.features.layer_separation import separate_layers
from ai_prepress.features.match_look import match_look
from ai_prepress.features.retouch_faces import RetouchStrengths, retouch_faces
from ai_prepress.metadata import describe

_MAX_FACES = 32  # matches Capture One's own documented per-image face-detection ceiling

app = FastAPI(title="ai-prepress")

_STORE_DIR = Path(tempfile.gettempdir()) / "ai-prepress-results"
_STORE_DIR.mkdir(exist_ok=True)
_UI_DIR = Path(__file__).resolve().parent.parent.parent.parent / "ui"

# plenty for anything this UI displays a preview at - no point encoding,
# transferring, and decoding a full-resolution image for a ~300px thumbnail
_PREVIEW_MAX_EDGE = 1024


def _preview_stride(height: int, width: int, max_edge: int = _PREVIEW_MAX_EDGE) -> int:
    longest = max(height, width)
    return -(-longest // max_edge) if longest > max_edge else 1  # ceil division


def _downsample_for_preview(array: np.ndarray, max_edge: int = _PREVIEW_MAX_EDGE) -> np.ndarray:
    stride = _preview_stride(array.shape[0], array.shape[1], max_edge)
    return array if stride == 1 else array[::stride, ::stride]


def _store_bytes(data: bytes, suffix: str) -> str:
    file_id = uuid.uuid4().hex
    (_STORE_DIR / f"{file_id}{suffix}").write_bytes(data)
    return file_id


def _store_upload(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "").suffix or ".png"
    return _store_bytes(upload.file.read(), suffix)


def _find_file(file_id: str) -> Path:
    matches = list(_STORE_DIR.glob(f"{file_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="no such file")
    return matches[0]


# Detected landmarks are cached here so an "apply retouch" call can reuse the exact detection an
# earlier "/api/face-regions" call already ran, rather than re-running MediaPipe (cost) or
# risking a different face count/order on a second detection pass (correctness - face index 0 in
# a per-face strengths payload must mean the same face at apply time it meant at preview time).
# A SIBLING directory, not a same-directory "{file_id}.npy" file: _find_file's glob matches any
# suffix after the first "." and returns matches[0] from unsorted glob order, so a landmark cache
# file living next to the real upload would intermittently get returned as if it were the image.
_LANDMARK_DIR = _STORE_DIR / "landmarks"
_LANDMARK_DIR.mkdir(exist_ok=True)


def _store_landmarks(file_id: str, faces: list[FaceLandmarks]) -> None:
    stacked = np.stack([f.points for f in faces]) if faces else np.empty((0, 478, 2), dtype=np.float32)
    np.save(_LANDMARK_DIR / f"{file_id}.npy", stacked)


def _load_landmarks(file_id: str) -> list[FaceLandmarks]:
    path = _LANDMARK_DIR / f"{file_id}.npy"
    if not path.exists():
        raise HTTPException(status_code=404, detail="no cached face detection for this file_id - call /api/face-regions first")
    return [FaceLandmarks(points=pts) for pts in np.load(path)]


class RetouchStrengthsPayload(BaseModel):
    dark_circles: float = 0.0
    even_skin: float = 0.0
    even_skin_texture: float = 0.0
    contouring: float = 0.0
    feather_amount: float = 1.0
    edge_amount: float = 0.0
    eye_whiten: float = 0.0
    teeth_whiten: float = 0.0
    lip_enhance: float = 0.0

    def to_strengths(self) -> RetouchStrengths:
        return RetouchStrengths(**self.model_dump())


class RetouchApplyRequest(BaseModel):
    file_id: str
    mode: str  # "all_faces" | "per_face"
    strengths: RetouchStrengthsPayload = RetouchStrengthsPayload()
    per_face: dict[int, RetouchStrengthsPayload] = {}


@app.post("/api/inspect")
async def api_inspect(image: UploadFile = File(...)):
    file_id = _store_upload(image)
    info = describe(_find_file(file_id))
    return JSONResponse({"file_id": file_id, **dataclasses.asdict(info)})


@app.post("/api/match-look")
async def api_match_look(
    target: UploadFile = File(...),
    reference: UploadFile = File(...),
    method: str = Form("mkl"),
):
    target_id = _store_upload(target)
    reference_id = _store_upload(reference)

    target_image = core_io.load(_find_file(target_id))
    reference_image = core_io.load(_find_file(reference_id))
    result = match_look(target_image, reference_image, method=method)  # type: ignore[arg-type]

    result_id = _store_bytes(b"", ".tiff")
    core_io.save(result, _find_file(result_id))

    report = acceptance_report(target_image, result)

    return JSONResponse(
        {
            "target_id": target_id,
            "reference_id": reference_id,
            "result_id": result_id,
            "delta_e_mean": report.delta_e_mean,
            "delta_e_max": report.delta_e_max,
            "bit_depth_collapsed": report.bit_depth_collapsed,
            "icc_profile_present": report.icc_profile_present,
        }
    )


@app.post("/api/face-regions")
async def api_face_regions(image: UploadFile = File(...)):
    """Retouching regions derived from MediaPipe's 478-point face mesh - see
    ai_prepress.face_landmarks for why this replaced the old CelebAMask-HQ semantic-parsing
    endpoint: that scheme's 19 classes have no under-eye, cheek, or forehead class, which is
    what Retouch Faces' Dark Circles/Contouring sub-tools actually need a mask for.

    Detects every face in the frame (Capture One detects up to 32 per image; matched here) and
    caches the detection (_store_landmarks) so a later /api/retouch-faces/apply call reuses the
    same faces in the same order rather than re-detecting. Coordinates are returned already
    scaled to match /preview.jpg's downsampling (same stride, same source image), so the browser
    can draw regions straight onto the preview it already has with no extra reconciliation."""
    file_id = _store_upload(image)
    loaded = core_io.load(_find_file(file_id))

    unit = core_io.to_unit_float(loaded.array)[..., :3]
    rgb_8bit = (np.clip(unit, 0.0, 1.0) * 255 + 0.5).astype(np.uint8)

    faces = detect_landmarks(rgb_8bit, max_faces=_MAX_FACES)
    _store_landmarks(file_id, faces)
    if not faces:
        return JSONResponse({"file_id": file_id, "face_detected": False, "face_count": 0, "faces": []})

    height, width = rgb_8bit.shape[:2]
    stride = _preview_stride(height, width)

    def scaled(points) -> list[list[float]]:
        return (np.asarray(points) / stride).round(1).tolist()

    faces_payload = []
    for index, landmarks in enumerate(faces):
        oval, cutouts = skin_region(landmarks)
        face_oval_points = landmarks.subset(FACE_OVAL)
        x0, y0 = face_oval_points.min(axis=0)
        x1, y1 = face_oval_points.max(axis=0)
        faces_payload.append(
            {
                "face_index": index,
                "bbox": scaled([[x0, y0], [x1, y1]]),
                "regions": {
                    "skin_oval": scaled(oval),
                    "skin_cutouts": [scaled(c) for c in cutouts],
                    "forehead": scaled(forehead_region(landmarks)),
                    "cheek_right": scaled(cheek_region(landmarks, "right")),
                    "cheek_left": scaled(cheek_region(landmarks, "left")),
                    "under_eye_right": scaled(under_eye_band(landmarks, "right")),
                    "under_eye_left": scaled(under_eye_band(landmarks, "left")),
                },
            }
        )

    return JSONResponse(
        {"file_id": file_id, "face_detected": True, "face_count": len(faces), "faces": faces_payload}
    )


@app.post("/api/retouch-faces/apply")
async def api_retouch_faces_apply(request: RetouchApplyRequest):
    """Low-frequency masked edits on top of the face-region masks - see
    ai_prepress.features.retouch_faces for the frequency-separation approach. Takes a file_id
    from an earlier /api/face-regions call (not a fresh upload) so it can reuse that call's exact
    detection via _load_landmarks - mode "all_faces" broadcasts `strengths` to every detected
    face, "per_face" edits only the faces present in `per_face`."""
    loaded = core_io.load(_find_file(request.file_id))
    faces = _load_landmarks(request.file_id)

    if request.mode == "per_face":
        strengths: RetouchStrengths | dict[int, RetouchStrengths] = {
            index: payload.to_strengths() for index, payload in request.per_face.items()
        }
    else:
        strengths = request.strengths.to_strengths()

    try:
        result = retouch_faces(loaded, strengths, landmarks=faces)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if result is None:
        return JSONResponse({"file_id": request.file_id, "face_detected": False})

    result_id = _store_bytes(b"", ".tiff")
    core_io.save(result, _find_file(result_id))

    report = acceptance_report(loaded, result)

    return JSONResponse(
        {
            "file_id": request.file_id,
            "result_id": result_id,
            "face_detected": True,
            "face_count": len(faces),
            "delta_e_mean": report.delta_e_mean,
            "delta_e_max": report.delta_e_max,
            "bit_depth_collapsed": report.bit_depth_collapsed,
            "icc_profile_present": report.icc_profile_present,
        }
    )


@app.post("/api/layer-separation")
async def api_layer_separation(image: UploadFile = File(...)):
    """Photo -> per-object RGBA layers + a reconstructed background plate. Single upload -> one
    call -> JSON, same shape as /api/face-regions and /api/match-look - unlike Retouch Faces,
    there's no cheaper "detect only" phase to cache separately, since the decomposition itself is
    both the expensive step and the final result. See features.layer_separation for the remote
    call and bit-depth-preservation logic; this endpoint can block for minutes on a cold Modal
    start (see ai_prepress.layer_decompose's module docstring for why that's still a plain
    synchronous call rather than a job-polling API at this layer)."""
    file_id = _store_upload(image)
    loaded = core_io.load(_find_file(file_id))

    try:
        result = separate_layers(loaded)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc

    background_id = _store_bytes(b"", ".tiff")
    core_io.save(result.background, _find_file(background_id))

    layers_payload = []
    for index, layer in enumerate(result.layers):
        layer_id = _store_bytes(b"", ".tiff")
        core_io.save(layer.image, _find_file(layer_id))
        layers_payload.append(
            {
                "layer_index": index,
                "result_id": layer_id,
                "bbox": layer.bbox,
                "alpha_coverage": float((layer.image.array[..., 3] > 0).mean()),
            }
        )

    return JSONResponse(
        {
            "file_id": file_id,
            "background_id": background_id,
            "layer_count": len(result.layers),
            "layers": layers_payload,
        }
    )


@app.get("/api/file/{file_id}/preview.jpg")
def get_preview(file_id: str):
    """Browsers can't render 16-bit TIFF, so this downsamples to 8-bit just for display.
    The stored file itself (/download) keeps full bit depth and resolution - this is a
    thumbnail, not a deliverable, so it's downsized and JPEG-compressed for speed."""
    loaded = core_io.load(_find_file(file_id))
    small = _downsample_for_preview(loaded.array)
    as_8bit = core_io.from_unit_float(core_io.to_unit_float(small), np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(as_8bit[..., :3]).save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/jpeg")


@app.get("/api/file/{file_id}/preview.png")
def get_preview_png(file_id: str):
    """Same downsampling as preview.jpg, but PNG with alpha kept intact - JPEG can't carry an
    alpha channel at all, so pointing the .jpg route at a layer wouldn't error, it would silently
    render a solid rectangle with the alpha discarded. Used for layer-separation thumbnails and
    the background plate; every other tab keeps using .jpg, unchanged."""
    loaded = core_io.load(_find_file(file_id))
    small = _downsample_for_preview(loaded.array)
    as_8bit = core_io.from_unit_float(core_io.to_unit_float(small), np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(as_8bit).save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")


@app.get("/api/file/{file_id}/download")
def download_file(file_id: str):
    path = _find_file(file_id)
    return FileResponse(path, filename=path.name)


app.mount("/", StaticFiles(directory=_UI_DIR, html=True), name="ui")
