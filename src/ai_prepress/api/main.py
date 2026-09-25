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

from ai_prepress import io as core_io
from ai_prepress.checks import acceptance_report
from ai_prepress.features.match_look import match_look
from ai_prepress.metadata import describe

app = FastAPI(title="ai-prepress")

_STORE_DIR = Path(tempfile.gettempdir()) / "ai-prepress-results"
_STORE_DIR.mkdir(exist_ok=True)
_UI_DIR = Path(__file__).resolve().parent.parent.parent.parent / "ui"


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


@app.get("/api/file/{file_id}/preview.png")
def get_preview(file_id: str):
    """Browsers can't render 16-bit TIFF, so this downsamples to 8-bit just for display.
    The stored file itself (/download) keeps full bit depth."""
    loaded = core_io.load(_find_file(file_id))
    as_8bit = core_io.from_unit_float(core_io.to_unit_float(loaded.array), np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(as_8bit[..., :3]).save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")


@app.get("/api/file/{file_id}/download")
def download_file(file_id: str):
    path = _find_file(file_id)
    return FileResponse(path, filename=path.name)


app.mount("/", StaticFiles(directory=_UI_DIR, html=True), name="ui")
