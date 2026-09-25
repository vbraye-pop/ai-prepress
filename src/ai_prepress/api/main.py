"""Local HTTP API wrapping the core package.

Everything runs on one machine for v1, but nothing talks to
ai_prepress.features directly except this app - that's what lets a future
Photoshop plugin or a local-inference mode become just another client of
the same API, instead of a rewrite.
"""

from __future__ import annotations

import io
import tempfile
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from ai_prepress import io as core_io
from ai_prepress.checks import acceptance_report
from ai_prepress.features.match_look import match_look

app = FastAPI(title="ai-prepress")

_RESULTS_DIR = Path(tempfile.gettempdir()) / "ai-prepress-results"
_RESULTS_DIR.mkdir(exist_ok=True)
_UI_DIR = Path(__file__).resolve().parent.parent.parent.parent / "ui"


def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "").suffix or ".png"
    dest = _RESULTS_DIR / f"upload-{uuid.uuid4().hex}{suffix}"
    dest.write_bytes(upload.file.read())
    return dest


@app.post("/api/match-look")
async def api_match_look(
    target: UploadFile = File(...),
    reference: UploadFile = File(...),
    method: str = Form("mkl"),
):
    target_path = _save_upload(target)
    reference_path = _save_upload(reference)

    target_image = core_io.load(target_path)
    reference_image = core_io.load(reference_path)
    result = match_look(target_image, reference_image, method=method)  # type: ignore[arg-type]

    result_id = uuid.uuid4().hex
    result_path = _RESULTS_DIR / f"result-{result_id}.tiff"
    core_io.save(result, result_path)

    report = acceptance_report(target_image, result)

    return JSONResponse(
        {
            "result_id": result_id,
            "delta_e_mean": report.delta_e_mean,
            "delta_e_max": report.delta_e_max,
            "bit_depth_collapsed": report.bit_depth_collapsed,
            "icc_profile_present": report.icc_profile_present,
        }
    )


@app.get("/api/result/{result_id}/preview.png")
def get_preview(result_id: str):
    """Browsers can't render 16-bit TIFF, so this downsamples to 8-bit just for display.
    The actual result file (/download) keeps full bit depth."""
    result_path = _RESULTS_DIR / f"result-{result_id}.tiff"
    loaded = core_io.load(result_path)
    as_8bit = core_io.from_unit_float(core_io.to_unit_float(loaded.array), np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(as_8bit).save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")


@app.get("/api/result/{result_id}/download")
def download_result(result_id: str):
    result_path = _RESULTS_DIR / f"result-{result_id}.tiff"
    return FileResponse(result_path, filename="match-look-result.tiff")


app.mount("/", StaticFiles(directory=_UI_DIR, html=True), name="ui")
