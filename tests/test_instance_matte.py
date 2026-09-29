import io
import zipfile

import httpx
import numpy as np
import pytest
from PIL import Image

from ai_prepress.instance_matte import matte_crops


def _rgb_crop(height=8, width=8) -> np.ndarray:
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[2:6, 2:6] = [200, 100, 50]
    return array


def _alpha_png_bytes(height: int, width: int, value: int) -> bytes:
    array = np.full((height, width), value, dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def _zip_bytes(height: int, width: int, alpha_values: list[int]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for i, value in enumerate(alpha_values):
            zf.writestr(f"alpha_{i:02d}.png", _alpha_png_bytes(height, width, value))
        zf.writestr("manifest.json", '{"count": %d}' % len(alpha_values))
    return buffer.getvalue()


def _fake_zip_response(content: bytes, url: str) -> httpx.Response:
    return httpx.Response(
        200,
        content=content,
        headers={"content-type": "application/zip"},
        request=httpx.Request("POST", url),
    )


def test_matte_crops_returns_alphas_in_order(monkeypatch):
    captured = {}

    def fake_post(url, files, timeout):
        captured["url"] = url
        captured["files"] = files
        return _fake_zip_response(_zip_bytes(8, 8, [255, 0]), url)

    monkeypatch.setattr(httpx, "post", fake_post)

    alphas = matte_crops([_rgb_crop(), _rgb_crop()], endpoint="https://example.invalid")

    assert captured["url"] == "https://example.invalid"
    assert len(captured["files"]) == 2
    assert len(alphas) == 2
    assert alphas[0].shape == (8, 8)
    assert alphas[0].max() == 255
    assert alphas[1].max() == 0


def test_matte_crops_returns_empty_list_for_no_crops(monkeypatch):
    # must not raise ValueError for a missing endpoint when there's nothing to matte anyway
    monkeypatch.delenv("AI_PREPRESS_INSTANCE_MATTE_URL", raising=False)
    assert matte_crops([]) == []


def test_matte_crops_requires_an_endpoint(monkeypatch):
    monkeypatch.delenv("AI_PREPRESS_INSTANCE_MATTE_URL", raising=False)
    with pytest.raises(ValueError):
        matte_crops([_rgb_crop()])


def test_matte_crops_reads_endpoint_from_env(monkeypatch):
    def fake_post(url, files, timeout):
        assert url == "https://from-env.invalid"
        return _fake_zip_response(_zip_bytes(4, 4, [128]), url)

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setenv("AI_PREPRESS_INSTANCE_MATTE_URL", "https://from-env.invalid")

    alphas = matte_crops([_rgb_crop(4, 4)])
    assert len(alphas) == 1


def test_matte_crops_raises_on_http_error(monkeypatch):
    def fake_post(url, files, timeout):
        return httpx.Response(500, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(httpx.HTTPStatusError):
        matte_crops([_rgb_crop()], endpoint="https://example.invalid")


def test_matte_crops_sends_crops_as_pngs_under_the_files_field(monkeypatch):
    captured = {}

    def fake_post(url, files, timeout):
        captured["files"] = files
        return _fake_zip_response(_zip_bytes(8, 8, [255]), url)

    monkeypatch.setattr(httpx, "post", fake_post)
    matte_crops([_rgb_crop()], endpoint="https://example.invalid")

    field_name, (filename, content, content_type) = captured["files"][0]
    assert field_name == "files"
    assert content_type == "image/png"
    decoded = Image.open(io.BytesIO(content))
    assert decoded.mode == "RGB"
    assert decoded.size == (8, 8)
