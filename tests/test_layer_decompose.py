import io
import zipfile

import httpx
import numpy as np
import pytest
from PIL import Image, ImageCms

from ai_prepress.layer_decompose import decompose_layers
from ai_prepress.io import LoadedImage

_SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _test_image(height=16, width=16) -> LoadedImage:
    array = np.full((height, width, 3), 40000, dtype=np.uint16)
    return LoadedImage(array=array, icc_profile=_SRGB, bit_depth=16)


def _fake_response(status_code: int, *, content: bytes = b"", json_body: dict | None = None, url: str) -> httpx.Response:
    if json_body is not None:
        return httpx.Response(status_code, json=json_body, request=httpx.Request("GET", url))
    return httpx.Response(
        status_code,
        content=content,
        headers={"content-type": "application/zip"},
        request=httpx.Request("GET", url),
    )


def _png_bytes(height: int, width: int, mode: str, value) -> bytes:
    if mode == "RGB":
        array = np.full((height, width, 3), value, dtype=np.uint8)
    else:
        array = np.full((height, width), value, dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array, mode=mode).save(buffer, format="PNG")
    return buffer.getvalue()


def _zip_bytes(height: int, width: int, layer_count: int) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("background.png", _png_bytes(height, width, "RGB", 128))
        for i in range(layer_count):
            zf.writestr(f"alpha_{i:02d}.png", _png_bytes(height, width, "L", 255 - i))
        zf.writestr("manifest.json", "{}")
    return buffer.getvalue()


def test_decompose_layers_submits_then_polls_until_the_zip_is_ready(monkeypatch):
    calls = {"get": 0}

    def fake_post(url, files, timeout):
        assert url == "https://example.invalid/submit"
        return _fake_response(200, json_body={"call_id": "abc123"}, url=url)

    def fake_get(url, params, timeout):
        assert url == "https://example.invalid/result"
        assert params == {"call_id": "abc123"}
        calls["get"] += 1
        if calls["get"] < 3:
            return _fake_response(200, json_body={"status": "pending"}, url=url)
        return _fake_response(200, content=_zip_bytes(16, 16, 2), url=url)

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)

    result = decompose_layers(_test_image(), endpoint="https://example.invalid", poll_interval=0)

    assert calls["get"] == 3
    assert result.background.shape == (16, 16, 3)
    assert len(result.layer_alphas) == 2
    assert result.layer_alphas[0].shape == (16, 16)


def test_decompose_layers_requires_an_endpoint(monkeypatch):
    monkeypatch.delenv("AI_PREPRESS_LAYER_SEPARATION_URL", raising=False)
    with pytest.raises(ValueError):
        decompose_layers(_test_image())


def test_decompose_layers_raises_on_expiry(monkeypatch):
    def fake_post(url, files, timeout):
        return _fake_response(200, json_body={"call_id": "abc123"}, url=url)

    def fake_get(url, params, timeout):
        return _fake_response(200, json_body={"status": "expired"}, url=url)

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)

    with pytest.raises(RuntimeError, match="expired"):
        decompose_layers(_test_image(), endpoint="https://example.invalid", poll_interval=0)


def test_decompose_layers_times_out(monkeypatch):
    def fake_post(url, files, timeout):
        return _fake_response(200, json_body={"call_id": "abc123"}, url=url)

    def fake_get(url, params, timeout):
        return _fake_response(200, json_body={"status": "pending"}, url=url)

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)

    with pytest.raises(TimeoutError):
        decompose_layers(_test_image(), endpoint="https://example.invalid", timeout=0, poll_interval=0)


def test_decompose_layers_retries_a_transient_poll_failure(monkeypatch):
    # a real deployment hit exactly this: one flaky httpx.ReadTimeout on an otherwise-successful
    # multi-minute call shouldn't abort the whole operation
    calls = {"get": 0}

    def fake_post(url, files, timeout):
        return _fake_response(200, json_body={"call_id": "abc123"}, url=url)

    def fake_get(url, params, timeout):
        calls["get"] += 1
        if calls["get"] == 1:
            raise httpx.ReadTimeout("simulated transient failure")
        return _fake_response(200, content=_zip_bytes(16, 16, 1), url=url)

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)

    result = decompose_layers(_test_image(), endpoint="https://example.invalid", poll_interval=0)
    assert calls["get"] == 2
    assert len(result.layer_alphas) == 1


def test_decompose_layers_still_times_out_if_polling_never_recovers(monkeypatch):
    def fake_post(url, files, timeout):
        return _fake_response(200, json_body={"call_id": "abc123"}, url=url)

    def fake_get(url, params, timeout):
        raise httpx.ReadTimeout("simulated permanent failure")

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)

    with pytest.raises(httpx.ReadTimeout):
        decompose_layers(_test_image(), endpoint="https://example.invalid", timeout=0, poll_interval=0)
