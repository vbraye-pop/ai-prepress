import io
import json
import zipfile

import httpx
import numpy as np
import pytest
from PIL import Image, ImageCms

from ai_prepress.object_segment import segment_boxes
from ai_prepress.io import LoadedImage

_SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _test_image(height=16, width=16) -> LoadedImage:
    array = np.full((height, width, 3), 40000, dtype=np.uint16)
    return LoadedImage(array=array, icc_profile=_SRGB, bit_depth=16)


def _mask_png_bytes(height: int, width: int, value: int) -> bytes:
    array = np.full((height, width), value, dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def _zip_bytes(height: int, width: int, mask_values: list[int], scores: list[float]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for i, value in enumerate(mask_values):
            zf.writestr(f"mask_{i:02d}.png", _mask_png_bytes(height, width, value))
        zf.writestr("manifest.json", json.dumps({"scores": scores}))
    return buffer.getvalue()


def _fake_zip_response(content: bytes, url: str) -> httpx.Response:
    return httpx.Response(
        200,
        content=content,
        headers={"content-type": "application/zip"},
        request=httpx.Request("POST", url),
    )


def test_segment_boxes_sends_boxes_as_json_and_returns_masks_in_order(monkeypatch):
    captured = {}

    def fake_post(url, files, data, timeout):
        captured["url"] = url
        captured["data"] = data
        return _fake_zip_response(_zip_bytes(8, 8, [255, 0], [0.95, 0.61]), url)

    monkeypatch.setattr(httpx, "post", fake_post)

    boxes = [[10.0, 20.0, 110.0, 220.0], [5.0, 5.0, 50.0, 50.0]]
    masks = segment_boxes(_test_image(), boxes, endpoint="https://example.invalid")

    assert captured["url"] == "https://example.invalid"
    assert json.loads(captured["data"]["boxes"]) == boxes
    assert len(masks) == 2
    assert masks[0].shape == (8, 8)
    assert masks[0].max() == 255
    assert masks[1].max() == 0


def test_segment_boxes_requires_an_endpoint(monkeypatch):
    monkeypatch.delenv("AI_PREPRESS_OBJECT_SEGMENT_URL", raising=False)
    with pytest.raises(ValueError):
        segment_boxes(_test_image(), [[0.0, 0.0, 10.0, 10.0]])


def test_segment_boxes_reads_endpoint_from_env(monkeypatch):
    def fake_post(url, files, data, timeout):
        assert url == "https://from-env.invalid"
        return _fake_zip_response(_zip_bytes(4, 4, [255], [0.9]), url)

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setenv("AI_PREPRESS_OBJECT_SEGMENT_URL", "https://from-env.invalid")

    masks = segment_boxes(_test_image(), [[0.0, 0.0, 4.0, 4.0]])
    assert len(masks) == 1


def test_segment_boxes_raises_on_http_error(monkeypatch):
    def fake_post(url, files, data, timeout):
        return httpx.Response(500, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(httpx.HTTPStatusError):
        segment_boxes(_test_image(), [[0.0, 0.0, 4.0, 4.0]], endpoint="https://example.invalid")
