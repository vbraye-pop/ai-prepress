import httpx
import numpy as np
import pytest
from PIL import ImageCms

from ai_prepress.io import LoadedImage
from ai_prepress.object_detect import detect_objects

_SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _test_image(height=16, width=16) -> LoadedImage:
    array = np.full((height, width, 3), 40000, dtype=np.uint16)
    return LoadedImage(array=array, icc_profile=_SRGB, bit_depth=16)


def _fake_response(status_code: int, json_body: dict, url: str) -> httpx.Response:
    return httpx.Response(status_code, json=json_body, request=httpx.Request("POST", url))


def test_detect_objects_returns_the_server_reported_detections(monkeypatch):
    detections = [{"label": "a chair", "score": 0.87, "box": [1.0, 2.0, 30.0, 40.0]}]

    def fake_post(url, files, data, timeout):
        return _fake_response(200, {"detections": detections}, url)

    monkeypatch.setattr(httpx, "post", fake_post)

    result = detect_objects(_test_image(), "a chair.", endpoint="https://example.invalid/detect")
    assert result == detections


def test_detect_objects_requires_an_endpoint(monkeypatch):
    monkeypatch.delenv("AI_PREPRESS_OBJECT_DETECT_URL", raising=False)
    with pytest.raises(ValueError):
        detect_objects(_test_image(), "a chair.")


def test_detect_objects_reads_endpoint_from_env(monkeypatch):
    monkeypatch.setenv("AI_PREPRESS_OBJECT_DETECT_URL", "https://example.invalid/detect")

    def fake_post(url, files, data, timeout):
        assert url == "https://example.invalid/detect"
        return _fake_response(200, {"detections": []}, url)

    monkeypatch.setattr(httpx, "post", fake_post)
    assert detect_objects(_test_image(), "a chair.") == []


def test_detect_objects_sends_the_prompt_and_thresholds(monkeypatch):
    def fake_post(url, files, data, timeout):
        assert data["text_prompt"] == "a pitcher. an orange."
        assert data["box_threshold"] == 0.5
        assert data["text_threshold"] == 0.35
        return _fake_response(200, {"detections": []}, url)

    monkeypatch.setattr(httpx, "post", fake_post)
    detect_objects(
        _test_image(),
        "a pitcher. an orange.",
        endpoint="https://example.invalid/detect",
        box_threshold=0.5,
        text_threshold=0.35,
    )
