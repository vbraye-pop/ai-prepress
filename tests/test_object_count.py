import io

import httpx
import numpy as np
import pytest
from PIL import Image, ImageCms

from ai_prepress.io import LoadedImage
from ai_prepress.object_count import count_objects

_SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _test_image(height=16, width=16) -> LoadedImage:
    array = np.full((height, width, 3), 40000, dtype=np.uint16)
    return LoadedImage(array=array, icc_profile=_SRGB, bit_depth=16)


def _fake_response(status_code: int, json_body: dict, url: str) -> httpx.Response:
    return httpx.Response(status_code, json=json_body, request=httpx.Request("POST", url))


def test_count_objects_returns_the_server_reported_count(monkeypatch):
    def fake_post(url, files, timeout):
        return _fake_response(200, {"object_count": 5}, url)

    monkeypatch.setattr(httpx, "post", fake_post)

    result = count_objects(_test_image(), endpoint="https://example.invalid/count")
    assert result == 5


def test_count_objects_requires_an_endpoint(monkeypatch):
    monkeypatch.delenv("AI_PREPRESS_OBJECT_COUNT_URL", raising=False)
    with pytest.raises(ValueError):
        count_objects(_test_image())


def test_count_objects_reads_endpoint_from_env(monkeypatch):
    monkeypatch.setenv("AI_PREPRESS_OBJECT_COUNT_URL", "https://example.invalid/count")

    def fake_post(url, files, timeout):
        assert url == "https://example.invalid/count"
        return _fake_response(200, {"object_count": 2}, url)

    monkeypatch.setattr(httpx, "post", fake_post)
    assert count_objects(_test_image()) == 2
