import numpy as np
import httpx
import pytest

from ai_prepress.io import LoadedImage
from ai_prepress.layer_naming import name_layers


def _rgba_layer(height=8, width=8) -> LoadedImage:
    array = np.zeros((height, width, 4), dtype=np.uint8)
    array[2:6, 2:6] = [200, 100, 50, 255]  # an opaque square, rest fully transparent
    return LoadedImage(array=array, icc_profile=None, bit_depth=8)


def _fake_response(status_code: int, json_body: dict, url: str) -> httpx.Response:
    return httpx.Response(status_code, json=json_body, request=httpx.Request("POST", url))


def test_name_layers_returns_labels_in_order(monkeypatch):
    def fake_post(url, files, timeout):
        assert url == "https://example.invalid/name"
        assert len(files) == 2
        return _fake_response(200, {"labels": ["Mug", "Chair"]}, url)

    monkeypatch.setattr(httpx, "post", fake_post)

    result = name_layers([_rgba_layer(), _rgba_layer()], endpoint="https://example.invalid/name")
    assert result == ["Mug", "Chair"]


def test_name_layers_returns_empty_list_for_no_layers(monkeypatch):
    # must not raise ValueError for a missing endpoint when there's nothing to name anyway
    monkeypatch.delenv("AI_PREPRESS_LAYER_NAMING_URL", raising=False)
    assert name_layers([]) == []


def test_name_layers_requires_an_endpoint(monkeypatch):
    monkeypatch.delenv("AI_PREPRESS_LAYER_NAMING_URL", raising=False)
    with pytest.raises(ValueError):
        name_layers([_rgba_layer()])


def test_name_layers_flattens_rgba_onto_white_before_sending(monkeypatch):
    captured = {}

    def fake_post(url, files, timeout):
        captured["files"] = files
        return _fake_response(200, {"labels": ["Thing"]}, url)

    monkeypatch.setattr(httpx, "post", fake_post)
    name_layers([_rgba_layer()], endpoint="https://example.invalid/name")

    from PIL import Image
    import io

    field_name, (filename, content, content_type) = captured["files"][0]
    assert field_name == "files"
    assert content_type == "image/png"
    decoded = Image.open(io.BytesIO(content))
    assert decoded.mode == "RGB"  # flattened, no alpha channel sent over the wire
    # the fully-transparent corner should now read as white, not black or leftover RGB noise
    assert decoded.getpixel((0, 0)) == (255, 255, 255)
