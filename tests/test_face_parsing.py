import io

import httpx
import numpy as np
import pytest
from PIL import Image, ImageCms

from ai_prepress.face_parsing import LABELS, FaceParsingResult, parse_face
from ai_prepress.io import LoadedImage

_SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _label_png_bytes(height: int, width: int, value: int) -> bytes:
    array = np.full((height, width), value, dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def _test_image(height=32, width=48) -> LoadedImage:
    array = np.full((height, width, 3), 40000, dtype=np.uint16)
    return LoadedImage(array=array, icc_profile=_SRGB, bit_depth=16)


def _fake_response(status_code: int, content: bytes, url: str) -> httpx.Response:
    # a bare httpx.Response has no attached request, and raise_for_status() needs one -
    # matches what a real httpx.post() call actually returns, unlike a constructor call alone
    return httpx.Response(status_code, content=content, request=httpx.Request("POST", url))


def test_parse_face_decodes_the_returned_label_map(monkeypatch):
    skin = LABELS.index("skin")

    def fake_post(url, files, timeout):
        return _fake_response(200, _label_png_bytes(32, 48, skin), url)

    monkeypatch.setattr(httpx, "post", fake_post)

    result = parse_face(_test_image(), endpoint="https://example.invalid/parse")
    assert result.labels.shape == (32, 48)
    assert (result.labels == skin).all()
    assert result.label_names == LABELS


def test_parse_face_requires_an_endpoint(monkeypatch):
    monkeypatch.delenv("AI_PREPRESS_FACE_PARSING_URL", raising=False)
    with pytest.raises(ValueError):
        parse_face(_test_image())


def test_parse_face_reads_endpoint_from_env(monkeypatch):
    monkeypatch.setenv("AI_PREPRESS_FACE_PARSING_URL", "https://example.invalid/parse")

    def fake_post(url, files, timeout):
        assert url == "https://example.invalid/parse"
        return _fake_response(200, _label_png_bytes(4, 4, 0), url)

    monkeypatch.setattr(httpx, "post", fake_post)
    parse_face(_test_image(4, 4))  # no endpoint= passed, must come from the env var


def test_mask_for_combines_multiple_labels():
    labels = np.array(
        [
            [LABELS.index("l_eye"), LABELS.index("r_eye")],
            [LABELS.index("skin"), LABELS.index("hair")],
        ],
        dtype=np.uint8,
    )
    result = FaceParsingResult(labels=labels, label_names=LABELS)
    assert result.mask_for("l_eye", "r_eye").tolist() == [[True, True], [False, False]]
    assert result.mask_for("skin").tolist() == [[False, False], [True, False]]
