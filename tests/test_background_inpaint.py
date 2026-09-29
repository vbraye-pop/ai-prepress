import io

import httpx
import numpy as np
import pytest
from PIL import Image

from ai_prepress.background_inpaint import inpaint_background
from deploy.background_inpaint import (
    compute_inpaint_scale,
    composite_inpaint_result,
    crop_to_source_size,
    dilate_mask,
    is_hard_binary_mask,
)


def _fake_response(status_code: int, content: bytes, url: str) -> httpx.Response:
    return httpx.Response(status_code, content=content, request=httpx.Request("POST", url))


def _png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


# ---- client (ai_prepress.background_inpaint) ----


def test_inpaint_background_decodes_the_returned_image(monkeypatch):
    returned = np.full((10, 12, 3), 200, dtype=np.uint8)

    def fake_post(url, files, timeout):
        return _fake_response(200, _png_bytes(returned), url)

    monkeypatch.setattr(httpx, "post", fake_post)

    source = np.full((10, 12, 3), 50, dtype=np.uint8)
    mask = np.zeros((10, 12), dtype=np.uint8)
    result = inpaint_background(source, mask, endpoint="https://example.invalid/inpaint")

    assert result.shape == (10, 12, 3)
    assert (result == 200).all()


def test_inpaint_background_sends_both_source_and_mask_as_multipart_files(monkeypatch):
    seen = {}

    def fake_post(url, files, timeout):
        seen["files"] = files
        returned = np.zeros((4, 4, 3), dtype=np.uint8)
        return _fake_response(200, _png_bytes(returned), url)

    monkeypatch.setattr(httpx, "post", fake_post)

    source = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    inpaint_background(source, mask, endpoint="https://example.invalid/inpaint")

    assert "file" in seen["files"]
    assert "mask" in seen["files"]


def test_inpaint_background_requires_an_endpoint(monkeypatch):
    monkeypatch.delenv("AI_PREPRESS_BACKGROUND_INPAINT_URL", raising=False)
    source = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        inpaint_background(source, mask)


def test_inpaint_background_reads_endpoint_from_env(monkeypatch):
    monkeypatch.setenv("AI_PREPRESS_BACKGROUND_INPAINT_URL", "https://example.invalid/inpaint")

    def fake_post(url, files, timeout):
        assert url == "https://example.invalid/inpaint"
        returned = np.zeros((4, 4, 3), dtype=np.uint8)
        return _fake_response(200, _png_bytes(returned), url)

    monkeypatch.setattr(httpx, "post", fake_post)
    source = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    inpaint_background(source, mask)  # no endpoint= passed, must come from the env var


# ---- server-side pure logic (deploy.background_inpaint) ----


def test_is_hard_binary_mask_accepts_pure_0_and_255():
    mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    assert is_hard_binary_mask(mask) is True


def test_is_hard_binary_mask_rejects_any_intermediate_value():
    # the exact failure mode bug #2 exists to catch: a feathered/anti-aliased edge value
    mask = np.array([[0, 127], [255, 0]], dtype=np.uint8)
    assert is_hard_binary_mask(mask) is False


def test_compute_inpaint_scale_is_identity_below_the_cap():
    assert compute_inpaint_scale(height=800, width=600, max_edge=2048) == 1.0


def test_compute_inpaint_scale_downscales_the_longer_edge_to_the_cap():
    scale = compute_inpaint_scale(height=4000, width=2000, max_edge=2000)
    assert scale == pytest.approx(0.5)


def test_crop_to_source_size_undoes_pad_out_to_modulo_8():
    # reproduces the exact bug: simple-lama-inpainting's own prepare_img_and_mask pads to a
    # multiple of 8 (symmetric padding, extra rows/cols on the bottom/right) and its __call__
    # never crops that back off - a non-multiple-of-8 source is the regression case that catches it
    width, height = 13, 11
    original = np.arange(width * height * 3, dtype=np.uint8).reshape(height, width, 3)
    padded = np.pad(original, ((0, 5), (0, 3), (0, 0)), mode="symmetric")  # 11->16, 13->16
    assert padded.shape == (16, 16, 3)

    cropped = crop_to_source_size(Image.fromarray(padded), width, height)

    assert cropped.size == (width, height)
    assert np.array_equal(np.array(cropped), original)


def test_composite_inpaint_result_is_fully_synthesized_everywhere_inside_the_hole():
    # regression test for a real bug: an earlier version of this function ramped alpha INWARD
    # from the mask edge, which blends source_rgb (the removed object's own pixels, since it's
    # what's inside the hole) back into the result - a visible ghost ring of the removed object.
    # Every pixel where mask > 0 must be fully LaMa's content, no exception.
    source = np.full((20, 20, 3), 10, dtype=np.uint8)
    inpainted = np.full((20, 20, 3), 200, dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255

    composite = composite_inpaint_result(source, mask, inpainted, feather_px=4)

    inside = mask > 0
    assert (composite[inside] == 200).all()


def test_composite_inpaint_result_keeps_original_pixels_well_outside_the_hole():
    source = np.full((20, 20, 3), 10, dtype=np.uint8)
    inpainted = np.full((20, 20, 3), 200, dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255

    composite = composite_inpaint_result(source, mask, inpainted, feather_px=4)

    # row 0 is 5px from the hole, past the 4px feather - must be untouched source
    assert (composite[0, :] == source[0, :]).all()


def test_composite_inpaint_result_ramps_on_the_background_side_of_the_boundary():
    source = np.full((20, 20, 3), 10, dtype=np.uint8)
    inpainted = np.full((20, 20, 3), 200, dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255

    composite = composite_inpaint_result(source, mask, inpainted, feather_px=4)

    just_outside = composite[4, 10]  # one row above the hole, background side of the edge
    assert 10 < just_outside[0] < 200


def test_dilate_mask_grows_the_hole_by_the_given_radius():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[10, 10] = 255

    dilated = dilate_mask(mask, radius_px=2)

    assert dilated[10, 12] == 255  # 2px right, now inside the grown hole
    assert dilated[10, 14] == 0  # 4px right, still outside


def test_dilate_mask_is_a_noop_at_zero_radius():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[3:6, 3:6] = 255

    assert np.array_equal(dilate_mask(mask, radius_px=0), mask)


def test_composite_inpaint_result_returns_source_unchanged_when_mask_is_empty():
    source = np.full((10, 10, 3), 77, dtype=np.uint8)
    inpainted = np.full((10, 10, 3), 200, dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)

    composite = composite_inpaint_result(source, mask, inpainted)

    assert np.array_equal(composite, source)
