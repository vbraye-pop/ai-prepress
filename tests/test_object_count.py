import io

import httpx
import numpy as np
import pytest
from PIL import Image, ImageCms

from ai_prepress.io import LoadedImage
from ai_prepress.object_count import count_objects
from deploy.object_count import dedupe_overlapping_masks, drop_background_masks, filter_masks

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


def _rect_mask(x, y, w, h, size=20):
    segmentation = np.zeros((size, size), dtype=bool)
    segmentation[y : y + h, x : x + w] = True
    return {
        "bbox": [x, y, w, h],
        "area": int(segmentation.sum()),
        "segmentation": segmentation,
        "predicted_iou": 0.95,
        "stability_score": 0.95,
    }


def test_drop_background_masks_removes_multi_edge_fill():
    background = _rect_mask(0, 0, 100, 60, size=100)  # touches left, top, right
    foreground = _rect_mask(40, 70, 20, 20, size=100)  # touches nothing

    kept = drop_background_masks([background, foreground], width=100, height=100)

    assert kept == [foreground]


def test_drop_background_masks_keeps_a_large_mask_touching_only_one_edge():
    # a standing person filling most of the frame but touching only the bottom edge - the
    # regression case a naive single-edge-touch cut would wrongly drop
    person = _rect_mask(10, 20, 80, 80, size=100)  # touches only the bottom edge (y2 == height)

    kept = drop_background_masks([person], width=100, height=100)

    assert kept == [person]


def test_dedupe_overlapping_masks_keeps_adjacent_objects_despite_bbox_containment():
    # the table's bbox fully contains the orange's bbox, but their real pixels never overlap -
    # bbox-only containment would wrongly merge these into one object
    table = _rect_mask(0, 0, 20, 20, size=20)
    table["segmentation"][7:12, 7:12] = False
    table["area"] = int(table["segmentation"].sum())
    orange = _rect_mask(7, 7, 5, 5, size=20)

    kept = dedupe_overlapping_masks([table, orange])

    assert {id(m) for m in kept} == {id(table), id(orange)}


def test_dedupe_overlapping_masks_drops_a_sub_region_of_an_already_kept_mask():
    whole_object = _rect_mask(0, 0, 10, 10, size=20)
    sub_region = _rect_mask(0, 0, 10, 5, size=20)  # exactly the top half of whole_object

    kept = dedupe_overlapping_masks([whole_object, sub_region])

    assert kept == [whole_object]


def test_dedupe_overlapping_masks_containment_and_iou_thresholds_are_independent():
    # a small mask fully inside a much larger one: containment is high, IoU is low because the
    # union is dominated by the larger mask. Passing a high iou_thresh alongside the default
    # containment_thresh must still catch this as a duplicate via containment alone.
    whole_object = _rect_mask(0, 0, 20, 20, size=20)
    small_part = _rect_mask(0, 0, 4, 4, size=20)  # fully inside whole_object

    kept = dedupe_overlapping_masks(
        [whole_object, small_part], containment_thresh=0.9, iou_thresh=0.99
    )

    assert kept == [whole_object]


def test_drop_background_masks_respects_the_border_tolerance():
    # inset by 1px from every edge - still counts as "touching" within the default 2px tolerance,
    # which exists because min_mask_region_area's connected-component pass can leave that kind of
    # small margin around a true edge-to-edge background region
    background = _rect_mask(1, 1, 98, 98, size=100)

    kept = drop_background_masks([background], width=100, height=100)

    assert kept == []


def test_filter_masks_combines_both_passes():
    background = _rect_mask(0, 0, 20, 12, size=20)  # touches left, top, right
    whole_object = _rect_mask(0, 15, 10, 5, size=20)
    sub_region = _rect_mask(0, 15, 10, 2, size=20)  # top strip of whole_object, same object

    kept = filter_masks([background, whole_object, sub_region], width=20, height=20)

    assert kept == [whole_object]
