import numpy as np

from ai_prepress.api.main import _PREVIEW_MAX_EDGE, _downsample_for_preview


def test_small_array_passes_through_unchanged():
    array = np.zeros((100, 200, 3), dtype=np.uint8)
    assert _downsample_for_preview(array) is array


def test_large_array_is_capped_to_max_edge():
    array = np.zeros((6000, 8000, 3), dtype=np.uint16)
    small = _downsample_for_preview(array)
    assert max(small.shape[0], small.shape[1]) <= _PREVIEW_MAX_EDGE
    assert small.shape[2] == 3
