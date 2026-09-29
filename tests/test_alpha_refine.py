import cv2
import numpy as np
import pytest

from ai_prepress.alpha_refine import refine_alpha

# a hard vertical edge in the source RGB, with a coarse alpha simulating a Lanczos-upsampled
# transition around it - synthetic, but sigma=8 sits inside the real range this stage exists to
# fix (median 13px, 8-38px measured on portrait.png's hair edge, see alpha_refine.py's docstring)
_HEIGHT, _WIDTH = 200, 200
_EDGE_X = 100
_BLUR_SIGMA = 8.0


def _hard_edge_source() -> np.ndarray:
    source_rgb = np.zeros((_HEIGHT, _WIDTH, 3), dtype=np.uint8)
    source_rgb[:, _EDGE_X:] = 255
    return source_rgb


def _blurred_coarse_alpha() -> np.ndarray:
    sharp_alpha = np.zeros((_HEIGHT, _WIDTH), dtype=np.float32)
    sharp_alpha[:, _EDGE_X:] = 255
    return cv2.GaussianBlur(sharp_alpha, (0, 0), sigmaX=_BLUR_SIGMA).astype(np.uint8)


def _transition_width_px(row: np.ndarray) -> float | None:
    """10%-90% transition width in pixels, anchored to the row's actual plateau levels (the
    median of its first/last 20px) rather than its raw min/max. Anchoring to min/max looked right
    on a flat synthetic guide but silently broke on real photo texture this session: a single
    texture-noise pixel far from the edge could sit above/below the true plateau and get read as
    the start of the transition, reporting a width that was really just noise-to-edge distance.
    Walking outward from the 50% crossing instead means a stray wiggle far from the edge can't
    corrupt the number - same fix used to validate DEFAULT_RADIUS/DEFAULT_EPS against real texture."""
    values = row.astype(np.float64)
    low = np.median(values[:20])
    high = np.median(values[-20:])
    if high < low:
        low, high, values = high, low, values[::-1]
    if high - low < 50:
        return None
    unit = np.clip((values - low) / (high - low), 0.0, 1.0)
    midpoint = int(np.argmax(unit >= 0.5))
    left = midpoint
    while left > 0 and unit[left] > 0.1:
        left -= 1
    right = midpoint
    while right < len(unit) - 1 and unit[right] < 0.9:
        right += 1
    return float(right - left)


def _median_transition_width(alpha: np.ndarray, rows: range) -> float:
    widths = [_transition_width_px(alpha[row]) for row in rows]
    widths = [w for w in widths if w is not None]
    return float(np.median(widths))


def test_refine_alpha_sharpens_a_blurred_edge_against_the_real_source_edge():
    source_rgb = _hard_edge_source()
    coarse_alpha = _blurred_coarse_alpha()
    rows = range(20, _HEIGHT - 20, 10)

    refined = refine_alpha(coarse_alpha, source_rgb)

    coarse_width = _median_transition_width(coarse_alpha, rows)
    refined_width = _median_transition_width(refined, rows)
    assert refined_width < coarse_width
    assert coarse_width == pytest.approx(21.0, abs=1.0)
    assert refined_width <= 2.0


def test_refine_alpha_preserves_shape_and_dtype():
    source_rgb = _hard_edge_source()
    coarse_alpha = _blurred_coarse_alpha()

    refined = refine_alpha(coarse_alpha, source_rgb)

    assert refined.shape == coarse_alpha.shape
    assert refined.dtype == coarse_alpha.dtype


def test_refine_alpha_leaves_pixels_outside_visible_region_untouched():
    source_rgb = _hard_edge_source()
    coarse_alpha = _blurred_coarse_alpha()
    visible_region = np.zeros((_HEIGHT, _WIDTH), dtype=bool)
    visible_region[:, _EDGE_X - 50 : _EDGE_X + 50] = True

    refined = refine_alpha(coarse_alpha, source_rgb, visible_region=visible_region)

    outside = ~visible_region
    assert np.array_equal(refined[outside], coarse_alpha[outside])


def test_refine_alpha_does_not_pull_toward_an_edge_outside_the_visible_region():
    """The occlusion-boundary case from the research this module is based on: a second, unrelated
    edge sits just past the visible region's boundary (e.g. a chair edge behind an occluding
    person). Refinement must not bend the alpha toward it."""
    source_rgb = _hard_edge_source()
    # a second, unrelated edge outside the object's own visible region
    source_rgb[:, 160:] = 0
    coarse_alpha = _blurred_coarse_alpha()
    visible_region = np.zeros((_HEIGHT, _WIDTH), dtype=bool)
    visible_region[:, _EDGE_X - 30 : _EDGE_X + 30] = True

    refined = refine_alpha(coarse_alpha, source_rgb, visible_region=visible_region)

    untouched_far_side = refined[:, 160:]
    assert np.array_equal(untouched_far_side, coarse_alpha[:, 160:])


def test_refine_alpha_default_visible_region_comes_from_the_coarse_alpha_itself():
    """No explicit visible_region given, so refine_alpha derives one from the coarse alpha's own
    footprint - dilated by `radius` so the visible/not-visible cut lands on a flat plateau instead
    of inside the soft edge (see DEFAULT_VISIBLE_REGION_THRESHOLD's comment). The object here is a
    band, not an open-ended fill, so there's background on the far side wide enough that some of
    it sits outside even the dilated reach - proving the default mask doesn't just default to
    "everything," which would defeat the whole point of having it."""
    width = 500
    band_start, band_end = 150, 300
    source_rgb = np.zeros((_HEIGHT, width, 3), dtype=np.uint8)
    source_rgb[:, band_start:band_end] = 255
    sharp_alpha = np.zeros((_HEIGHT, width), dtype=np.float32)
    sharp_alpha[:, band_start:band_end] = 255
    coarse_alpha = cv2.GaussianBlur(sharp_alpha, (0, 0), sigmaX=_BLUR_SIGMA).astype(np.uint8)
    # far enough past band_end that even a radius=DEFAULT_RADIUS dilation can't reach it - given a
    # distinguishable nonzero value so a guided-filter pass reaching it (a bug) would visibly
    # change it, rather than leaving a flat-zero region that would look "untouched" either way
    far_patch = slice(width - 20, width)
    coarse_alpha[:, far_patch] = 5  # below DEFAULT_VISIBLE_REGION_THRESHOLD, so this alone isn't
    # what keeps it out of the default mask - only being far past the dilated band is

    refined = refine_alpha(coarse_alpha, source_rgb)

    assert np.array_equal(refined[:, far_patch], coarse_alpha[:, far_patch])


def test_refine_alpha_rejects_mismatched_resolution():
    source_rgb = np.zeros((_HEIGHT, _WIDTH, 3), dtype=np.uint8)
    coarse_alpha = np.zeros((_HEIGHT + 1, _WIDTH), dtype=np.uint8)

    with pytest.raises(ValueError):
        refine_alpha(coarse_alpha, source_rgb)
