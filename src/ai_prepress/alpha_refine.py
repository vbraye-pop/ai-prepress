"""Guided-filter alpha refinement: sharpen a layer's coarse alpha against the real edges of the
full-precision source RGB it was cut from.

Qwen-Image-Layered's own alpha comes back at diffusion-model resolution and gets Lanczos-upsampled
to source resolution before anything downstream sees it - soft by construction, not because the
real object edge is soft. Measured directly on a real hair edge (portrait.png, full source
resolution): the coarse alpha's 10%-90% transition runs a median 13px wide, 8-38px range, nowhere
near hair-sharp. A guided filter (cv2.ximgproc.guidedFilter) re-aligns the alpha to the source
RGB's own local structure, using the source image as the "guide" that tells the filter where a
real edge actually sits - see the comment above DEFAULT_RADIUS for what's actually been verified
about how well it does that on real photo texture versus on an idealized synthetic edge.

The one rule this module exists to enforce: refinement must never reach across an occlusion
boundary into territory a layer doesn't actually occupy. There's no real "chair edge" behind a
person's leg for the chair layer to align to, only the person's own edge - so filtering the whole
frame unscoped would pull an occluded layer's alpha toward a wrong boundary and make bleed worse,
not better. `visible_region` scopes refinement to where a layer is actually known to have content;
outside it, the coarse alpha passes through untouched.

NOT currently wired into features/layer_separation.py - see that module's own docstring for why.
The "calibrate against live model output" flag above got acted on: refine_alpha() was run against
the real coarse alpha this module's own 13px measurement came from (same portrait.png hair edge),
at DEFAULT_RADIUS/DEFAULT_EPS and a full sweep around them (radius 2-64, eps 1e-4 to 1e-1, plus a
trimap-style band limiting the filter to a narrow region around the coarse edge instead of the
whole visible_region). A plain 10%-90%-crossing measurement got WORSE at every setting that
changed the alpha by more than a rounding error. Row-level inspection of the refined output shows
why: real hair texture in the guide RGB reads as edge signal throughout what should be a flat
opaque interior (the alpha oscillates there instead of holding steady near 255), not only at the
true boundary - a boundary-crossing metric reads that as a much wider transition, whether or not
the true edge itself got sharper. Either way, no radius/eps combination was found that both
changed the coarse alpha meaningfully and held up under that measurement, so there's nothing left
to justify shipping it on. The 21px-to-~1px synthetic result above still holds for a clean
idealized edge; it just doesn't generalize to a real photographic one the way this module hoped. A
future fix needs a different approach (e.g. limiting the guide to gradient magnitude rather than
raw RGB, or matting-style trimap estimation), not a parameter retune.
"""

from __future__ import annotations

import cv2
import numpy as np

from ai_prepress.io import to_unit_float

# DEFAULT_RADIUS/DEFAULT_EPS provenance, honestly stated: a synthetic flat two-region step edge
# (this module's own unit tests) snaps from a 21px transition down to ~1px at these settings - a
# strong, clean result, but on an idealized guide with zero internal texture, which is the easy
# case for any local-regression filter. A second check this session built a real-texture guide
# instead (hair/sky patches cropped from portrait.png, same blur-then-refine setup, transition
# width measured by anchoring to the actual plateau levels rather than the row's raw min/max - the
# naive min/max version falsely reads texture noise far from the edge as part of the transition
# and was caught giving nonsense widths during that check). But the two crops were spliced into a
# synthetic butt-join, not a real photographic boundary, so its result (21px to 20.5px, no
# measurable change) is weak evidence, not a second confirmation - a genuine real-edge validation
# still needs an actual Qwen-Image-Layered coarse alpha, which this module's test suite
# deliberately doesn't call Modal for. What that check DOES support, more robustly: eps an order of
# magnitude smaller or larger than 5e-3 at this radius blew the same real-texture transition out
# past 100px, a large and repeatable failure mode (near-zero window variance or texture-swamped
# variance makes the box-filtered a/b coefficients a bad local fit) worth avoiding regardless of
# how the neutral-case number holds up. Flagged here for whoever wires this in: calibrate against
# live model output before trusting these defaults on a real photo.
DEFAULT_RADIUS = 64
DEFAULT_EPS = 5e-3  # in [0, 1]-normalized intensity units, matching guidedFilter's own convention
# the coarse alpha's own default stand-in for a real visible-region mask: low enough to keep the
# full soft edge (not just the >127 majority-opaque core layer_separation.py's bbox uses), high
# enough to drop the near-zero noise floor documented in that module. Dilated by DEFAULT_RADIUS so
# the visible/not-visible cut itself lands on a flat plateau instead of inside the soft edge -
# otherwise the hard np.where below would stitch refined and unrefined values together mid-
# transition and manufacture a visible seam right where sharpness matters most
DEFAULT_VISIBLE_REGION_THRESHOLD = 16


def refine_alpha(
    coarse_alpha: np.ndarray,
    source_rgb: np.ndarray,
    visible_region: np.ndarray | None = None,
    radius: int = DEFAULT_RADIUS,
    eps: float = DEFAULT_EPS,
) -> np.ndarray:
    """Align `coarse_alpha`'s edges to `source_rgb`'s real edges with a guided filter.

    `coarse_alpha` is (H, W) uint8, already at `source_rgb`'s resolution - layer_decompose.py's
    own alpha convention. `source_rgb` is (H, W, 3) in any dtype ai_prepress.io.to_unit_float
    handles. `visible_region` is an (H, W) mask (nonzero = visible) that pixels are only allowed
    to change within; every pixel outside it comes back exactly as `coarse_alpha` had it. Left as
    None, it defaults to `coarse_alpha` thresholded at DEFAULT_VISIBLE_REGION_THRESHOLD and dilated
    by `radius` (see the comment on DEFAULT_VISIBLE_REGION_THRESHOLD for why) - a placeholder good
    enough to test this module standalone, not a substitute for the real per-layer visibility check
    features/layer_separation.py will supply once it wires this in.
    """
    if coarse_alpha.ndim != 2:
        raise ValueError(f"coarse_alpha must be (H, W), got shape {coarse_alpha.shape}")
    if source_rgb.shape[:2] != coarse_alpha.shape:
        raise ValueError(
            f"source_rgb {source_rgb.shape[:2]} and coarse_alpha {coarse_alpha.shape} resolution mismatch"
        )

    if visible_region is None:
        core = (coarse_alpha > DEFAULT_VISIBLE_REGION_THRESHOLD).astype(np.uint8)
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        visible_region = cv2.dilate(core, kernel) > 0

    guide = to_unit_float(source_rgb).astype(np.float32)
    alpha_unit = coarse_alpha.astype(np.float32) / 255.0

    refined_unit = cv2.ximgproc.guidedFilter(guide=guide, src=alpha_unit, radius=radius, eps=eps)
    refined = np.clip(refined_unit * 255.0 + 0.5, 0, 255).astype(np.uint8)

    return np.where(visible_region, refined, coarse_alpha)
