"""Modal deployment for deterministic background-plate inpainting: the fourth stage of the
per-instance separation pipeline, reconstructing the background once the union of every detected
object's mask has been removed (see features/layer_separation.py for the earlier stages this
follows - object counting, per-instance decomposition, contamination repair).

Model: LaMa (advimman/lama's big-lama checkpoint), loaded through the `simple-lama-inpainting`
PyPI package rather than the raw repo - Apache-2.0, confirmed directly on that package's own
GitHub repo license metadata, not assumed from the LaMa name. It's a thin TorchScript wrapper
(one `.pt` file, no manual OmegaConf config setup the raw repo's own predict.py needs) around the
same real big-lama weights, exactly the kind of dependency-surface reduction this project already
did once for SAM2/InternVL (see deploy/object_count.py, deploy/layer_naming.py).

Two real bugs were found by reading the wrapper's own source directly (its PyPI 0.1.2 wheel),
neither documented anywhere on the package's README, and both are worked around below rather than
inherited silently:

1. Missing crop-back-off-the-pad. `SimpleLama.__call__` runs every image/mask through
   `prepare_img_and_mask(..., pad_out_to_modulo=8)`, which symmetric-pads both up to a multiple of
   8 in each dimension before the forward pass - but the wrapper never crops that pad back off the
   result before returning it. The original advimman/lama `predict.py` this wraps DOES crop
   (`cur_res[:orig_height, :orig_width]`); this wrapper dropped that line. Left alone, any input
   whose dimensions aren't already a multiple of 8 comes back misaligned by up to 7px along the
   bottom and right edges. Fixed below with an explicit `.crop((0, 0, w, h))` on the wrapper's
   result before it's used for anything - see the crop-fix test in tests/test_background_inpaint.py
   for a non-multiple-of-8 regression case.

2. Hard mask binarization with no soft-mask warning. `prepare_img_and_mask` does
   `out_mask = (out_mask > 0) * 1` - ANY nonzero pixel is treated as full hole, not weighted by how
   nonzero it is. A feathered/soft mask sent here doesn't get a softer inpaint; its entire soft-edge
   support (which can extend well past the object's real boundary, see alpha_refine.py's own
   measurements of how wide a real coarse alpha's transition band gets) silently becomes 100% hole.
   This project already hit exactly this class of bug once before: BBOX_ALPHA_THRESHOLD in
   features/layer_separation.py exists specifically because raw model alpha carries low-level noise
   across nearly the whole frame, and treating "any nonzero" as signal there produced garbage
   bounding boxes until a real threshold was added. The contract here is the same lesson applied
   up front: the incoming mask MUST be hard binary 0/255, enforced by rejecting anything else (see
   is_hard_binary_mask below) rather than silently reproducing that bug a second time.

Deploy: uv run --group deploy modal deploy deploy/background_inpaint.py

Resolution handling: prepress sources can be 20+ megapixels and LaMa's own network has no internal
tiling or size cap of its own. MAX_INPAINT_EDGE below downscales source+mask to a bounded working
resolution before the forward pass - 2048 is a first-cut guess, not a measured optimum, anchored on
BiRefNet's own native training resolution purely as a convenient existing number already in this
project's vocabulary (see features/layer_separation.py's own RESOLUTION_BUCKET commentary for the
same "state the anchor honestly, correct it against real measurement later" treatment of every
other resolution constant here). Only the actually-generated hole content ever comes from that
downscaled pass - composite_inpaint_result() below pastes it back into the ORIGINAL full-precision
source, so every pixel outside the mask keeps the source's own precision untouched regardless of
MAX_INPAINT_EDGE.

The mask LaMa actually sees is dilated by INPAINT_MASK_FEATHER_PX before the forward pass
(dilate_mask below), not the caller's exact mask - two independent reasons, both real:
1. composite_inpaint_result's own feather ramp needs to land on genuinely LaMa-synthesized
   background, not on unprocessed context pixels, or the "soften the seam" blend would really be
   blending the removed object's OWN edge pixels back in (see that function's docstring for why
   that's a real, not hypothetical, ghosting bug).
2. The mask gets NEAREST-downscaled to MAX_INPAINT_EDGE resolution before this call - a hole pixel
   right at the caller's exact boundary can round to "not hole" at the working resolution, which
   would leave LaMa treating a sliver of the removed object as valid context to preserve rather
   than content to synthesize. A few pixels of dilation margin is meant to absorb that rounding
   error too - checked so far only indirectly, by confirming a real downscaled request (art.jpg
   upscaled 2.5x to force scale < 1.0) comes back with every far-from-the-hole pixel byte-exact
   and no visible misalignment crescent at the hole boundary, not by an isolated ablation that
   disables dilation and shows the rounding artifact appearing without it.

GPU tier T4, `@app.cls` + `@modal.enter()` rather than a plain `@app.function`: `torch.jit.load`-ing
the TorchScript module fresh on every request would be real, avoidable overhead for what should
otherwise be a single fast feed-forward pass once the input is capped at MAX_INPAINT_EDGE. Plain
synchronous endpoint, no submit/poll split - but note a genuinely huge upload body (a raw 20+MP
source file, before any of the downscaling above happens) itself eats into Modal's 150s
web-endpoint ceiling independent of how fast the capped-resolution compute itself is, since the
body has to finish uploading before this function even starts running.

Checkpoint is baked into the image at build time (`SimpleLama(device="cpu")` inside run_commands),
not left to download lazily on first request - same bake-not-download pattern as every other
deploy script here. device="cpu" is deliberate for the BUILD step specifically: Modal's image-build
step has no GPU attached, so "cpu" is the only device that can construct the object there at all;
it still forces the real big-lama.pt TorchScript weights into the image layer's torch.hub
checkpoint cache, which @modal.enter() below then loads from disk (no re-download) with the real
device="cuda" at request time.

torch/simple_lama_inpainting/PIL are only ever imported inside method bodies below, never at
module level - same reasoning as every other deploy script in this project (the local venv
deliberately doesn't carry torch). numpy/cv2 are the deliberate exception, same as
deploy/object_count.py and ai_prepress/alpha_refine.py: both are already main dependencies here,
and importing them at module level is what lets compute_inpaint_scale/is_hard_binary_mask/
composite_inpaint_result below be imported and unit-tested locally without ever touching torch.
"""

import io

import cv2
import modal
import numpy as np
from fastapi import File, HTTPException, Response, UploadFile

app = modal.App("ai-prepress-background-inpaint")

image = (
    modal.Image.debian_slim(python_version="3.12")
    # simple-lama-inpainting's own PyPI metadata pins `pillow>=9.5.0,<10.0.0` - and no 9.x Pillow
    # release ever shipped a Python 3.12 wheel, confirmed directly against a real deploy attempt
    # (this project's own first try at this file): normal dependency resolution pulled that pin in,
    # pip fell back to building Pillow 9.x from source, and that build failed outright on
    # debian_slim's missing zlib headers. Installed here with modern, wheel-available versions
    # FIRST, then simple-lama-inpainting itself goes in separately with --no-deps so its own
    # pin never enters resolution at all. opencv-python-headless is added explicitly (not left to
    # --no-deps drop it) since it's the one real runtime import simple-lama-inpainting's own
    # utils/util.py needs (`import cv2`) beyond what's already installed here; `fire` (its other
    # real dependency) is skipped on purpose - it backs the package's own CLI entry point only,
    # never imported by the `SimpleLama` class this file actually calls.
    .pip_install(
        "torch", "torchvision", "pillow", "numpy", "opencv-python-headless", "fastapi",
        "python-multipart",
    )
    .pip_install("simple-lama-inpainting", extra_options="--no-deps")
    .run_commands(
        "python -c \"from simple_lama_inpainting import SimpleLama; SimpleLama(device='cpu')\""
    )
)

# First-cut guess, not a measured optimum yet - see the module docstring's "Resolution handling"
# section for the BiRefNet-anchor reasoning and what would need to change to correct this.
MAX_INPAINT_EDGE = 2048

# Two roles, both in pixels, deliberately the same constant (see the module docstring's
# "Resolution handling" section for why they need to match): how far dilate_mask grows the mask
# LaMa actually solves against, and how far composite_inpaint_result's alpha ramp extends outward
# from the hole into that dilated margin. 6 sits in the middle of a reasoned 4-8px band: narrow
# enough not to visibly soften real LaMa detail near the boundary, wide enough that a straight
# per-pixel switch (the 0px case) doesn't read as a hard seam on a print-resolution source.
INPAINT_MASK_FEATHER_PX = 6


def is_hard_binary_mask(mask: np.ndarray) -> bool:
    """True iff every pixel is exactly 0 or 255 - the contract this endpoint requires (see the
    module docstring's bug #2). Anything else (an 8-bit gradient, JPEG ringing, an anti-aliased
    polygon edge) must be rejected here rather than silently swallowed by the wrapper's own
    `(out_mask > 0) * 1` binarization, which would turn a soft edge into full-hole well past the
    object's real boundary."""
    return bool(np.isin(mask, (0, 255)).all())


def compute_inpaint_scale(height: int, width: int, max_edge: int = MAX_INPAINT_EDGE) -> float:
    """1.0 if the longer edge already fits within max_edge, otherwise the downscale factor that
    brings it exactly to max_edge. Never upscales."""
    longest = max(height, width)
    if longest <= max_edge:
        return 1.0
    return max_edge / longest


def dilate_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Grow a hard-binary hole by radius_px pixels. Used to give LaMa's own forward pass a margin
    beyond the caller's exact mask edge - see the module docstring's "Resolution handling" section
    for why that margin matters (it's what makes composite_inpaint_result's feather ramp land on
    real synthesized content instead of raw context pixels, and what absorbs the mask-downscale
    rounding error at MAX_INPAINT_EDGE). No-op at radius_px <= 0."""
    if radius_px <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1))
    return cv2.dilate(mask, kernel)


def composite_inpaint_result(
    source_rgb: np.ndarray,
    mask: np.ndarray,
    inpainted_rgb: np.ndarray,
    feather_px: int = INPAINT_MASK_FEATHER_PX,
) -> np.ndarray:
    """Paste `inpainted_rgb` (already resized to source_rgb's resolution, and produced against a
    `mask` dilated by dilate_mask - see the endpoint below) into `source_rgb`.

    Every pixel where `mask` is nonzero gets alpha == 1.0 (fully LaMa's own content) with no
    exception - this was NOT the first version of this function. An earlier version ramped alpha
    INWARD from the mask edge, which blends the hole's own edge with `source_rgb` there - but
    `source_rgb` inside the hole IS the object being removed, so that ramp put a `feather_px`-wide
    ring of the removed object's own highest-contrast edge pixels back into the result. That is
    real, visible ghosting, not a hypothetical, so alpha inside the hole is now unconditionally 1.0.

    The feather instead softens the BACKGROUND side of the boundary only: alpha ramps down from
    (near) 1.0 immediately outside the hole to 0.0 by `feather_px` pixels further out, using
    `cv2.distanceTransform` on the non-hole region to get each background pixel's distance to the
    nearest hole pixel. Because the mask LaMa actually ran against was dilated by this same
    `feather_px` (dilate_mask, called before the forward pass), this ramp blends real
    LaMa-synthesized background against real source background - never the removed object itself,
    and never raw unprocessed context passed straight through. Pixels more than `feather_px` away
    from the hole are untouched source, exactly.
    """
    hole = mask > 0
    if not hole.any():
        return source_rgb.copy()

    if feather_px > 0:
        distance_outside = cv2.distanceTransform((~hole).astype(np.uint8), cv2.DIST_L2, 5)
        alpha_background = np.clip(1.0 - distance_outside / feather_px, 0.0, 1.0)
    else:
        alpha_background = np.zeros(hole.shape, dtype=np.float64)
    alpha = np.where(hole, 1.0, alpha_background)[..., np.newaxis]

    composite = source_rgb.astype(np.float64) * (1 - alpha) + inpainted_rgb.astype(np.float64) * alpha
    return np.clip(composite + 0.5, 0, 255).astype(np.uint8)


def crop_to_source_size(inpainted, width: int, height: int):
    """Crop LaMa's own output back down to the exact pre-pad (width, height) - see the module
    docstring's bug #1. `inpainted` is a PIL Image; kept untyped in the signature so this stays
    importable (and testable, see tests/test_background_inpaint.py) without a PIL import at
    module level, matching this project's torch/PIL-only-inside-method-bodies convention."""
    return inpainted.crop((0, 0, width, height))


@app.cls(image=image, gpu="T4", scaledown_window=120)
class BackgroundInpainter:
    @modal.enter()
    def load(self):
        from simple_lama_inpainting import SimpleLama

        self.lama = SimpleLama(device="cuda")

    @modal.fastapi_endpoint(method="POST")
    async def inpaint(self, file: UploadFile = File(...), mask: UploadFile = File(...)) -> Response:
        from PIL import Image

        source = Image.open(io.BytesIO(await file.read())).convert("RGB")
        mask_image = Image.open(io.BytesIO(await mask.read())).convert("L")

        if mask_image.size != source.size:
            raise HTTPException(
                status_code=400,
                detail=f"mask size {mask_image.size} does not match source size {source.size}",
            )

        mask_array = np.array(mask_image)
        if not is_hard_binary_mask(mask_array):
            raise HTTPException(
                status_code=400,
                detail="mask must be hard binary 0/255 - see deploy/background_inpaint.py's "
                "module docstring for why a soft/feathered mask is rejected rather than accepted",
            )

        # dilated, not the caller's exact mask - this is what LaMa's own forward pass solves
        # against, see the module docstring's "Resolution handling" section for why
        solve_mask_array = dilate_mask(mask_array, INPAINT_MASK_FEATHER_PX)
        solve_mask_image = Image.fromarray(solve_mask_array)

        width, height = source.size
        scale = compute_inpaint_scale(height, width)
        if scale < 1.0:
            small_size = (max(1, round(width * scale)), max(1, round(height * scale)))
            small_source = source.resize(small_size, Image.Resampling.LANCZOS)
            # NEAREST, not LANCZOS: the mask must stay hard binary through the resize, any
            # smoothing resampler would reintroduce exactly the soft-edge values this endpoint
            # just rejected the caller for sending
            small_mask = solve_mask_image.resize(small_size, Image.Resampling.NEAREST)
        else:
            small_source, small_mask = source, solve_mask_image

        result = self.lama(small_source, small_mask)
        # BUG FIX (see module docstring #1 and crop_to_source_size's own docstring): the wrapper
        # pads to a multiple of 8 internally and never crops the pad back off on its own.
        result = crop_to_source_size(result, small_source.width, small_source.height)

        if scale < 1.0:
            result = result.resize((width, height), Image.Resampling.LANCZOS)

        composite = composite_inpaint_result(np.array(source), mask_array, np.array(result))

        buffer = io.BytesIO()
        Image.fromarray(composite).save(buffer, format="PNG")
        return Response(content=buffer.getvalue(), media_type="image/png")
