"""Layer separation: photo -> per-object RGBA layers + a reconstructed background plate.

Orchestrates ai_prepress.layer_decompose (the remote Qwen-Image-Layered client) - this is the
first features/ module that calls a remote model directly, since face_parsing.py's Modal
precedent was historically wired straight into api/main.py and is now retired from the UI (see
that module's docstring).

Bit-depth handling is the one piece of real logic here: the remote model's own RGB output is
8-bit diffusion-model output, discarded per-layer in favor of the ORIGINAL image's own
full-precision RGB (this project's whole identity is never silently collapsing bit depth, and a
higher-fidelity source already exists for every non-occluded layer pixel). Only the background
plate is genuinely novel content - reconstructed pixels with no source data behind them - so it
stays honestly 8-bit, a documented exception rather than an implicit gap.

Two more remote models sit around the main decomposition call, both explicitly ENHANCEMENTS over
a working fallback, never hard dependencies of it - a bad day for either one must never take down
the core separation result:
- ai_prepress.object_count (SAM2) estimates how many distinct objects are in the photo BEFORE
  decomposing, so `layers` is sized per-photo instead of a fixed constant. If it fails for any
  reason, falls back to FALLBACK_LAYER_COUNT (today's old hardcoded default).
- ai_prepress.layer_naming (InternVL3.5-2B) suggests a short label per layer AFTER decomposing,
  to pre-fill the UI's rename field. If it fails for any reason, every layer's `suggested_name`
  is just None and the UI falls back to its existing generic "Layer N" placeholder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import ImageCms

from ai_prepress import layer_decompose, layer_naming, object_count
from ai_prepress.io import LoadedImage, from_unit_float, to_unit_float

MIN_LAYERS = 2
MAX_LAYERS = 8
# first-cut bounds on the auto-computed layer count, correct against real photos like every
# other untested numeric constant this project has shipped and then measured
FALLBACK_LAYER_COUNT = 4  # today's old hardcoded default - now only reached if object-counting
# itself fails, not the normal path


@dataclass
class SeparatedLayer:
    image: LoadedImage  # (H, W, 4), dtype matches the source image's own bit depth
    bbox: tuple[int, int, int, int]  # (x0, y0, x1, y1), derived from the alpha mask's own extent
    suggested_name: str | None = None  # from layer_naming - None if naming failed or wasn't run


@dataclass
class SeparatedLayers:
    background: LoadedImage  # (H, W, 3), 8-bit - inherently model-generated, see module docstring
    layers: list[SeparatedLayer]
    requested_layers: int  # the actual N sent to the model - not len(layers)+1, which is what
    # came BACK; surfaced for transparency while MIN_LAYERS/MAX_LAYERS/the count formula still
    # need tuning against real photos, see _estimate_layer_count


BBOX_ALPHA_THRESHOLD = 127  # majority-opaque, not "any nonzero" - see below. Not underscore-
# prefixed since api/main.py's alpha_coverage stat reuses it for the same reason.

def _bbox_from_alpha(alpha: np.ndarray) -> tuple[int, int, int, int]:
    """A real deployment against Qwen-Image-Layered (not a synthetic test fixture) showed the
    model's own alpha output isn't clean binary - it carries widespread low-level noise (values
    1-10) spread across nearly the whole frame, not just the object's own soft edge. `alpha > 0`
    picked that noise up and blew the bbox out to cover almost the entire image on a real photo.
    A majority-opacity threshold (>127) tracks the object's actual visible extent instead, while
    compositing (separate_layers below) still uses the full soft alpha unclamped, since edge
    softness is wanted there - this threshold is only for "where is this layer's content," not
    for the alpha values a caller actually gets back."""
    coords = np.argwhere(alpha > BBOX_ALPHA_THRESHOLD)
    if coords.size == 0:
        return (0, 0, 0, 0)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    return (int(x0), int(y0), int(x1) + 1, int(y1) + 1)


def _estimate_layer_count(image: LoadedImage) -> int:
    try:
        estimated_objects = object_count.count_objects(image)
    except Exception:
        # object-counting is an enhancement over a fixed default, never a hard dependency of
        # separation itself - a down/erroring endpoint must not fail the whole request
        return FALLBACK_LAYER_COUNT
    return max(MIN_LAYERS, min(MAX_LAYERS, estimated_objects + 1))  # +1 for the background plate


def separate_layers(image: LoadedImage, endpoint: str | None = None) -> SeparatedLayers:
    """Returns every remote-detected layer as a full-precision RGBA LoadedImage, plus the
    background plate. Never returns None - unlike retouch_faces' "no face found" case, a photo
    with nothing separable just comes back with an empty `layers` list, still a valid result."""
    layers_requested = _estimate_layer_count(image)
    decomposed = layer_decompose.decompose_layers(image, endpoint=endpoint, layers=layers_requested)

    icc_profile = image.icc_profile
    if icc_profile is None:
        # io.save() requires one - same fallback retouch_faces.py and match_look.py use
        icc_profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

    source_rgb_unit = to_unit_float(image.array)[..., :3]

    layers = []
    for alpha in decomposed.layer_alphas:
        alpha_unit = to_unit_float(alpha)
        rgba_unit = np.dstack([source_rgb_unit, alpha_unit])
        layer_image = LoadedImage(
            array=from_unit_float(rgba_unit, image.array.dtype),
            icc_profile=icc_profile,
            bit_depth=image.bit_depth,
        )
        layers.append(SeparatedLayer(image=layer_image, bbox=_bbox_from_alpha(alpha)))

    try:
        suggested_names = layer_naming.name_layers([layer.image for layer in layers])
    except Exception:
        # naming is cosmetic - it only ever pre-fills a rename field the UI already has, so a
        # failure here must not take down an otherwise-successful separation
        suggested_names = [None] * len(layers)
    for layer, name in zip(layers, suggested_names):
        layer.suggested_name = name

    background = LoadedImage(array=decomposed.background, icc_profile=icc_profile, bit_depth=8)
    return SeparatedLayers(background=background, layers=layers, requested_layers=layers_requested)
