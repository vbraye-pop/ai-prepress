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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import ImageCms

from ai_prepress import layer_decompose
from ai_prepress.io import LoadedImage, from_unit_float, to_unit_float


@dataclass
class SeparatedLayer:
    image: LoadedImage  # (H, W, 4), dtype matches the source image's own bit depth
    bbox: tuple[int, int, int, int]  # (x0, y0, x1, y1), derived from the alpha mask's own extent


@dataclass
class SeparatedLayers:
    background: LoadedImage  # (H, W, 3), 8-bit - inherently model-generated, see module docstring
    layers: list[SeparatedLayer]


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


def separate_layers(image: LoadedImage, endpoint: str | None = None) -> SeparatedLayers:
    """Returns every remote-detected layer as a full-precision RGBA LoadedImage, plus the
    background plate. Never returns None - unlike retouch_faces' "no face found" case, a photo
    with nothing separable just comes back with an empty `layers` list, still a valid result."""
    decomposed = layer_decompose.decompose_layers(image, endpoint=endpoint)

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

    background = LoadedImage(array=decomposed.background, icc_profile=icc_profile, bit_depth=8)
    return SeparatedLayers(background=background, layers=layers)
