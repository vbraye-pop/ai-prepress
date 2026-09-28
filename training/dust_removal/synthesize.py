"""Synthesize sensor-dust artifacts for training a dust-spot detector.

Dust sits on the sensor's cover glass, out of the lens's focal plane, so it shows up as a soft,
partially-transparent shadow rather than a sharp dark mark. Modeled here as multiplicative
attenuation applied in linear light, matching how DxO describes building their own detector's
training data - applying it to gamma-encoded pixels directly gives the wrong falloff shape, so
every spot goes through the image's own cctf_decoding/cctf_encoding pair (via
identify_colourspace, same as match_look) rather than a hardcoded sRGB gamma.

RGB only for now, same limitation as match_look - a fourth channel (alpha) is left untouched if
present, not processed.

This only produces detection targets (bounding boxes). Dust fill stays a separate, deliberately
non-generative step later - see the project README for why.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import colour
import numpy as np

from ai_prepress.io import LoadedImage, from_unit_float, identify_colourspace, load, save, to_unit_float

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


@dataclass
class DustSpot:
    cx: float
    cy: float
    radius: float
    edge_softness: float
    strength: float  # fraction of light blocked at the spot's center, 0-1

    def bbox(self) -> tuple[int, int, int, int]:
        """Ground-truth box, padded past the visible radius to cover the soft edge falloff."""
        pad = self.radius + self.edge_softness
        return (
            round(self.cx - pad),
            round(self.cy - pad),
            round(self.cx + pad),
            round(self.cy + pad),
        )


def _soft_disc(height: int, width: int, spot: DustSpot) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    dist = np.hypot(xx - spot.cx, yy - spot.cy)
    return 1.0 - np.clip((dist - spot.radius) / max(spot.edge_softness, 1e-3), 0.0, 1.0)


def add_dust(
    image: LoadedImage,
    n_spots: tuple[int, int] = (1, 4),
    radius_px: tuple[float, float] = (3.0, 18.0),
    strength: tuple[float, float] = (0.25, 0.75),
    edge_softness_px: tuple[float, float] = (1.5, 4.0),
    rng: np.random.Generator | None = None,
) -> tuple[LoadedImage, list[DustSpot]]:
    """Return a copy of `image` with synthetic dust spots added, plus their ground-truth boxes."""
    rng = rng or np.random.default_rng()
    height, width = image.array.shape[:2]
    space = colour.RGB_COLOURSPACES[identify_colourspace(image.icc_profile)]

    unit = to_unit_float(image.array)[..., :3]
    linear = space.cctf_decoding(unit)

    spots: list[DustSpot] = []
    count = int(rng.integers(n_spots[0], n_spots[1] + 1))
    for _ in range(count):
        spot = DustSpot(
            cx=float(rng.uniform(0, width)),
            cy=float(rng.uniform(0, height)),
            radius=float(rng.uniform(*radius_px)),
            edge_softness=float(rng.uniform(*edge_softness_px)),
            strength=float(rng.uniform(*strength)),
        )
        attenuation = 1.0 - _soft_disc(height, width, spot) * spot.strength
        linear = linear * attenuation[..., None]
        spots.append(spot)

    result_unit = np.clip(space.cctf_encoding(linear), 0.0, 1.0)
    result_array = image.array.copy()
    result_array[..., :3] = from_unit_float(result_unit, image.array.dtype)

    return (
        LoadedImage(array=result_array, icc_profile=image.icc_profile, bit_depth=image.bit_depth),
        spots,
    )


def generate_dataset(source_dir: Path, output_dir: Path, seed: int = 0) -> Path:
    """Run add_dust over every image in source_dir (assumed dust-free) and write a COCO-style
    annotation file alongside the augmented images in output_dir. Returns the annotation path."""
    rng = np.random.default_rng(seed)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "dust_spot"}]}
    next_annotation_id = 1

    source_paths = sorted(p for p in source_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    for image_id, path in enumerate(source_paths, start=1):
        clean = load(path)
        dusty, spots = add_dust(clean, rng=rng)

        out_name = f"{path.stem}_dust.tiff"
        save(dusty, images_dir / out_name)

        height, width = clean.array.shape[:2]
        coco["images"].append({"id": image_id, "file_name": out_name, "width": width, "height": height})
        for spot in spots:
            x0, y0, x1, y1 = spot.bbox()
            coco["annotations"].append(
                {
                    "id": next_annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "area": (x1 - x0) * (y1 - y0),
                    "iscrowd": 0,
                    "dust_strength": spot.strength,
                }
            )
            next_annotation_id += 1

    annotation_path = output_dir / "annotations.json"
    annotation_path.write_text(json.dumps(coco, indent=2))
    return annotation_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize sensor-dust training data.")
    parser.add_argument("source_dir", type=Path, help="directory of clean, dust-free images")
    parser.add_argument("output_dir", type=Path, help="where to write dusty images + annotations.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    annotation_path = generate_dataset(args.source_dir, args.output_dir, seed=args.seed)
    print(f"wrote {annotation_path}")


if __name__ == "__main__":
    main()
