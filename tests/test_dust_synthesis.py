import json

import numpy as np
from PIL import ImageCms

from ai_prepress.io import LoadedImage, save
from training.dust_removal.synthesize import add_dust, generate_dataset

_SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _flat_image(value, size=128, dtype=np.uint16) -> LoadedImage:
    max_value = np.iinfo(dtype).max
    array = np.full((size, size, 3), round(value * max_value), dtype=dtype)
    return LoadedImage(array=array, icc_profile=_SRGB, bit_depth=array.dtype.itemsize * 8)


def test_dust_only_darkens_never_brightens():
    clean = _flat_image(0.8, size=128)
    dusty, spots = add_dust(clean, n_spots=(3, 3), rng=np.random.default_rng(0))
    assert dusty.array.astype(np.int64).sum() < clean.array.astype(np.int64).sum()
    assert len(spots) == 3


def test_dust_is_localized_not_global():
    clean = _flat_image(0.8, size=256)
    dusty, _ = add_dust(
        clean, n_spots=(1, 1), radius_px=(3.0, 5.0), edge_softness_px=(1.0, 1.0), rng=np.random.default_rng(1)
    )
    diff = np.abs(dusty.array.astype(np.int32) - clean.array.astype(np.int32)).max(axis=-1)
    affected = np.count_nonzero(diff > 0)
    # one small spot on a 256x256 image should touch a small fraction of pixels, not the whole frame
    assert 0 < affected < 0.05 * clean.array[..., 0].size


def test_deterministic_with_a_fixed_seed():
    clean = _flat_image(0.6)
    a, spots_a = add_dust(clean, rng=np.random.default_rng(42))
    b, spots_b = add_dust(clean, rng=np.random.default_rng(42))
    assert np.array_equal(a.array, b.array)
    assert [s.__dict__ for s in spots_a] == [s.__dict__ for s in spots_b]


def test_preserves_bit_depth_and_icc_profile():
    clean = _flat_image(0.5, dtype=np.uint16)
    dusty, _ = add_dust(clean, rng=np.random.default_rng(2))
    assert dusty.array.dtype == np.uint16
    assert dusty.icc_profile == clean.icc_profile


def test_generate_dataset_writes_valid_coco_annotations(tmp_path):
    source_dir = tmp_path / "clean"
    source_dir.mkdir()
    for i in range(2):
        save(_flat_image(0.7, size=64), source_dir / f"img_{i}.tiff")

    output_dir = tmp_path / "out"
    annotation_path = generate_dataset(source_dir, output_dir, seed=0)

    data = json.loads(annotation_path.read_text())
    assert len(data["images"]) == 2
    assert len(data["annotations"]) > 0
    assert all((output_dir / "images" / img["file_name"]).exists() for img in data["images"])
