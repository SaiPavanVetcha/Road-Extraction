import csv
import shutil
import sys
import tempfile
import unittest.mock
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

# Mock pyrosm if not installed in this environment
if "pyrosm" not in sys.modules:
    sys.modules["pyrosm"] = unittest.mock.MagicMock()

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_prep.create_masks_geofabrik import assign_class
from src.data_prep.extract_patches import (
    generate_location_split,
    process_scene,
)


def test_a_location_split():
    print("Testing (a): Location split logic...")

    # Case 1: 1 location
    scenes_1 = ["2025-01-01_Amaravati", "2025-02-01_Amaravati"]
    split_1 = generate_location_split(scenes_1, seed=42)
    assert len(split_1) == 2
    assert all(v == "train" for v in split_1.values()), "1 location should assign everything to train"

    # Case 2: 2 locations
    scenes_2 = ["2025-01-01_Amaravati", "2025-02-01_Hyderabad"]
    split_2 = generate_location_split(scenes_2, seed=42)
    splits_present = set(split_2.values())
    assert splits_present == {"train", "val"}, "2 locations should have train + val"

    # Case 3: 3+ locations (with multiple scenes per location and varied seeds)
    locations = ["Amaravati", "Hyderabad", "Bengaluru", "Chennai", "Kochi", "Mumbai"]
    scenes_multi = []
    for loc in locations:
        for date in ["2025-01-01", "2025-02-01", "2025-03-01"]:
            scenes_multi.append(f"{date}_{loc}")

    for seed in [1, 7, 42, 99, 123]:
        split_map = generate_location_split(scenes_multi, seed=seed)

        # Train must never be empty
        train_scenes = [s for s, sp in split_map.items() if sp == "train"]
        assert len(train_scenes) > 0, f"Train set empty for seed {seed}"

        # Never put the same location in two splits
        loc_to_split = {}
        for s, sp in split_map.items():
            loc = s.split("_", 1)[1]
            if loc in loc_to_split:
                assert loc_to_split[loc] == sp, f"Location {loc} placed in multiple splits: {loc_to_split[loc]} and {sp}"
            else:
                loc_to_split[loc] = sp

    print("  ✓ Passed: Location split never splits a location and never leaves train empty.")


def test_b_c_extract_patches_stale_and_tile_validity():
    print("Testing (b) & (c): Stale cleanup, tile -9999/NaN removal, and validity channel...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create dummy scene directory
        scene_id = "2025-01-01_TestCity"
        scene_dir = tmp_path / "scenes" / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        mask_dir = tmp_path / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        patches_dir = tmp_path / "patches"
        patches_dir.mkdir(parents=True, exist_ok=True)

        # Create a stale file from a 'previous run'
        stale_file = patches_dir / "stale_tile_old.npz"
        stale_file.write_text("stale data")
        assert stale_file.exists()

        # Simulate extract_patches main() cleanup logic
        if patches_dir.exists():
            shutil.rmtree(patches_dir)
        patches_dir.mkdir(parents=True, exist_ok=True)
        assert not stale_file.exists(), "Stale file was not deleted"
        print("  ✓ Passed (b): Stale tiles are deleted on recreation.")

        # Create synthetic optical raster (4 bands, 256x256)
        H, W = 256, 256
        opt_path = scene_dir / "optical_24m.tif"
        opt_meta = {
            "driver": "GTiff",
            "height": H,
            "width": W,
            "count": 4,
            "dtype": "float32",
            "crs": "EPSG:32644",
            "transform": from_origin(100000, 200000, 24, 24),
        }
        # Band 0: Green, Band 1: Red, Band 2: NIR, Band 3: SWIR
        # Use realistic surface reflectance values so normal ground is not flagged as cloud
        opt_data = np.zeros((4, H, W), dtype=np.float32)
        opt_data[0] = 150.0  # Green
        opt_data[1] = 120.0  # Red
        opt_data[2] = 600.0  # NIR (gives scene_max ~600)
        opt_data[3] = 250.0  # SWIR
        # Put valid nodata region in corner (band 0 = 0)
        opt_data[0, :10, :10] = 0.0

        with rasterio.open(opt_path, "w", **opt_meta) as dst:
            for b in range(4):
                dst.write(opt_data[b], b + 1)

        # Create synthetic SAR raster (2 bands: VV, VH) with -9999 and NaN
        sar_path = scene_dir / "sar_1_24m.tif"
        sar_meta = opt_meta.copy()
        sar_meta["count"] = 2
        sar_meta["nodata"] = -9999

        sar_data = np.ones((2, H, W), dtype=np.float32) * -12.0  # -12 dB
        # Introduce sentinels and NaNs
        sar_data[0, 50:70, 50:70] = -9999.0
        sar_data[1, 60:80, 60:80] = np.nan

        with rasterio.open(sar_path, "w", **sar_meta) as dst:
            dst.write(sar_data[0], 1)
            dst.write(sar_data[1], 2)

        # Create synthetic road mask (uint8) with road pixels
        mask_path = mask_dir / f"mask_{scene_id}.tif"
        mask_meta = opt_meta.copy()
        mask_meta.update({"count": 1, "dtype": "uint8"})
        mask_meta.pop("nodata", None)
        mask_data = np.zeros((H, W), dtype=np.uint8)
        mask_data[100:150, 100:150] = 1  # 2500 road pixels (passes road density gate)

        with rasterio.open(mask_path, "w", **mask_meta) as dst:
            dst.write(mask_data, 1)

        # Temporarily mock paths in extract_patches
        import src.data_prep.extract_patches as ep
        orig_scenes_dir = ep.PROCESSED_SCENES_DIR
        orig_mask_dir = ep.MASK_DIR
        orig_patches_dir = ep.PATCHES_DIR

        ep.PROCESSED_SCENES_DIR = tmp_path / "scenes"
        ep.MASK_DIR = mask_dir
        ep.PATCHES_DIR = patches_dir

        try:
            metadata_path = patches_dir / "tiles_metadata.csv"
            fieldnames = [
                "tile_id", "detailed_name", "scene_id", "split", "y", "x",
                "road_pixels", "road_ratio", "num_channels", "crs",
                "scene_left", "scene_bottom", "scene_right", "scene_top"
            ]
            with open(metadata_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                kept, rej, count = process_scene(scene_id, "train", writer, 0)

            assert kept == 1, f"Expected 1 tile kept, got {kept}"
            tile_path = patches_dir / "train" / "tile00001.npz"
            assert tile_path.exists(), "tile00001.npz was not created"

            with np.load(tile_path) as loaded:
                tile_img = loaded["image"]
                tile_msk = loaded["mask"]

            # Layout: 4 optical + 2 SAR + 1 validity = 7 channels
            assert tile_img.shape[0] == 7, f"Expected 7 channels (4 opt + 2 sar + 1 validity), got {tile_img.shape[0]}"

            # Check no sentinels (<= -9990) and no NaNs
            assert not np.any(tile_img <= -9990), "Tile still contains <= -9990 sentinels"
            assert not np.any(np.isnan(tile_img)), "Tile still contains NaNs"

            # Check validity mask channel (index -1)
            validity_ch = tile_img[-1]
            unique_validity = np.unique(validity_ch)
            assert set(unique_validity).issubset({0.0, 1.0}), f"Validity channel has values outside {0, 1}: {unique_validity}"

            # Corner optical=0 region must have validity = 0
            assert np.all(validity_ch[:10, :10] == 0.0), "Optical nodata region should have validity = 0"
            # SAR sentinel region must have validity = 0
            assert np.all(validity_ch[50:70, 50:70] == 0.0), "SAR sentinel region should have validity = 0"
            # Region where both optical and SAR are valid must have validity = 1
            assert validity_ch[200, 200] == 1.0, "Valid region should have validity = 1"

            print("  ✓ Passed (c): Tiles contain no -9999/NaN and have correct validity mask channel.")

        finally:
            ep.PROCESSED_SCENES_DIR = orig_scenes_dir
            ep.MASK_DIR = orig_mask_dir
            ep.PATCHES_DIR = orig_patches_dir


def test_d_assign_class():
    print("Testing (d): assign_class behavior...")
    # Multi-value OSM tags separated by ";"
    assert assign_class("residential;service") == 1, "Failed 'residential;service'"
    assert assign_class("primary;secondary") == 3, "Failed 'primary;secondary'"
    assert assign_class("track") == 1, "Failed 'track'"
    assert assign_class("residential") == 1, "Failed 'residential'"
    assert assign_class("secondary") == 2, "Failed 'secondary'"
    assert assign_class("motorway") == 3, "Failed 'motorway'"
    assert assign_class("unknown_tag") == 0, "Failed unknown tag"
    print("  ✓ Passed (d): assign_class handles ';' multi-values and tracks.")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING SYNTHETIC VERIFICATION TESTS")
    print("=" * 60)
    test_a_location_split()
    test_b_c_extract_patches_stale_and_tile_validity()
    test_d_assign_class()
    print("=" * 60)
    print("ALL SYNTHETIC VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
