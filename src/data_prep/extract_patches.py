import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio

# Handle Windows console encoding for Unicode checkmarks and symbols
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.paths import (
    MASK_DIR,
    PATCHES_DIR,
    PROCESSED_SCENES_DIR,
)
from configs.settings import MIN_VALID_PCT, REQUIRED_SAR_PRODUCTS
from src.utils.scene_selection import get_qualifying_scenes

# ============================================================
# CONFIGURATION
# ============================================================
PATCH_SIZE = 256
STRIDE = 128  # 50% overlap stride

# Quality Gate Thresholds
MAX_NODATA_PCT = 0.10  # Max 10% nodata allowed per tile
MAX_CLOUD_PCT = 0.18  # Max 18% cloud cover allowed
MIN_ROAD_PIXELS = 80  # Min 80 road pixels to keep a tile
MAX_ROAD_RATIO = 0.65  # Max 65% road ratio to prevent junction dominance

RANDOM_SEED = 42
# ============================================================


def generate_location_split(scenes, seed=RANDOM_SEED):
    """
    Splits scenes by unique geographic location to prevent spatial leakage across train/val/test.
    Unique locations are sorted, shuffled with random.Random(seed), and partitioned so that all
    scenes from a given location belong to the same split.
    """
    if not scenes:
        return {}

    unique_locations = sorted(
        list(set(s.split("_", 1)[1] if "_" in s else s for s in scenes))
    )
    rng = random.Random(seed)
    shuffled_locations = unique_locations.copy()
    rng.shuffle(shuffled_locations)

    n = len(shuffled_locations)

    if n == 1:
        print(
            "Notice: Only 1 unique location found. All scenes assigned to train (val and test are empty)."
        )
        train_locs = set(shuffled_locations)
        val_locs = set()
        test_locs = set()
    elif n == 2:
        print(
            "Notice: Only 2 unique locations found. Assigned 1 to train, 1 to val (test is empty)."
        )
        train_locs = {shuffled_locations[0]}
        val_locs = {shuffled_locations[1]}
        test_locs = set()
    else:
        if n <= 5:
            n_val = 1
            n_test = 1
            n_train = n - 2
        elif n <= 12:
            n_val = 2
            n_test = 2
            n_train = n - 4
        else:
            n_val = max(2, int(n * 0.15))
            n_test = max(2, int(n * 0.15))
            n_train = n - n_val - n_test

        train_locs = set(shuffled_locations[:n_train])
        val_locs = set(shuffled_locations[n_train : n_train + n_val])
        test_locs = set(shuffled_locations[n_train + n_val :])

    split_map = {}
    for s in scenes:
        loc = s.split("_", 1)[1] if "_" in s else s
        if loc in train_locs:
            split_map[s] = "train"
        elif loc in val_locs:
            split_map[s] = "val"
        else:
            split_map[s] = "test"

    return split_map


def process_scene(scene_id, split_name, metadata_writer, tile_counter):
    print(f"=== [{split_name.upper()}] {scene_id} ===")

    scene_dir = PROCESSED_SCENES_DIR / scene_id
    opt_path = scene_dir / "optical_24m.tif"
    mask_path = MASK_DIR / f"mask_{scene_id}.tif"
    sar_paths = sorted(scene_dir.glob("sar_*_24m.tif"))

    if not opt_path.exists() or not mask_path.exists() or not sar_paths:
        print("  Missing required raster files – skipped")
        return 0, 0, tile_counter

    # Read Optical Channels
    with rasterio.open(opt_path) as src:
        opt = src.read().astype(np.float32)
        opt_crs = str(src.crs)
        opt_bounds = src.bounds

    valid_opt_pixels = opt[0] > 0
    if np.any(valid_opt_pixels):
        scene_max = float(np.percentile(opt[2][valid_opt_pixels], 99.9))
    else:
        scene_max = 1.0
    print(f"  Detected scene optical max (band 2 99.9th percentile): {scene_max:.1f}")

    # Read SAR Channels (All available polarizations)
    sar_bands = []
    for sp in sar_paths:
        with rasterio.open(sp) as src:
            for b in range(1, src.count + 1):
                sar_bands.append(src.read(b).astype(np.float32))
    sar = np.stack(sar_bands, axis=0)

    # Read Multi-Class Target Mask
    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype(np.uint8)

    # Concatenate modalities into optical + SAR stack
    num_opt = opt.shape[0]
    image = np.concatenate([opt, sar], axis=0)
    _, H, W = image.shape

    split_dir = PATCHES_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    rejected = 0

    # Sliding Window Extraction
    for y in range(0, H - PATCH_SIZE + 1, STRIDE):
        for x in range(0, W - PATCH_SIZE + 1, STRIDE):
            img_tile = image[:, y : y + PATCH_SIZE, x : x + PATCH_SIZE]
            msk_tile = mask[y : y + PATCH_SIZE, x : x + PATCH_SIZE]

            # Quality Gate 1: Check Nodata (Optical channel 0 and across all SAR channels)
            opt_nodata = np.mean(img_tile[0] <= 0)
            sar_nodata = np.mean(img_tile[num_opt:] <= -9990)
            if opt_nodata > MAX_NODATA_PCT or sar_nodata > MAX_NODATA_PCT:
                rejected += 1
                continue

            # Quality Gate 2: Relative cloud reflectance gate
            vis_brightness = 0.5 * (img_tile[0] + img_tile[1])
            nir_swir_contrast = np.abs(img_tile[2] - img_tile[3])
            cloud_mask = (vis_brightness > 0.8 * scene_max) & (
                nir_swir_contrast < 0.1 * scene_max
            )
            cloud_pct = float(np.mean(cloud_mask))
            if cloud_pct > MAX_CLOUD_PCT:
                rejected += 1
                continue

            # Quality Gate 3: Road Pixel Density Bounds
            road_pixels = int(np.sum(msk_tile > 0))
            road_ratio = road_pixels / (PATCH_SIZE * PATCH_SIZE)

            if road_pixels < MIN_ROAD_PIXELS or road_ratio > MAX_ROAD_RATIO:
                rejected += 1
                continue

            # Compute validity channel: 1.0 where all SAR channels are valid and optical band 0 > 0
            sar_valid = np.all(
                (img_tile[num_opt:] > -9990) & np.isfinite(img_tile[num_opt:]),
                axis=0,
            )
            opt_valid = (img_tile[0] > 0) & np.isfinite(img_tile[0])
            validity = np.where(sar_valid & opt_valid, 1.0, 0.0).astype(
                np.float32
            )

            # Replace SAR -9999 sentinels and any NaN with 0.0
            img_tile = np.where(
                (img_tile <= -9990) | np.isnan(img_tile), 0.0, img_tile
            ).astype(np.float32)

            # Append validity mask as final channel: (4 optical + 2 per SAR + 1 validity)
            img_tile = np.concatenate(
                [img_tile, validity[np.newaxis, :, :]], axis=0
            )
            num_channels = img_tile.shape[0]

            # Global unique tile indexing
            tile_counter += 1
            tile_id = f"tile{tile_counter:05d}"
            detailed_name = f"{scene_id}__{tile_id}__y{y:04d}_x{x:04d}"

            # Save clean compressed NPZ file
            out_path = split_dir / f"{tile_id}.npz"
            np.savez_compressed(out_path, image=img_tile, mask=msk_tile)

            # Record spatial and pixel metadata
            metadata_writer.writerow(
                {
                    "tile_id": tile_id,
                    "detailed_name": detailed_name,
                    "scene_id": scene_id,
                    "split": split_name,
                    "y": y,
                    "x": x,
                    "road_pixels": road_pixels,
                    "road_ratio": round(road_ratio, 4),
                    "num_channels": num_channels,
                    "crs": opt_crs,
                    "scene_left": opt_bounds.left,
                    "scene_bottom": opt_bounds.bottom,
                    "scene_right": opt_bounds.right,
                    "scene_top": opt_bounds.top,
                }
            )

            kept += 1

    print(f"  Kept: {kept:,} | Rejected: {rejected:,}\n")
    return kept, rejected, tile_counter


def main():
    parser = argparse.ArgumentParser(
        description="Extract 256x256 multi-modal patches from scenes."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for location split (default: {RANDOM_SEED})",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Starting Dynamic 256×256 Patch Extraction\n")

    # Clean stale tiles and ensure fresh patches directory
    if PATCHES_DIR.exists():
        shutil.rmtree(PATCHES_DIR)
    PATCHES_DIR.mkdir(parents=True, exist_ok=True)

    scenes = get_qualifying_scenes(
        min_valid=MIN_VALID_PCT, min_sar=REQUIRED_SAR_PRODUCTS
    )
    print(f"Qualifying scenes: {len(scenes)}")

    split_map = generate_location_split(scenes, seed=args.seed)

    print("\nGuaranteed Dynamic Location → Split Assignment:")
    for s, split in split_map.items():
        print(f"  {s:<35} → {split}")
    print()

    metadata_path = PATCHES_DIR / "tiles_metadata.csv"

    fieldnames = [
        "tile_id",
        "detailed_name",
        "scene_id",
        "split",
        "y",
        "x",
        "road_pixels",
        "road_ratio",
        "num_channels",
        "crs",
        "scene_left",
        "scene_bottom",
        "scene_right",
        "scene_top",
    ]

    total_kept = total_rej = 0
    stats = {"train": 0, "val": 0, "test": 0}
    tile_counter = 0

    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for scene_id in scenes:
            split_name = split_map.get(scene_id, "train")
            kept, rej, tile_counter = process_scene(
                scene_id, split_name, writer, tile_counter
            )
            stats[split_name] += kept
            total_kept += kept
            total_rej += rej

    print("=" * 65)
    print("PATCH EXTRACTION COMPLETE")
    print(f"Total patches kept   : {total_kept:,}")
    print(f"Total tiles rejected : {total_rej:,}")
    print("Split distribution:")
    for k, v in stats.items():
        print(f"  {k.capitalize():<6}: {v:,}")
    print(f"\nMetadata saved → {metadata_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
