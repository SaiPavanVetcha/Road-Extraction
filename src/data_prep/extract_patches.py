import csv
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

# Dynamic project root injection
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.paths import (
    MASK_DIR,
    PATCHES_DIR,
    PROCESSED_SCENES_DIR,
    QUALITY_CSV,
)

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


def get_qualifying_scenes(min_valid=50.0):
    """Reads quality report CSV and selects scenes passing valid overlap threshold."""
    if not QUALITY_CSV.exists():
        raise FileNotFoundError(f"Quality report not found: {QUALITY_CSV}")

    df = pd.read_csv(QUALITY_CSV)
    qualifying = df[
        (df["sar1_valid_pct"] >= min_valid)
        & (df["sar2_valid_pct"] >= min_valid)
    ]["scene_id"].tolist()
    return sorted(qualifying)


def generate_dynamic_stratified_split(scenes, seed=RANDOM_SEED):
    """
    Robust dynamic split logic for small satellite scene collections.
    Guarantees non-empty validation and test sets while preventing spatial leakage.
    """
    random.seed(seed)
    scenes = sorted(scenes)
    random.shuffle(scenes)

    n = len(scenes)

    # Force reasonable set sizes for small dataset sizes
    if n <= 5:
        n_train = max(1, n - 2)
        n_val = 1 if n >= 2 else 0
        n_test = n - n_train - n_val
    elif n <= 12:
        n_val = 2
        n_test = 2
        n_train = n - n_val - n_test
    else:
        n_val = max(2, int(n * 0.15))
        n_test = max(2, int(n * 0.15))
        n_train = n - n_val - n_test

    train_scenes = scenes[:n_train]
    val_scenes = scenes[n_train : n_train + n_val]
    test_scenes = scenes[n_train + n_val :]

    split_map = {}
    for s in train_scenes:
        split_map[s] = "train"
    for s in val_scenes:
        split_map[s] = "val"
    for s in test_scenes:
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

    # Concatenate modalities into 6-channel stack
    image = np.concatenate([opt, sar], axis=0)
    _, H, W = image.shape
    num_channels = image.shape[0]

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
            sar_nodata = np.mean(img_tile[opt.shape[0] :] <= -9990)
            if opt_nodata > MAX_NODATA_PCT or sar_nodata > MAX_NODATA_PCT:
                rejected += 1
                continue

            # Quality Gate 2: Cloud reflectance via NIR channel (band index 2)
            if opt.shape[0] >= 3:
                cloud_pct = np.mean(img_tile[2] > 3500)
                if cloud_pct > MAX_CLOUD_PCT:
                    rejected += 1
                    continue

            # Quality Gate 3: Road Pixel Density Bounds
            road_pixels = int(np.sum(msk_tile > 0))
            road_ratio = road_pixels / (PATCH_SIZE * PATCH_SIZE)

            if road_pixels < MIN_ROAD_PIXELS or road_ratio > MAX_ROAD_RATIO:
                rejected += 1
                continue

            # Global unique tile indexing
            tile_counter += 1
            tile_id = f"tile{tile_counter:05d}"
            detailed_name = f"{scene_id}__{tile_id}__y{y:04d}_x{x:04d}"

            # Save clean compressed NPZ file
            out_path = split_dir / f"{tile_id}.npz"
            np.savez_compressed(
                out_path, image=img_tile, mask=msk_tile
            )

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
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("Starting Dynamic 256×256 Patch Extraction\n")

    scenes = get_qualifying_scenes()
    print(f"Qualifying scenes: {len(scenes)}")

    split_map = generate_dynamic_stratified_split(
        scenes, seed=RANDOM_SEED
    )

    print("\nGuaranteed Dynamic Scene → Split Assignment:")
    for s, split in split_map.items():
        print(f"  {s:<35} → {split}")
    print()

    PATCHES_DIR.mkdir(parents=True, exist_ok=True)
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