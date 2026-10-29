import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio

# Windows UTF-8 console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.paths import MASK_DIR, PATCHES_DIR, PROCESSED_SCENES_DIR
from configs.settings import MIN_VALID_PCT, REQUIRED_SAR_PRODUCTS, CHANNELS_PER_TILE
from src.utils.scene_selection import get_qualifying_scenes

# ============================================================
PATCH_SIZE = 256
STRIDE = 128

MAX_NODATA_PCT = 0.10
MAX_CLOUD_PCT = 0.18
MIN_ROAD_PIXELS = 80
MAX_ROAD_RATIO = 0.65

RANDOM_SEED = 42
# ============================================================


def generate_location_split(scenes, seed=RANDOM_SEED):
    """
    Split by unique geographic location to prevent spatial leakage.
    Forces Train to be the largest split when possible.
    """
    rng = random.Random(seed)

    def location_of(scene_id):
        return scene_id.split("_", 1)[1] if "_" in scene_id else scene_id

    unique_locations = sorted({location_of(s) for s in scenes})
    rng.shuffle(unique_locations)
    n = len(unique_locations)

    if n == 1:
        print("Notice: Only 1 unique location → all scenes assigned to train.")
        train_locs, val_locs, test_locs = set(unique_locations), set(), set()
    elif n == 2:
        print("Notice: Only 2 unique locations → train + val (test empty).")
        train_locs = {unique_locations[0]}
        val_locs = {unique_locations[1]}
        test_locs = set()
    else:
        # Aim for ~70% train, 15% val, 15% test by location count
        n_val = max(1, round(n * 0.15))
        n_test = max(1, round(n * 0.15))
        n_train = n - n_val - n_test
        if n_train < 1:
            n_train = 1
            n_val = max(1, (n - 1) // 2)
            n_test = n - n_train - n_val

        train_locs = set(unique_locations[:n_train])
        val_locs = set(unique_locations[n_train : n_train + n_val])
        test_locs = set(unique_locations[n_train + n_val :])

    split_map = {}
    for s in scenes:
        loc = location_of(s)
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
        print("  Missing required files – skipped")
        return 0, 0, tile_counter

    # Read Optical
    with rasterio.open(opt_path) as src:
        opt = src.read().astype(np.float32)  # (4, H, W)
        opt_crs = str(src.crs)
        opt_bounds = src.bounds

    # Read Mask
    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype(np.uint8)

    _, H, W = opt.shape

    # Pre-compute scene optical max for relative cloud gate
    valid_opt = opt[2][opt[2] > 0]
    scene_max = float(np.percentile(valid_opt, 99.9)) if valid_opt.size > 100 else 4000.0
    print(f"  Detected scene optical max (band 2 99.9th percentile): {scene_max:.1f}")

    split_dir = PATCHES_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    rejected = 0

    for y in range(0, H - PATCH_SIZE + 1, STRIDE):
        for x in range(0, W - PATCH_SIZE + 1, STRIDE):
            opt_tile = opt[:, y : y + PATCH_SIZE, x : x + PATCH_SIZE]
            msk_tile = mask[y : y + PATCH_SIZE, x : x + PATCH_SIZE]

            # Optical nodata gate
            opt_nodata = np.mean(opt_tile[0] <= 0)
            if opt_nodata > MAX_NODATA_PCT:
                rejected += 1
                continue

            # Relative cloud gate
            vis_brightness = (opt_tile[0] + opt_tile[1]) / 2.0
            nir_swir_diff = np.abs(opt_tile[2] - opt_tile[3])
            cloud_mask = (vis_brightness > 0.8 * scene_max) & (nir_swir_diff < 0.1 * scene_max)
            if float(np.mean(cloud_mask)) > MAX_CLOUD_PCT:
                rejected += 1
                continue

            # Road density gate
            road_pixels = int(np.sum(msk_tile > 0))
            road_ratio = road_pixels / (PATCH_SIZE * PATCH_SIZE)
            if road_pixels < MIN_ROAD_PIXELS or road_ratio > MAX_ROAD_RATIO:
                rejected += 1
                continue

            # Create one tile per SAR product
            for sar_path in sar_paths:
                sar_name = sar_path.stem  # e.g. sar_1_24m
                sar_product = sar_name.replace("_24m", "")  # sar_1 or sar_2

                with rasterio.open(sar_path) as src:
                    sar = src.read().astype(np.float32)  # (2, H, W)

                sar_tile = sar[:, y : y + PATCH_SIZE, x : x + PATCH_SIZE]

                # SAR nodata gate
                sar_nodata = np.mean(sar_tile <= -9990)
                if sar_nodata > MAX_NODATA_PCT:
                    rejected += 1
                    continue

                # Validity channel
                opt_valid = opt_tile[0] > 0
                sar_valid = (sar_tile[0] > -9990) & (sar_tile[1] > -9990)
                validity = np.where(opt_valid & sar_valid, 1.0, 0.0).astype(np.float32)

                # Clean SAR values
                sar_clean = sar_tile.copy()
                sar_clean[sar_clean <= -9990] = 0.0
                sar_clean = np.nan_to_num(sar_clean, nan=0.0)

                # Stack: 4 optical + 2 SAR + 1 validity = 7 channels
                img_tile = np.concatenate(
                    [opt_tile, sar_clean, validity[np.newaxis, :, :]], axis=0
                )

                assert img_tile.shape[0] == CHANNELS_PER_TILE, (
                    f"Expected {CHANNELS_PER_TILE} channels, got {img_tile.shape[0]}"
                )

                # Global tile ID
                tile_counter += 1
                tile_id = f"tile{tile_counter:05d}"

                out_path = split_dir / f"{tile_id}.npz"
                np.savez_compressed(out_path, image=img_tile, mask=msk_tile)

                metadata_writer.writerow({
                    "tile_id": tile_id,
                    "scene_id": scene_id,
                    "sar_product": sar_product,
                    "sar_filename": sar_path.name,
                    "split": split_name,
                    "y": y,
                    "x": x,
                    "road_pixels": road_pixels,
                    "road_ratio": round(road_ratio, 4),
                    "num_channels": CHANNELS_PER_TILE,
                    "channels": "B2,B3,B4,B5,VV,VH,validity",
                    "crs": opt_crs,
                    "scene_left": opt_bounds.left,
                    "scene_bottom": opt_bounds.bottom,
                    "scene_right": opt_bounds.right,
                    "scene_top": opt_bounds.top,
                })

                kept += 1

    print(f"  Kept: {kept:,} | Rejected: {rejected:,}\n")
    return kept, rejected, tile_counter


def main():
    parser = argparse.ArgumentParser(description="Extract 256x256 multi-modal patches")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for location split")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Starting 256×256 Patch Extraction (Optical × each SAR)\n")

    # Clean previous patches
    if PATCHES_DIR.exists():
        shutil.rmtree(PATCHES_DIR)
    PATCHES_DIR.mkdir(parents=True, exist_ok=True)

    scenes = get_qualifying_scenes(min_valid=MIN_VALID_PCT, min_sar=REQUIRED_SAR_PRODUCTS)
    print(f"Qualifying scenes: {len(scenes)}")

    split_map = generate_location_split(scenes, seed=args.seed)

    print("\nLocation → Split Assignment:")
    for s, split in split_map.items():
        print(f"  {s:<35} → {split}")
    print()

    metadata_path = PATCHES_DIR / "tiles_metadata.csv"
    fieldnames = [
        "tile_id", "scene_id", "sar_product", "sar_filename", "split",
        "y", "x", "road_pixels", "road_ratio", "num_channels", "channels",
        "crs", "scene_left", "scene_bottom", "scene_right", "scene_top",
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