import sys
from pathlib import Path
import numpy as np
import pandas as pd
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

from configs.paths import PROCESSED_SCENES_DIR, MASK_DIR, QUALITY_CSV
from configs.settings import MIN_VALID_PCT, SAR_MIN_DB, SAR_MAX_DB
from src.utils.scene_selection import get_qualifying_scenes

# ============================================================
PROCESSED_ROOT = PROCESSED_SCENES_DIR
MASK_ROOT = MASK_DIR
# ============================================================


def verify_scene(scene_id):
    scene_dir = PROCESSED_ROOT / scene_id
    opt_path = scene_dir / "optical_24m.tif"
    sar_paths = sorted(list(scene_dir.glob("sar_*_24m.tif")))
    mask_path = MASK_ROOT / f"mask_{scene_id}.tif"

    errors = []
    warnings = []

    # 1. File Existence Check
    if not opt_path.exists():
        errors.append("Missing optical_24m.tif")
    if not mask_path.exists():
        errors.append(f"Missing mask_{scene_id}.tif")
    if not sar_paths:
        errors.append("No SAR rasters found")

    if errors:
        return {"scene_id": scene_id, "status": "FAILED", "issues": "; ".join(errors)}

    # 2. Geometry, Shape, Transform & Band Alignment Check
    with rasterio.open(opt_path) as opt_src:
        opt_shape = (opt_src.height, opt_src.width)
        opt_crs = opt_src.crs
        opt_transform = opt_src.transform
        opt_bounds = opt_src.bounds
        opt_bands = opt_src.count

    if opt_bands != 4:
        errors.append(f"Expected 4 optical bands, found {opt_bands}")

    with rasterio.open(mask_path) as mask_src:
        mask_shape = (mask_src.height, mask_src.width)
        mask_crs = mask_src.crs
        mask_transform = mask_src.transform
        mask_bounds = mask_src.bounds

    if opt_shape != mask_shape:
        errors.append(f"Shape Mismatch: Optical {opt_shape} vs Mask {mask_shape}")
    if opt_crs != mask_crs:
        errors.append(f"CRS Mismatch: Optical {opt_crs} vs Mask {mask_crs}")
    if not np.allclose(list(opt_transform)[:6], list(mask_transform)[:6], atol=1e-3):
        errors.append("Transform Mismatch: Optical vs Mask")
    if not np.allclose(list(opt_bounds), list(mask_bounds), atol=1e-3):
        errors.append("Bounds Mismatch: Optical vs Mask")

    # 3. SAR Signal Dynamic Range, Shape, Bounds & Transform Check
    sar_shapes_ok = True
    sar_valid_db = True
    sar_channels = 0

    for sar_p in sar_paths:
        with rasterio.open(sar_p) as sar_src:
            if (sar_src.height, sar_src.width) != opt_shape:
                sar_shapes_ok = False
            if not np.allclose(list(sar_src.transform)[:6], list(opt_transform)[:6], atol=1e-3):
                errors.append(f"Transform Mismatch: Optical vs SAR ({sar_p.name})")
            if not np.allclose(list(sar_src.bounds), list(opt_bounds), atol=1e-3):
                errors.append(f"Bounds Mismatch: Optical vs SAR ({sar_p.name})")
            if sar_src.crs != opt_crs:
                errors.append(f"CRS Mismatch: Optical vs SAR ({sar_p.name})")

            sar_arr = sar_src.read(1)
            valid_vals = sar_arr[sar_arr > -9990]

            if valid_vals.size > 0:
                min_db, max_db = float(np.min(valid_vals)), float(np.max(valid_vals))
                if min_db < SAR_MIN_DB or max_db > SAR_MAX_DB:
                    sar_valid_db = False
            sar_channels += sar_src.count

    if not sar_shapes_ok:
        errors.append("SAR raster shape does not match Optical grid")
    if not sar_valid_db:
        warnings.append(
            f"SAR dB values outside typical range [{SAR_MIN_DB:.0f} dB, {SAR_MAX_DB:.0f} dB]"
        )

    # 4. Mask Class Presence Check
    with rasterio.open(mask_path) as mask_src:
        mask_arr = mask_src.read(1)
        unique_classes = np.unique(mask_arr)
        total_road_pixels = int(np.sum(mask_arr > 0))

    if total_road_pixels < 1000:
        warnings.append(f"Low road density: only {total_road_pixels:,} road pixels")

    status = "READY" if not errors else "FAILED"
    issues_str = "; ".join(errors + warnings) if (errors or warnings) else "None"

    return {
        "scene_id": scene_id,
        "status": status,
        "optical_bands": opt_bands,
        "sar_files": len(sar_paths),
        "total_pixels": f"{opt_shape[0]}x{opt_shape[1]}",
        "classes_found": list(unique_classes),
        "total_road_px": total_road_pixels,
        "issues": issues_str,
    }


def main():
    if not QUALITY_CSV.exists():
        print(f"Error: {QUALITY_CSV} not found.")
        return

    try:
        selected_scenes = get_qualifying_scenes(min_valid=MIN_VALID_PCT, min_sar=1)
    except Exception as e:
        print(f"Error reading qualifying scenes: {e}")
        return

    if not selected_scenes:
        print(f"No qualifying scenes found with >= {MIN_VALID_PCT}% valid overlap.")
        return

    print(f"Auditing Dataset Readiness across {len(selected_scenes)} scenes...\n")

    results = []
    for s in selected_scenes:
        res = verify_scene(s)
        results.append(res)

    df_res = pd.DataFrame(results)

    print("=" * 80)
    print("DATASET PREPARATION READINESS REPORT")
    print("=" * 80)
    for _, row in df_res.iterrows():
        try:
            icon = "✓" if row["status"] == "READY" else "❌"
            print(f"{icon} Scene: {row['scene_id']}")
        except UnicodeEncodeError:
            icon = "[READY]" if row["status"] == "READY" else "[FAILED]"
            print(f"{icon} Scene: {row['scene_id']}")
        print(
            f"   Shape: {row['total_pixels']} | Optical Bands: {row['optical_bands']} | SAR Rasters: {row['sar_files']}"
        )
        print(
            f"   Classes Present: {row['classes_found']} | Total Road Pixels: {row['total_road_px']:,}"
        )
        print(f"   Issues: {row['issues']}\n")

    ready_count = len(df_res[df_res["status"] == "READY"])
    print("=" * 80)
    print(
        f"Summary: {ready_count}/{len(selected_scenes)} Scenes are READY for 256x256 Patch Tiling"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
