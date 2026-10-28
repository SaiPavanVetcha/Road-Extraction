from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

# ============================================================
PROCESSED_ROOT = Path("processed")
MASK_ROOT = Path("processed/masks")
QUALITY_CSV = Path("processed/quality_report.csv")
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

    # 2. Geometry & Shape Alignment Check
    with rasterio.open(opt_path) as opt_src:
        opt_shape = (opt_src.height, opt_src.width)
        opt_crs = opt_src.crs
        opt_bands = opt_src.count

    with rasterio.open(mask_path) as mask_src:
        mask_shape = (mask_src.height, mask_src.width)
        mask_crs = mask_src.crs

    if opt_shape != mask_shape:
        errors.append(f"Shape Mismatch: Optical {opt_shape} vs Mask {mask_shape}")
    if opt_crs != mask_crs:
        errors.append(f"CRS Mismatch: Optical {opt_crs} vs Mask {mask_crs}")

    # 3. SAR Signal Dynamic Range Check
    sar_shapes_ok = True
    sar_valid_db = True
    sar_channels = 0

    for sar_p in sar_paths:
        with rasterio.open(sar_p) as sar_src:
            if (sar_src.height, sar_src.width) != opt_shape:
                sar_shapes_ok = False
            sar_arr = sar_src.read(1)
            valid_vals = sar_arr[sar_arr > -9990]

            if valid_vals.size > 0:
                min_db, max_db = np.min(valid_vals), np.max(valid_vals)
                # Typical SAR Sentinel-1 GRD dB bounds: [-35 dB, +5 dB]
                if min_db < -50 or max_db > 15:
                    sar_valid_db = False
            sar_channels += sar_src.count

    if not sar_shapes_ok:
        errors.append("SAR raster shape does not match Optical grid")
    if not sar_valid_db:
        warnings.append("SAR dB values outside typical range [-35 dB, +5 dB]")

    # 4. Mask Class Presence Check
    with rasterio.open(mask_path) as mask_src:
        mask_arr = mask_src.read(1)
        unique_classes = np.unique(mask_arr)
        total_road_pixels = np.sum(mask_arr > 0)

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

    df_qc = pd.read_csv(QUALITY_CSV)
    selected_scenes = df_qc[
        (df_qc["sar1_valid_pct"] >= 50.0) & (df_qc["sar2_valid_pct"] >= 50.0)
    ]["scene_id"].tolist()

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
        icon = "✓" if row["status"] == "READY" else "❌"
        print(f"{icon} Scene: {row['scene_id']}")
        print(f"   Shape: {row['total_pixels']} | Optical Bands: {row['optical_bands']} | SAR Rasters: {row['sar_files']}")
        print(f"   Classes Present: {row['classes_found']} | Total Road Pixels: {row['total_road_px']:,}")
        print(f"   Issues: {row['issues']}\n")

    ready_count = len(df_res[df_res["status"] == "READY"])
    print("=" * 80)
    print(f"Summary: {ready_count}/{len(selected_scenes)} Scenes are 100% READY for 256x256 Patch Tiling")
    print("=" * 80)


if __name__ == "__main__":
    main()