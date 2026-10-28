import os
from pathlib import Path
import numpy as np
import rasterio
import pandas as pd

# ============================================================
PROCESSED_ROOT = Path("processed")
OUTPUT_CSV = Path("processed/quality_report.csv")
# ============================================================

def calculate_valid_overlap(opt_path, sar_path):
    """Calculate percentage of valid overlapping pixels"""
    with rasterio.open(opt_path) as opt, rasterio.open(sar_path) as sar:
        # Basic checks
        if opt.shape != sar.shape:
            return {
                "shape_match": False,
                "valid_pct": 0.0,
                "opt_valid_pct": 0.0,
                "sar_valid_pct": 0.0
            }

        opt_data = opt.read(1)
        sar_data = sar.read(1)

        # Optical valid (assuming 0 is nodata for optical)
        opt_valid = opt_data > 0

        # SAR valid (nodata = -9999)
        sar_valid = sar_data > -9990

        # Common valid area
        common_valid = opt_valid & sar_valid

        total_pixels = opt_data.size
        valid_pct = 100.0 * common_valid.sum() / total_pixels
        opt_valid_pct = 100.0 * opt_valid.sum() / total_pixels
        sar_valid_pct = 100.0 * sar_valid.sum() / total_pixels

        return {
            "shape_match": True,
            "valid_pct": round(valid_pct, 2),
            "opt_valid_pct": round(opt_valid_pct, 2),
            "sar_valid_pct": round(sar_valid_pct, 2)
        }


def main():
    scene_dirs = sorted([
        d for d in PROCESSED_ROOT.iterdir()
        if d.is_dir() and d.name != "masks"
    ])

    print(f"Found {len(scene_dirs)} scenes\n")
    records = []

    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        print(f"Checking: {scene_id}")

        opt_path = scene_dir / "optical_24m.tif"
        sar1_path = scene_dir / "sar_1_24m.tif"
        sar2_path = scene_dir / "sar_2_24m.tif"

        if not opt_path.exists():
            print("  ✗ optical_24m.tif missing")
            continue

        row = {"scene_id": scene_id}

        # Check SAR 1
        if sar1_path.exists():
            stats1 = calculate_valid_overlap(opt_path, sar1_path)
            row.update({
                "sar1_shape_match": stats1["shape_match"],
                "sar1_valid_pct": stats1["valid_pct"],
                "sar1_opt_valid": stats1["opt_valid_pct"],
                "sar1_sar_valid": stats1["sar_valid_pct"],
            })
            print(f"  SAR 1 valid overlap: {stats1['valid_pct']}%")
        else:
            row.update({
                "sar1_shape_match": False,
                "sar1_valid_pct": 0.0,
                "sar1_opt_valid": 0.0,
                "sar1_sar_valid": 0.0,
            })
            print("  SAR 1 missing")

        # Check SAR 2
        if sar2_path.exists():
            stats2 = calculate_valid_overlap(opt_path, sar2_path)
            row.update({
                "sar2_shape_match": stats2["shape_match"],
                "sar2_valid_pct": stats2["valid_pct"],
                "sar2_opt_valid": stats2["opt_valid_pct"],
                "sar2_sar_valid": stats2["sar_valid_pct"],
            })
            print(f"  SAR 2 valid overlap: {stats2['valid_pct']}%")
        else:
            row.update({
                "sar2_shape_match": False,
                "sar2_valid_pct": 0.0,
                "sar2_opt_valid": 0.0,
                "sar2_sar_valid": 0.0,
            })
            print("  SAR 2 missing")

        records.append(row)

    # Save report
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n=== Quality report saved → {OUTPUT_CSV} ===")

    # Summary
    print("\nSummary:")
    print(df[["scene_id", "sar1_valid_pct", "sar2_valid_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()