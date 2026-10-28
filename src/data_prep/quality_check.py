import sys
from pathlib import Path
import pandas as pd
import rasterio

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

from configs.paths import PROCESSED_SCENES_DIR, QUALITY_CSV

# ============================================================
PROCESSED_ROOT = PROCESSED_SCENES_DIR
OUTPUT_CSV = QUALITY_CSV
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
# ============================================================


def calculate_valid_overlap(opt_path, sar_path):
    """Calculate percentage of valid overlapping pixels across all optical and SAR bands."""
    with rasterio.open(opt_path) as opt, rasterio.open(sar_path) as sar:
        # Basic checks
        if opt.shape != sar.shape:
            return {
                "shape_match": False,
                "valid_pct": 0.0,
                "opt_valid_pct": 0.0,
                "sar_valid_pct": 0.0,
            }

        opt_data = opt.read()  # (count, H, W)
        sar_data = sar.read()  # (count, H, W)

        # Optical valid: all 4 bands > 0
        opt_valid = (opt_data > 0).all(axis=0)

        # SAR valid: both VV and VH > -9990
        sar_valid = (sar_data > -9990).all(axis=0)

        # Common valid area
        common_valid = opt_valid & sar_valid

        total_pixels = opt_valid.size
        valid_pct = 100.0 * float(common_valid.sum()) / total_pixels
        opt_valid_pct = 100.0 * float(opt_valid.sum()) / total_pixels
        sar_valid_pct = 100.0 * float(sar_valid.sum()) / total_pixels

        return {
            "shape_match": True,
            "valid_pct": round(valid_pct, 2),
            "opt_valid_pct": round(opt_valid_pct, 2),
            "sar_valid_pct": round(sar_valid_pct, 2),
        }


def main():
    scene_dirs = (
        sorted([d for d in PROCESSED_ROOT.iterdir() if d.is_dir()])
        if PROCESSED_ROOT.exists()
        else []
    )

    print(f"Found {len(scene_dirs)} scenes\n")
    records = []

    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        print(f"Checking: {scene_id}")

        opt_path = scene_dir / "optical_24m.tif"
        sar_paths = sorted(scene_dir.glob("sar_*_24m.tif"))

        if not opt_path.exists():
            print("  ✗ optical_24m.tif missing")
            continue

        row = {"scene_id": scene_id, "n_sar": len(sar_paths)}

        valid_pcts = []
        for idx, sar_path in enumerate(sar_paths, start=1):
            stats = calculate_valid_overlap(opt_path, sar_path)
            row[f"sar{idx}_shape_match"] = stats["shape_match"]
            row[f"sar{idx}_valid_pct"] = stats["valid_pct"]
            row[f"sar{idx}_opt_valid"] = stats["opt_valid_pct"]
            row[f"sar{idx}_sar_valid"] = stats["sar_valid_pct"]
            valid_pcts.append(stats["valid_pct"])
            print(f"  SAR {idx} valid overlap: {stats['valid_pct']}%")

        row["valid_pct_best"] = max(valid_pcts) if valid_pcts else 0.0
        records.append(row)

    if not records:
        print("No scenes found.")
        df = pd.DataFrame(columns=["scene_id", "n_sar", "valid_pct_best"])
        df.to_csv(OUTPUT_CSV, index=False)
        return

    # Save report
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n=== Quality report saved → {OUTPUT_CSV} ===")

    # Summary
    print("\nSummary:")
    summary_cols = ["scene_id", "n_sar", "valid_pct_best"]
    sar_pct_cols = [
        c
        for c in df.columns
        if c.startswith("sar") and c.endswith("_valid_pct")
    ]
    display_cols = [c for c in summary_cols + sar_pct_cols if c in df.columns]
    print(df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
