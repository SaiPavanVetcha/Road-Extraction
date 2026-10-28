import argparse
import re
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject

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

from configs.paths import PROCESSED_SCENES_DIR, RAW_OPTICAL_DIR, RAW_SAR_DIR
from configs.settings import SAR_CALIBRATION_NOTE, TARGET_RES

# ============================================================
# CONFIGURATION
# ============================================================

OPTICAL_ROOT = RAW_OPTICAL_DIR
SAR_ROOT = RAW_SAR_DIR
PROCESSED_ROOT = PROCESSED_SCENES_DIR
PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)


def parse_safe_timestamp(safe_folder):
    """
    Parses the 15-character acquisition start timestamp (e.g. 20251115T123456)
    from the SAFE folder name for chronological ordering.
    """
    match = re.search(r"(\d{8}T\d{6})", safe_folder.name)
    if match:
        return match.group(1)
    return safe_folder.name


def find_all_safe_folders(scene_id):
    """Return all .SAFE folders for a scene sorted chronologically by acquisition timestamp."""
    sar_dir = SAR_ROOT / scene_id
    if not sar_dir.exists():
        return []
    safe_folders = list(sar_dir.glob("*.SAFE"))
    if not safe_folders:
        safe_folders = [
            f for f in sar_dir.iterdir() if f.is_dir() and "S1" in f.name
        ]
    return sorted(safe_folders, key=parse_safe_timestamp)


def process_optical(scene_id, output_dir):
    """Reproject 4 optical bands straight to optical_24m.tif without intermediate file."""
    print(f"\n=== Optical: {scene_id} ===")
    optical_dir = OPTICAL_ROOT / scene_id
    band_files = sorted(list(optical_dir.glob("BAND*.tif")))[:4]

    if len(band_files) < 4:
        raise FileNotFoundError(f"Less than 4 bands found in {optical_dir}")

    opt_24 = output_dir / "optical_24m.tif"

    with rasterio.open(band_files[0]) as ref_src:
        src_crs = ref_src.crs
        src_bounds = ref_src.bounds
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs,
            src_crs,
            ref_src.width,
            ref_src.height,
            *src_bounds,
            resolution=TARGET_RES,
        )
        dst_meta = ref_src.meta.copy()
        dst_meta.update(
            {
                "count": 4,
                "dtype": "float32",
                "transform": dst_transform,
                "width": dst_width,
                "height": dst_height,
            }
        )

    with rasterio.open(opt_24, "w", **dst_meta) as dst:
        for i, bf in enumerate(band_files, start=1):
            with rasterio.open(bf) as src:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=src_crs,
                    resampling=Resampling.bilinear,
                )

    with rasterio.open(opt_24) as src:
        return opt_24, src.crs, src.bounds


def process_one_sar(safe_folder, opt_bounds, opt_crs, out_path):
    """Process a single SAFE product onto the optical grid using Resampling.average."""
    measurement = safe_folder / "measurement"
    if not measurement.is_dir():
        raise FileNotFoundError(
            f"Missing 'measurement' directory in SAFE folder: {safe_folder.name}"
        )

    vv_files = sorted(measurement.glob("*vv*.tiff"))
    vh_files = sorted(measurement.glob("*vh*.tiff"))

    if not vv_files or not vh_files:
        missing = []
        if not vv_files:
            missing.append("VV")
        if not vh_files:
            missing.append("VH")
        raise FileNotFoundError(
            f"Missing {', '.join(missing)} measurement TIFF in SAFE folder: {safe_folder.name}"
        )

    vv_path = vv_files[0]
    vh_path = vh_files[0]

    left, bottom, right, top = opt_bounds
    width = int(round((right - left) / TARGET_RES))
    height = int(round((top - bottom) / TARGET_RES))
    dst_transform = rasterio.transform.from_bounds(
        left, bottom, right, top, width, height
    )

    out_bands = []
    for pol_path in [vv_path, vh_path]:
        with rasterio.open(pol_path) as src:
            gcps, gcp_crs = src.gcps
            dst_arr = np.full((height, width), -9999, dtype=np.float32)

            reproject(
                source=rasterio.band(src, 1),
                destination=dst_arr,
                src_crs=gcp_crs,
                src_gcps=gcps,
                dst_transform=dst_transform,
                dst_crs=opt_crs,
                resampling=Resampling.average,
                src_nodata=0,
                dst_nodata=-9999,
            )

            valid = dst_arr > 0
            # Convert amplitude DN to intensity (DN**2), then to 10*log10 (uncalibrated backscatter (dB-like))
            db_arr = np.full_like(dst_arr, -9999, dtype=np.float32)
            intensity = dst_arr[valid] ** 2
            db_arr[valid] = 10.0 * np.log10(np.maximum(intensity, 1e-10))
            out_bands.append(db_arr)

    meta = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 2,
        "dtype": "float32",
        "crs": opt_crs,
        "transform": dst_transform,
        "nodata": -9999,
    }

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(out_bands[0], 1)  # VV
        dst.write(out_bands[1], 2)  # VH

    for pol_name, band_arr in [("VV", out_bands[0]), ("VH", out_bands[1])]:
        valid_vals = band_arr[band_arr > -9990]
        if valid_vals.size > 0:
            p5, p50, p95 = np.percentile(valid_vals, (5, 50, 95))
            print(
                f"    {pol_name} [{SAR_CALIBRATION_NOTE}]: "
                f"min={valid_vals.min():.1f}, p5={p5:.1f}, median={p50:.1f}, "
                f"p95={p95:.1f}, max={valid_vals.max():.1f} dB"
            )


def process_one_scene(scene_id, force=False):
    print("\n" + "=" * 70)
    print(f"Processing: {scene_id}")
    print("=" * 70)

    output_dir = PROCESSED_ROOT / scene_id
    output_dir.mkdir(parents=True, exist_ok=True)

    opt_24 = output_dir / "optical_24m.tif"
    safe_folders = find_all_safe_folders(scene_id)
    expected_sar_files = [
        output_dir / f"sar_{idx}_24m.tif"
        for idx in range(1, len(safe_folders) + 1)
    ]

    if (
        not force
        and opt_24.exists()
        and safe_folders
        and all(f.exists() for f in expected_sar_files)
    ):
        print(
            f"✓ Skipping {scene_id}: optical_24m.tif and all {len(safe_folders)} SAR rasters exist (use --force to overwrite)"
        )
        return True

    try:
        # 1. Optical
        opt_24, opt_crs, opt_bounds = process_optical(scene_id, output_dir)

        # 2. All SARs
        print(f"Found {len(safe_folders)} SAR products")
        if not safe_folders:
            raise RuntimeError(f"No SAR folders found for scene {scene_id}")

        processed_sar_count = 0
        for idx, safe_folder in enumerate(safe_folders, start=1):
            try:
                print(f"  → Processing SAR {idx}: {safe_folder.name}")
                out_sar = output_dir / f"sar_{idx}_24m.tif"
                process_one_sar(safe_folder, opt_bounds, opt_crs, out_sar)
                print(f"    Saved → {out_sar.name}")
                processed_sar_count += 1
            except Exception as e:
                print(f"    ⚠️ Skipping {safe_folder.name}: {e}")

        if processed_sar_count == 0:
            raise RuntimeError(
                f"No usable SAR products could be processed for scene {scene_id}"
            )

        print(
            f"✓ {scene_id} completed successfully ({processed_sar_count} SAR products)"
        )
        return True

    except Exception as e:
        print(f"✗ {scene_id} FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess optical and SAR scenes to 24m grid."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if output files exist.",
    )
    args = parser.parse_args()

    scene_ids = (
        sorted([f.name for f in OPTICAL_ROOT.iterdir() if f.is_dir()])
        if OPTICAL_ROOT.exists()
        else []
    )
    print(f"Starting batch processing of {len(scene_ids)} scenes...\n")

    success = 0
    failed = []

    for scene_id in scene_ids:
        if process_one_scene(scene_id, force=args.force):
            success += 1
        else:
            failed.append(scene_id)

    print("\n" + "=" * 70)
    print("BATCH PROCESSING FINISHED")
    print(f"Successful : {success}/{len(scene_ids)}")
    if failed:
        print("Failed scenes:")
        for f in failed:
            print(f"  - {f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
