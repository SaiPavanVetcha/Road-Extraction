import os
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject

# ============================================================
# CONFIGURATION
# ============================================================

OPTICAL_ROOT   = Path("raw/optical")
SAR_ROOT       = Path("raw/sar")
PROCESSED_ROOT = Path("processed")
PROCESSED_ROOT.mkdir(exist_ok=True)

TARGET_RES = 24.0

SCENE_IDS = [
    "2025-11-17_Amaravati",
    "2025-11-25_Amaravati",
    "2025-12-09_Visakhapatnam",
    "2025-12-11_Guntur",
    "2025-12-11_Tirupathi_1",
    "2025-12-11_Tirupathi_2",
    "2025-12-14_Visakhapatnam",
    "2025-12-16_Guntur",
    "2026-01-21_Guntur",
    "2026-02-28_Kurnool_1",
    "2026-02-28_Kurnool_2",
    "2026-03-19_Kurnool_1",
    "2026-03-19_Kurnool_2",
    "2026-04-22_Nellore",
    "2026-05-30_Hyderabad",
    "2026-06-11_Hyderabad",
    "2026-09-15_Anantapur",
    "2026-10-14_Warangal",
]

# ============================================================

def find_all_safe_folders(scene_id):
    """Return all .SAFE folders for a scene"""
    sar_dir = SAR_ROOT / scene_id
    safe_folders = list(sar_dir.glob("*.SAFE"))
    if not safe_folders:
        safe_folders = [f for f in sar_dir.iterdir() if f.is_dir() and "S1" in f.name]
    return sorted(safe_folders)


def process_optical(scene_id, output_dir):
    print(f"\n=== Optical: {scene_id} ===")
    optical_dir = OPTICAL_ROOT / scene_id
    band_files = sorted(list(optical_dir.glob("BAND*.tif")))[:4]

    if len(band_files) < 4:
        raise FileNotFoundError(f"Less than 4 bands found in {optical_dir}")

    with rasterio.open(band_files[0]) as src:
        meta = src.meta.copy()
        arrays = [rasterio.open(bf).read(1).astype(np.float32) for bf in band_files]
        stacked = np.stack(arrays, axis=0)

        meta.update({"count": 4, "dtype": "float32"})
        full_path = output_dir / "optical_full.tif"
        with rasterio.open(full_path, "w", **meta) as dst:
            dst.write(stacked)

    # Resample to exact 24 m
    opt_24 = output_dir / "optical_24m.tif"
    with rasterio.open(full_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, src.crs, src.width, src.height, *src.bounds, resolution=TARGET_RES
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "transform": transform,
            "width": width,
            "height": height,
            "dtype": "float32"
        })
        with rasterio.open(opt_24, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=src.crs,
                    resampling=Resampling.bilinear
                )

    with rasterio.open(opt_24) as src:
        return opt_24, src.crs, src.bounds


def process_one_sar(safe_folder, opt_bounds, opt_crs, out_path):
    """Process a single SAFE product onto the optical grid"""
    measurement = safe_folder / "measurement"
    vv_path = list(measurement.glob("*vv*.tiff"))[0]
    vh_path = list(measurement.glob("*vh*.tiff"))[0]

    left, bottom, right, top = opt_bounds
    width  = int(round((right - left) / TARGET_RES))
    height = int(round((top - bottom) / TARGET_RES))
    dst_transform = rasterio.transform.from_bounds(left, bottom, right, top, width, height)

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
                resampling=Resampling.bilinear,
                src_nodata=0,
                dst_nodata=-9999,
            )

            valid = dst_arr > 0
            db_arr = np.full_like(dst_arr, -9999, dtype=np.float32)
            db_arr[valid] = 10 * np.log10(dst_arr[valid])
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

    valid_vv = out_bands[0][out_bands[0] > -9990]
    if valid_vv.size > 0:
        print(f"    VV range: {valid_vv.min():.1f} → {valid_vv.max():.1f} dB")


def process_one_scene(scene_id):
    print("\n" + "="*70)
    print(f"Processing: {scene_id}")
    print("="*70)

    output_dir = PROCESSED_ROOT / scene_id
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Optical
        opt_24, opt_crs, opt_bounds = process_optical(scene_id, output_dir)

        # 2. All SARs
        safe_folders = find_all_safe_folders(scene_id)
        print(f"Found {len(safe_folders)} SAR products")

        for idx, safe_folder in enumerate(safe_folders, start=1):
            print(f"  → Processing SAR {idx}: {safe_folder.name}")
            out_sar = output_dir / f"sar_{idx}_24m.tif"
            process_one_sar(safe_folder, opt_bounds, opt_crs, out_sar)
            print(f"    Saved → {out_sar.name}")

        print(f"✓ {scene_id} completed successfully")
        return True

    except Exception as e:
        print(f"✗ {scene_id} FAILED: {e}")
        return False


def main():
    print(f"Starting batch processing of {len(SCENE_IDS)} scenes (both SARs)...\n")

    success = 0
    failed = []

    for scene_id in SCENE_IDS:
        if process_one_scene(scene_id):
            success += 1
        else:
            failed.append(scene_id)

    print("\n" + "="*70)
    print("BATCH PROCESSING FINISHED")
    print(f"Successful : {success}/{len(SCENE_IDS)}")
    if failed:
        print("Failed scenes:")
        for f in failed:
            print(f"  - {f}")
    print("="*70)


if __name__ == "__main__":
    main()