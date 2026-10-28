import os
import sys
import glob
import json
import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import box, mapping
import geopandas as gpd
from pathlib import Path
import pandas as pd

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

from configs.paths import RAW_OPTICAL_DIR, FOOTPRINTS_GEOJSON, FOOTPRINTS_CSV

# ============================================================
# CONFIGURATION
# ============================================================

OPTICAL_ROOT = RAW_OPTICAL_DIR          # folder containing all optical scenes
OUTPUT_GEOJSON = FOOTPRINTS_GEOJSON
OUTPUT_CSV = FOOTPRINTS_CSV

OUTPUT_GEOJSON.parent.mkdir(parents=True, exist_ok=True)

# ============================================================

def get_footprint(tif_path):
    """Extract footprint in WGS84 (EPSG:4326)"""
    with rasterio.open(tif_path) as src:
        # Get bounds in native CRS
        bounds = src.bounds
        crs = src.crs

        # Transform to WGS84
        left, bottom, right, top = transform_bounds(
            crs, "EPSG:4326",
            bounds.left, bounds.bottom, bounds.right, bounds.top
        )

        polygon = box(left, bottom, right, top)
        return polygon, {
            "left": left,
            "bottom": bottom,
            "right": right,
            "top": top,
            "width_deg": right - left,
            "height_deg": top - bottom,
            "crs_original": str(crs),
            "shape": src.shape
        }

def main():
    records = []
    geometries = []

    # Find all optical folders
    folders = sorted([f for f in Path(OPTICAL_ROOT).iterdir() if f.is_dir()])
    print(f"Found {len(folders)} optical scenes\n")

    for folder in folders:
        # Look for any band (prefer BAND2 or BAND3)
        band_candidates = list(folder.glob("BAND*.tif")) + list(folder.glob("*.tif"))
        
        if not band_candidates:
            print(f"⚠ No .tif found in {folder.name}")
            continue

        tif_path = band_candidates[0]
        print(f"Processing: {folder.name}")

        try:
            polygon, info = get_footprint(tif_path)

            record = {
                "scene_id": folder.name,
                "date": folder.name[:10],
                "location": "_".join(folder.name.split("_")[1:]),
                **info
            }
            records.append(record)
            geometries.append(polygon)

            print(f"  → Bounds: {info['left']:.4f}, {info['bottom']:.4f}, {info['right']:.4f}, {info['top']:.4f}")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(records, geometry=geometries, crs="EPSG:4326")
    gdf.to_file(OUTPUT_GEOJSON, driver="GeoJSON")
    print(f"\nSaved GeoJSON → {OUTPUT_GEOJSON}")

    # Save CSV summary
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved CSV     → {OUTPUT_CSV}")

    print("\n=== Footprint extraction completed ===")
    print(f"Total scenes processed: {len(records)}")

if __name__ == "__main__":
    main()