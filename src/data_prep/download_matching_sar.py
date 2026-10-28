import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import asf_search as asf
import geopandas as gpd
from dotenv import load_dotenv
from shapely.geometry import shape

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

from configs.paths import FOOTPRINTS_GEOJSON, RAW_SAR_DIR

# ============================================================
# CONFIGURATION
# ============================================================

FOOTPRINTS_FILE = FOOTPRINTS_GEOJSON
OUTPUT_ROOT = RAW_SAR_DIR
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Search window around optical date
DAYS_BEFORE = 12
DAYS_AFTER = 12

# Maximum SAR scenes to download per optical scene
MAX_SAR_PER_OPTICAL = 2

# Minimum intersection area ratio (0.25 = 25% overlap required)
MIN_OVERLAP_RATIO = 0.25

# Parallel processes (3-5 processes maximizes speed without hitting rate limits)
MAX_CONCURRENT_DOWNLOADS = 4

# Earthdata credentials
load_dotenv()
EARTHDATA_USER = os.getenv("EARTHDATA_USER")
EARTHDATA_PASS = os.getenv("EARTHDATA_PASS")

# ============================================================


def get_overlap_ratio(poly1, poly2):
    if poly1.is_empty or poly2.is_empty:
        return 0.0
    inter = poly1.intersection(poly2).area
    return inter / poly1.area


def process_and_collect_products(gdf):
    """Searches and groups missing SAR products by output directory."""
    download_queue = []

    for idx, row in gdf.iterrows():
        scene_id = row["scene_id"]
        date_str = row["date"]
        opt_poly = row.geometry

        print(f"\n{'='*60}")
        print(f"Optical Scene: {scene_id} | Date: {date_str}")

        if hasattr(date_str, "to_pydatetime"):
            opt_date = date_str.to_pydatetime()
        else:
            opt_date = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")

        start = (opt_date - timedelta(days=DAYS_BEFORE)).strftime("%Y-%m-%d")
        end = (opt_date + timedelta(days=DAYS_AFTER)).strftime("%Y-%m-%d")

        try:
            results = asf.geo_search(
                platform=[asf.PLATFORM.SENTINEL1],
                processingLevel=[asf.PRODUCT_TYPE.GRD_HD],
                intersectsWith=opt_poly.convex_hull.wkt,
                start=start,
                end=end,
                maxResults=30,
            )
        except Exception as e:
            print(f"  Search error: {e}")
            continue

        print(f"  Found {len(results)} candidate SAR scenes")

        scored = []
        for r in results:
            try:
                sar_poly = shape(r.geometry)
                overlap = get_overlap_ratio(opt_poly, sar_poly)
                if overlap >= MIN_OVERLAP_RATIO:
                    scored.append((overlap, r))
            except Exception:
                continue

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = scored[:MAX_SAR_PER_OPTICAL]

        if not selected:
            print("  → No SAR with sufficient overlap")
            continue

        out_dir = OUTPUT_ROOT / str(scene_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        for overlap, product in selected:
            file_name = product.properties.get("fileName")
            expected_file = out_dir / file_name if file_name else None

            # Skip files that already exist locally
            if expected_file and expected_file.exists():
                print(f"  [Skipped] Already downloaded: {file_name}")
            else:
                print(
                    f"  [Queued] Overlap: {overlap:.1%} | {product.properties.get('fileID')}"
                )
                download_queue.append((product, out_dir))

    return download_queue


def main():
    if not EARTHDATA_USER or not EARTHDATA_PASS:
        raise ValueError("EARTHDATA_USER or EARTHDATA_PASS not set in .env")

    print("Loading optical footprints...")
    gdf = gpd.read_file(FOOTPRINTS_FILE)
    print(f"Loaded {len(gdf)} optical scenes")

    session = asf.ASFSession()
    try:
        session.auth_with_creds(EARTHDATA_USER, EARTHDATA_PASS)
        print("Earthdata authentication successful")
    except Exception as e:
        print(f"Authentication error: {e}")
        return

    # Collect missing products
    print("\nQueuing scenes for download...")
    download_queue = process_and_collect_products(gdf)

    if not download_queue:
        print("\nAll target files are already downloaded!")
        return

    print(
        f"\nStarting parallel downloads for {len(download_queue)} items using {MAX_CONCURRENT_DOWNLOADS} workers..."
    )

    # Execute downloads with multi-process speed
    for product, out_dir in download_queue:
        try:
            # Packaging single product into ASFSearchResults unlocks the native parallel downloader
            search_container = asf.ASFSearchResults([product])
            search_container.download(
                path=str(out_dir), session=session, processes=MAX_CONCURRENT_DOWNLOADS
            )
            print(f"  ✓ Download completed: {product.properties.get('fileID')}")
        except Exception as e:
            print(
                f"  ❌ Download failed for {product.properties.get('fileID')}: {e}"
            )

    print("\n" + "=" * 60)
    print("=== All SAR download tasks finished ===")
    print(f"Check the directory: {OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()