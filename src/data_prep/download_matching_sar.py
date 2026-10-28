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


def parse_product_date(product):
    """Extract datetime from ASF product start time property."""
    start_time_str = product.properties.get("startTime")
    if start_time_str:
        try:
            return datetime.strptime(str(start_time_str)[:10], "%Y-%m-%d")
        except Exception:
            pass
    return None


def collect_products_across_scenes(gdf):
    """
    Searches SAR candidates for each optical footprint, ranks them by
    overlap descending and date proximity, de-duplicates by fileID across
    scenes, and groups queued downloads by output directory.
    """
    products_by_dir = {}
    file_locations = {}

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
                beamMode=asf.BEAMMODE.IW,
                polarization=[asf.POLARIZATION.VV_VH],
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
                    sar_date = parse_product_date(r)
                    date_diff = (
                        abs((sar_date - opt_date).days)
                        if sar_date
                        else 999999
                    )
                    scored.append((overlap, date_diff, r))
            except Exception:
                continue

        # Rank candidates by overlap descending, then smallest abs date difference
        scored.sort(key=lambda x: (-x[0], x[1]))
        selected = scored[:MAX_SAR_PER_OPTICAL]

        if not selected:
            print("  → No SAR with sufficient overlap")
            continue

        out_dir = OUTPUT_ROOT / str(scene_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        for overlap, date_diff, product in selected:
            file_name = product.properties.get("fileName")
            file_id = product.properties.get("fileID")
            expected_file = out_dir / file_name if file_name else None

            # Skip files that already exist locally in this scene's directory
            if expected_file and expected_file.exists():
                print(f"  [Skipped] Already downloaded: {file_name}")
                file_locations[file_id] = expected_file
                continue

            # Check if product is already downloaded or queued in another scene
            if file_id in file_locations:
                shared_src = file_locations[file_id]
                links_path = out_dir / "links.txt"
                with open(links_path, "a", encoding="utf-8") as lf:
                    lf.write(f"{file_id}\t{file_name}\t{shared_src}\n")
                print(
                    f"  [Shared] Product {file_id} already in {shared_src.parent.name} → linked in links.txt"
                )
                continue

            # Queue new product for download
            print(
                f"  [Queued] Overlap: {overlap:.1%}, Date Diff: {date_diff}d | {file_id}"
            )
            file_locations[file_id] = expected_file
            if out_dir not in products_by_dir:
                products_by_dir[out_dir] = []
            products_by_dir[out_dir].append(product)

    return products_by_dir


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

    # Collect missing products across all scenes
    print("\nQueuing scenes for download...")
    products_by_dir = collect_products_across_scenes(gdf)

    total_queued = sum(len(prods) for prods in products_by_dir.values())
    if total_queued == 0:
        print("\nAll target files are already downloaded or shared!")
        return

    print(
        f"\nStarting downloads for {total_queued} items across {len(products_by_dir)} directories using {MAX_CONCURRENT_DOWNLOADS} workers per directory..."
    )

    # Execute downloads per directory in batch
    for out_dir, products in products_by_dir.items():
        try:
            print(
                f"\nDownloading {len(products)} products to {out_dir.name}..."
            )
            search_container = asf.ASFSearchResults(products)
            search_container.download(
                path=str(out_dir),
                session=session,
                processes=MAX_CONCURRENT_DOWNLOADS,
            )
            print(f"  ✓ Finished downloading to {out_dir.name}")
        except Exception as e:
            print(f"  ❌ Download failed for directory {out_dir.name}: {e}")

    print("\n" + "=" * 60)
    print("=== All SAR download tasks finished ===")
    print(f"Check the directory: {OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()
