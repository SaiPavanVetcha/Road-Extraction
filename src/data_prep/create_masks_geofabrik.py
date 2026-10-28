import sys
from pathlib import Path
import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from pyrosm import OSM
from shapely.geometry import box

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

from configs.paths import PROCESSED_SCENES_DIR, MASK_DIR, PBF_PATH
from configs.settings import MIN_VALID_PCT

# ============================================================
# CONFIGURATION
# ============================================================
PROCESSED_ROOT = PROCESSED_SCENES_DIR
MASK_ROOT = MASK_DIR
MASK_ROOT.mkdir(parents=True, exist_ok=True)

# Shared threshold reference
MIN_VALID_THRESHOLD = MIN_VALID_PCT

INCLUDE_TRACKS = True

# Realistic class design for 24m spatial resolution
ROAD_CLASS_MAP = {
    "motorway": 3,
    "motorway_link": 3,
    "trunk": 3,
    "trunk_link": 3,
    "primary": 3,
    "primary_link": 3,
    "secondary": 2,
    "secondary_link": 2,
    "tertiary": 2,
    "tertiary_link": 2,
    "residential": 1,
    "unclassified": 1,
    "service": 1,
    "living_street": 1,
}
if INCLUDE_TRACKS:
    ROAD_CLASS_MAP["track"] = 1

CLASS_BUFFER = {
    1: 11.0,  # Local (~22m physical width = ~1 pixel)
    2: 17.0,  # Secondary (~34m physical width = ~1.5 pixels)
    3: 26.0,  # Major (~52m physical width = 2+ pixels)
}

MIN_ROAD_LENGTH_M = 5.0  # Ignore vector noise shorter than 5 meters
# ============================================================


def assign_class(tag):
    """Parses OSM highway tags into integer road classes (1, 2, or 3)."""
    if tag is None:
        return 0
    if isinstance(tag, list):
        tag = tag[0]
    if isinstance(tag, str) and ";" in tag:
        tag = tag.split(";")[0].strip()
    return ROAD_CLASS_MAP.get(str(tag).lower(), 0)


def get_selected_scenes():
    """Returns scene IDs where optical_24m.tif exists."""
    if not PROCESSED_ROOT.exists():
        return []

    selected = [
        d.name
        for d in sorted(PROCESSED_ROOT.iterdir())
        if d.is_dir() and (d / "optical_24m.tif").exists()
    ]

    print(f"Selected {len(selected)} scenes with valid optical raster:")
    for s in selected:
        print(f"  ✓ {s}")
    print()
    return selected


def create_mask_from_pbf(scene_id):
    """Clips local PBF data and rasterizes multi-class mask for a scene."""
    print(f"=== Creating mask: {scene_id} ===")

    opt_path = PROCESSED_ROOT / scene_id / "optical_24m.tif"
    if not opt_path.exists():
        print("  optical_24m.tif not found – skipped")
        return False

    with rasterio.open(opt_path) as src:
        meta = src.meta.copy()
        transform = src.transform
        raster_shape = (src.height, src.width)
        bounds = src.bounds
        crs = src.crs

    # 1. Bounding box in WGS84 [minx, miny, maxx, maxy] for Pyrosm
    gdf_bounds = gpd.GeoDataFrame(
        {"geometry": [box(*bounds)]}, crs=crs
    ).to_crs("EPSG:4326")
    minx, miny, maxx, maxy = gdf_bounds.total_bounds
    bounding_box = [minx, miny, maxx, maxy]

    # 2. Re-instantiate OSM parser per scene to prevent Cython memory accumulation
    try:
        osm_parser = OSM(str(PBF_PATH), bounding_box=bounding_box)
        roads = osm_parser.get_network(network_type="all")
    except Exception as e:
        print(f"  ❌ Pyrosm extraction failed: {e}")
        return False

    if roads is None or roads.empty or "highway" not in roads.columns:
        print("  ⚠️ No valid road network columns found in bounding box")
        mask = np.zeros(raster_shape, dtype=np.uint8)
    else:
        # Reproject roads to scene EPSG:32644
        roads = roads.to_crs(crs)
        print(f"  Extracted {len(roads)} road segments from local PBF")

        shapes_to_rasterize = []
        for _, row in roads.iterrows():
            if row.geometry.length < MIN_ROAD_LENGTH_M:
                continue

            cls = assign_class(row.get("highway"))
            if cls == 0:
                continue

            buffered = row.geometry.buffer(CLASS_BUFFER[cls])
            shapes_to_rasterize.append((buffered, cls))

        if not shapes_to_rasterize:
            print("  ⚠️ No valid classified roads found")
            mask = np.zeros(raster_shape, dtype=np.uint8)
        else:
            # Sort by Class ID ascending (1 -> 2 -> 3) so Major Highways overwrite Local Roads
            shapes_to_rasterize.sort(key=lambda x: x[1])

            mask = features.rasterize(
                shapes=shapes_to_rasterize,
                out_shape=raster_shape,
                transform=transform,
                fill=0,
                dtype=np.uint8,
                all_touched=False,
            )

    # Save multi-class mask file
    meta.update({"count": 1, "dtype": "uint8"})
    meta.pop("nodata", None)
    out_path = MASK_ROOT / f"mask_{scene_id}.tif"
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(mask, 1)

    unique, counts = np.unique(mask, return_counts=True)
    print(f"  Saved → {out_path.name}")
    print("  Class distribution:")
    for u, c in zip(unique, counts):
        label = "Background" if u == 0 else f"Class {u}"
        print(f"    {label}: {c:,} pixels")
    print()
    return True


def main():
    if not PBF_PATH.exists():
        raise FileNotFoundError(
            f"Geofabrik extract missing at {PBF_PATH}.\n"
            f"Please download south-india-latest.osm.pbf into data/raw/vectors/"
        )

    selected_scenes = get_selected_scenes()
    if not selected_scenes:
        print("No scenes passed the quality threshold.")
        return

    print(f"Starting offline mask generation for {len(selected_scenes)} scenes...\n")

    success = 0
    failed = []

    for scene_id in selected_scenes:
        try:
            if create_mask_from_pbf(scene_id):
                success += 1
            else:
                failed.append(scene_id)
        except Exception as e:
            print(f"  ❌ {scene_id} failed: {e}")
            failed.append(scene_id)

    print("=" * 60)
    print("LOCAL MASK GENERATION FINISHED")
    print(f"Successful : {success}/{len(selected_scenes)}")
    if failed:
        print("Failed scenes:")
        for f in failed:
            print(f"  - {f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
