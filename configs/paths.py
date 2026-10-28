from pathlib import Path

# Centralized workspace path definitions
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DOWNLOADS_DIR = DATA_DIR / "downloads"

RAW_OPTICAL_DIR = RAW_DIR / "optical"
RAW_SAR_DIR = RAW_DIR / "sar"
RAW_VECTORS_DIR = RAW_DIR / "vectors"

PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_SCENES_DIR = PROCESSED_DIR / "scenes"
MASK_DIR = PROCESSED_DIR / "masks"
PATCHES_DIR = PROCESSED_DIR / "patches"
QUALITY_CSV = PROCESSED_DIR / "quality_report.csv"
PBF_PATH = RAW_VECTORS_DIR / "south-india-latest.osm.pbf"

FOOTPRINTS_GEOJSON = PROCESSED_DIR / "footprints.geojson"
FOOTPRINTS_CSV = PROCESSED_DIR / "footprints_summary.csv"

PREVIEWS_DIR = PROCESSED_DIR / "previews"