# Multi-Modal Satellite Road Extraction (Optical + SAR + OSM)

A deep learning and geospatial remote sensing pipeline for extracting multi-class road networks using aligned Sentinel-2 (Optical), Sentinel-1 (SAR GRD), and OpenStreetMap (Geofabrik vector) data.

---

## Workspace Directory Structure

```text
Road-Extraction/
├── data/
│   ├── downloads/              # Raw optical archives (.zip)
│   ├── raw/
│   │   ├── optical/            # Extracted multi-spectral Sentinel-2 bands
│   │   ├── sar/                # Sentinel-1 SAR products (.SAFE)
│   │   └── vectors/            # Vector OSM data (south-india-latest.osm.pbf)
│   └── processed/
│       ├── scenes/             # 24m co-registered rasters (optical_24m.tif, sar_*_24m.tif)
│       ├── masks/              # Multi-class rasterized road masks (mask_*.tif)
│       ├── quality_report.csv  # Overlap metrics and sensor alignment audit
│       ├── footprints.geojson  # Bounding footprints for all optical scenes
│       └── footprints_summary.csv
├── src/
│   ├── data_prep/
│   │   ├── preprocess_all_scenes.py  # Co-registers optical & SAR onto identical 24m grids
│   │   ├── quality_check.py          # Calculates valid data overlap between optical & SAR
│   │   ├── create_masks_geofabrik.py # Extracts & buffers OSM roads into 3-class GeoTIFF masks
│   │   ├── organize_optical.py       # Unpacks optical zips from downloads/ into raw/optical/
│   │   ├── organize_sar.py           # Unpacks SAR zips into raw/sar/
│   │   ├── download_matching_sar.py  # Fetches Sentinel-1 scenes matching optical footprints
│   │   └── extract_footprints.py     # Extracts WGS84 footprints from optical scenes
│   ├── visualization/
│   │   ├── view_mask_validation.py   # Side-by-side plot of False-Color Optical vs Road Mask
│   │   └── view_scene.py             # Visualizes Optical + Multi-SAR (VV & VH in dB)
│   └── utils/
│       └── verify_dataset_readiness.py # Audits scenes and masks before patch tiling
├── configs/
│   └── paths.py                # Centralized, single-source-of-truth path definitions
├── notebooks/                  # Interactive experimentation and analysis notebooks
├── .env.example                # Earthdata credential template
├── .gitignore                  # Geospatial raster & big data exclusions
└── requirements.txt            # Python dependencies
```

---

## Centralized Configuration (`configs/paths.py`)

All scripts reference absolute or project-relative directories via `configs.paths`:

```python
from configs.paths import (
    DATA_DIR,
    RAW_OPTICAL_DIR,
    RAW_SAR_DIR,
    RAW_VECTORS_DIR,
    PROCESSED_SCENES_DIR,
    MASK_DIR,
    PATCHES_DIR,
    QUALITY_CSV,
    PBF_PATH,
)
```

---

## Road Classification Scheme (at 24m resolution)

| Class ID | Road Type | OSM Highway Tags | Physical Buffer |
| :---: | :---: | :---: | :---: |
| **0** | Background | Non-road pixels | - |
| **1** | Local / Minor | `residential`, `unclassified`, `service`, `living_street` | 11.0 m (~1 px) |
| **2** | Secondary | `secondary`, `tertiary`, `secondary_link`, `tertiary_link` | 17.0 m (~1.5 px) |
| **3** | Major / Highway | `motorway`, `trunk`, `primary`, link variants | 26.0 m (~2+ px) |

---

## Dataset Readiness Verification

To audit that all scenes, rasters, and masks are 100% aligned and ready for downstream patch extraction:

```bash
python src/utils/verify_dataset_readiness.py
```
