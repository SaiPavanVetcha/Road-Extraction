# Multi-Modal Satellite Road Extraction

**LISS-III Optical + Sentinel-1 SAR + OpenStreetMap**

A complete geospatial data-preparation pipeline that builds a high-quality training dataset for multi-class road extraction.

### Data Sources

| Modality    | Source                                      | Details                         |
| ----------- | ------------------------------------------- | ------------------------------- |
| **Optical** | ISRO Resourcesat-2 **LISS-III** (Bhoonidhi) | 4 bands: Green, Red, NIR, SWIR  |
| **SAR**     | Sentinel-1 **GRD** (ASF / Earthdata)        | VV + VH polarizations           |
| **Labels**  | OpenStreetMap (Geofabrik)                   | Multi-class buffered road masks |

**Output:** `256 × 256` multi-channel patches (`.npz`) with a 4-class road mask, split into `train / val / test` by geographic location.

> Current status: **Data preparation pipeline is complete** (steps 1–10).

---

## Pipeline Overview

```text
 Bhoonidhi (manual)                         ASF / Earthdata                    Geofabrik (manual)
  LISS-III .zip files                        Sentinel-1 GRD .zip                south .osm.pbf
        |                                          ^   |                              |
 [1] download                                      |   |                              |
        v                                          |   |                              |
 [2] organize_optical.py                           |   |                              |
        v                                          |   |                              |
 [3] extract_footprints.py -- footprints.geojson --+   |                              |
                                                   |   v                              |
                                   [4] download_matching_sar.py                       |
                                                       v                              |
                                   [5] organize_sar.py                                |
                                                       v                              |
 [6] preprocess_all_scenes.py  (optical_24m.tif + sar_N_24m.tif per scene)            |
        v                                                                             |
 [7] quality_check.py -> quality_report.csv                                           |
        v                                                                             |
 [8] download .osm.pbf  <-------------------------------------------------------------+
        v
 [9] create_masks_geofabrik.py -> masks/mask_<scene>.tif
        v
 [9b] verify_dataset_readiness.py (optional audit)
        v
 [10] extract_patches.py -> patches/{train,val,test}/tileNNNNN.npz + tiles_metadata.csv
```

---

## Repository Structure

```text
Road-Extraction/
├── configs/
│   ├── paths.py             # All directory & file paths
│   └── settings.py          # Shared thresholds & constants
├── src/
│   ├── data_prep/
│   │   ├── organize_optical.py
│   │   ├── extract_footprints.py
│   │   ├── download_matching_sar.py
│   │   ├── organize_sar.py
│   │   ├── preprocess_all_scenes.py
│   │   ├── quality_check.py
│   │   ├── create_masks_geofabrik.py
│   │   └── extract_patches.py
│   ├── utils/
│   │   ├── scene_selection.py
│   │   ├── verify_dataset_readiness.py
│   │   └── viz.py
│   └── visualization/
│       ├── view_scene.py
│       ├── view_mask_validation.py
│       └── view_patches.py
├── notebooks/
├── .env.example
├── requirements.txt
└── README.md
```

All large data lives under `data/` (git-ignored):

```text
data/
├── downloads/               # Original Bhoonidhi .zip files
├── raw/
│   ├── optical/<scene_id>/  # BAND*.tif
│   ├── sar/<scene_id>/      # .SAFE products
│   └── vectors/             # south-india-latest.osm.pbf
└── processed/
    ├── footprints.geojson
    ├── quality_report.csv
    ├── scenes/<scene_id>/   # optical_24m.tif, sar_1_24m.tif, ...
    ├── masks/mask_<scene_id>.tif
    ├── patches/
    │   ├── train/tileNNNNN.npz
    │   ├── val/tileNNNNN.npz
    │   ├── test/tileNNNNN.npz
    │   └── tiles_metadata.csv
    └── previews/
```

---

## Setup

**Requirements:** Python 3.10+ and a free [NASA Earthdata](https://urs.earthdata.nasa.gov/) account.

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

```ini
EARTHDATA_USER=your_username
EARTHDATA_PASS=your_password
```

---

## Step-by-Step Execution

### 1. Download Optical Scenes (Manual)
1. Download LISS-III scenes from [Bhoonidhi](https://bhoonidhi.nrsc.gov.in/).
2. Place the `.zip` files in `data/downloads/`.
3. **Rename** each file to `YYYY-MM-DD_Location.zip`  
   *Example:* `2025-11-17_Amaravati.zip`

> The file name becomes the official **scene ID** used by the entire pipeline.

### 2. Organize Optical
```bash
python -m src.data_prep.organize_optical
```

### 3. Extract Footprints
```bash
python -m src.data_prep.extract_footprints
```

### 4. Download Matching SAR
```bash
python -m src.data_prep.download_matching_sar
```
Searches ASF for Sentinel-1 GRD (IW, VV+VH) within ±12 days and ≥25% overlap. Downloads the 2 best matching products per optical scene.

### 5. Organize SAR
```bash
python -m src.data_prep.organize_sar
```

### 6. Preprocess & Co-register (24 m)
```bash
python -m src.data_prep.preprocess_all_scenes
```
- Resamples optical to 24 m
- Warps SAR onto the identical optical grid
- Converts SAR to uncalibrated backscatter (dB-like)

### 7. Quality Check
```bash
python -m src.data_prep.quality_check
```
Produces `quality_report.csv` with valid-overlap statistics.

### 8. Download OSM Extract (Manual)
Download the **Southern Zone** `.osm.pbf` from [Geofabrik India](https://download.geofabrik.de/asia/india.html) and save it as:
```text
data/raw/vectors/south-india-latest.osm.pbf
```

### 9. Create Road Masks
```bash
python -m src.data_prep.create_masks_geofabrik
```

| Class | Meaning       | OSM Tags                                                 | Buffer |
| :---: | ------------- | -------------------------------------------------------- | :----: |
|  0    | Background    | —                                                        |   —    |
|  1    | Local / Minor | residential, unclassified, service, living_street, track |  11 m  |
|  2    | Secondary     | secondary, tertiary (+ links)                            |  17 m  |
|  3    | Major         | motorway, trunk, primary (+ links)                       |  26 m  |

### 9b. (Optional) Verify Readiness
```bash
python -m src.utils.verify_dataset_readiness
```

### 10. Extract Patches
```bash
python -m src.data_prep.extract_patches [--seed 42]
```
**What it does:**
- Creates **two samples** per spatial window: `Optical + SAR1` and `Optical + SAR2`
- Applies quality gates (nodata, cloud proxy, road density)
- Splits by **geographic location** (prevents spatial leakage)
- Writes clean tile names: `tile00001.npz`, `tile00002.npz`, …
- Writes a detailed `tiles_metadata.csv`

---

## Data Format

### Patch Channels (always 7)

| Index | Content                               |
| :---: | ------------------------------------- |
|  0    | Optical Green (B2)                    |
|  1    | Optical Red (B3)                      |
|  2    | Optical NIR (B4)                      |
|  3    | Optical SWIR (B5)                     |
|  4    | SAR VV                                |
|  5    | SAR VH                                |
|  6    | Validity mask (1 = valid, 0 = nodata) |

```python
import numpy as np

data = np.load("data/processed/patches/train/tile00001.npz")
image = data["image"]  # (7, 256, 256) float32
mask = data["mask"]    # (256, 256) uint8
```

### Metadata (`tiles_metadata.csv`)

| Column                      | Description                  |
| --------------------------- | ---------------------------- |
| `tile_id`                   | `tile00001`                  |
| `scene_id`                  | Source optical scene         |
| `sar_product`               | `sar_1` or `sar_2`           |
| `sar_filename`              | Actual SAR file used         |
| `split`                     | train / val / test           |
| `y`, `x`                    | Window origin                |
| `road_pixels`, `road_ratio` | Road content statistics      |
| `num_channels`              | Always 7                     |
| `channels`                  | `B2,B3,B4,B5,VV,VH,validity` |
| `crs`, `scene_*`            | Geospatial bounds            |

---

## Visualization

```bash
# Full scene preview
python -m src.visualization.view_scene [SCENE_ID]

# Optical vs Mask
python -m src.visualization.view_mask_validation [SCENE_ID]

# Random patches
python -m src.visualization.view_patches --split train --n 4
```

---

## Configuration Reference

**`configs/settings.py`**

| Constant                    | Default  | Meaning                           |
| --------------------------- | -------- | --------------------------------- |
| `MIN_VALID_PCT`             | 50.0     | Minimum optical–SAR valid overlap |
| `TARGET_RES`                | 24.0     | Target resolution (metres)        |
| `REQUIRED_SAR_PRODUCTS`     | 2        | SAR products required per scene   |
| `CHANNELS_PER_TILE`         | 7        | Final patch channel count         |
| `SAR_MIN_DB` / `SAR_MAX_DB` | -50 / 15 | Expected SAR value range          |

**Key Pipeline Parameters**

| Parameter           | Value                               |
| ------------------- | ----------------------------------- |
| Patch size / stride | 256 × 256 / 128                     |
| Max nodata          | 10 %                                |
| Max cloud proxy     | 18 %                                |
| Min road pixels     | 80                                  |
| Split strategy      | Location-level (no spatial leakage) |

---

## License & Acknowledgements

- Optical data: [Bhoonidhi / NRSC / ISRO](https://bhoonidhi.nrsc.gov.in/)
- SAR data: [ASF DAAC / NASA Earthdata](https://search.asf.alaska.edu/)
- Vector data: [OpenStreetMap](https://www.openstreetmap.org/) via [Geofabrik](https://download.geofabrik.de/)