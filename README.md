# Multi-Modal Satellite Road Extraction (LISS-III Optical + Sentinel-1 SAR + OSM)

A geospatial data-preparation pipeline that builds a training dataset for multi-class road extraction. It combines:

- **Optical:** ISRO Resourcesat-2 **LISS-III** scenes (4 bands: Green, Red, NIR, SWIR) downloaded manually from [Bhoonidhi](https://bhoonidhi.nrsc.gov.in/)
- **SAR:** Sentinel-1 **GRD (VV + VH)** scenes downloaded automatically from the [ASF archive](https://search.asf.alaska.edu/) (NASA Earthdata login required)
- **Labels:** road masks rasterized offline from an **OpenStreetMap** extract ([Geofabrik](https://download.geofabrik.de/asia/india.html))

The output is a set of `256 x 256` multi-channel patches (`.npz`) with a 4-class road mask, split into `train / val / test` by location.

> Status: the repository currently covers **data preparation** (steps 1-10 below).

---

## Pipeline at a glance

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

## Repository structure

```text
Road-Extraction/
├── configs/
│   ├── paths.py                       # All directory/file paths (anchored to the repo root)
│   └── settings.py                    # Shared pipeline settings and thresholds
├── src/
│   ├── data_prep/
│   │   ├── organize_optical.py        # [2]  unzip optical archives -> data/raw/optical/<scene>/
│   │   ├── extract_footprints.py      # [3]  scene bounding boxes -> footprints.geojson / .csv
│   │   ├── download_matching_sar.py   # [4]  search + download Sentinel-1 GRD from ASF
│   │   ├── organize_sar.py            # [5]  unzip SAR archives -> data/raw/sar/<scene>/*.SAFE
│   │   ├── preprocess_all_scenes.py   # [6]  resample optical to 24 m, warp SAR onto the same grid
│   │   ├── quality_check.py           # [7]  valid optical/SAR overlap -> quality_report.csv
│   │   ├── create_masks_geofabrik.py  # [9]  OSM roads -> 3-class buffered raster masks
│   │   └── extract_patches.py         # [10] 256x256 tiles, quality gates, location-level split
│   ├── utils/
│   │   ├── scene_selection.py         # Select qualifying scenes meeting quality thresholds
│   │   ├── verify_dataset_readiness.py  # [9b] audit scenes + masks before tiling
│   │   └── viz.py                     # Shared robust stretch and SAR display normalization
│   └── visualization/
│       ├── view_scene.py              # optical false colour + SAR VV/VH preview
│       ├── view_mask_validation.py    # optical vs road mask side by side
│       └── view_patches.py            # inspect extracted patches
├── notebooks/                         # empty placeholder
├── .env.example                       # Earthdata credentials template
├── requirements.txt
└── README.md
```

Everything under `data/` is git-ignored and created on demand:

```text
data/
├── downloads/                  # your Bhoonidhi .zip files (input to step 2)
├── raw/
│   ├── optical/<scene_id>/     # BAND*.tif per scene          (step 2)
│   ├── sar/<scene_id>/         # Sentinel-1 .zip + .SAFE      (steps 4-5)
│   └── vectors/                # the .osm.pbf file            (step 8)
└── processed/
    ├── footprints.geojson      # step 3
    ├── footprints_summary.csv  # step 3
    ├── scenes/<scene_id>/      # optical_24m.tif, sar_1_24m.tif, sar_2_24m.tif  (step 6)
    ├── quality_report.csv      # step 7
    ├── masks/mask_<scene_id>.tif   # step 9
    ├── patches/{train,val,test}/tileNNNNN.npz + tiles_metadata.csv   # step 10
    └── previews/               # PNGs written by the viewers
```

---

## Setup

Requirements: Python 3.10+ and an [Earthdata](https://urs.earthdata.nasa.gov/) account (free) for the SAR download.

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

```bash
# Linux / macOS
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create your credentials file (it is git-ignored):

```powershell
Copy-Item .env.example .env      # Linux/macOS: cp .env.example .env
```

```ini
EARTHDATA_USER=your_earthdata_username
EARTHDATA_PASS=your_earthdata_password
```

All commands below are run from the **repository root**.

---

## Step-by-step execution

### Step 1 - Download optical scenes from Bhoonidhi (manual)

1. Log in to [Bhoonidhi](https://bhoonidhi.nrsc.gov.in/) and download the LISS-III scenes you need (one `.zip` per scene).
2. Put the archives in `data/downloads/`.
3. **Rename every zip to `YYYY-MM-DD_Location.zip`**, for example `2025-11-17_Amaravati.zip`.

The rename is required, not cosmetic. The zip file name becomes the **scene ID** used by every later step, and the scripts read the acquisition date from the first 10 characters (`extract_footprints.py`, then the SAR search window in `download_matching_sar.py`). A wrong name means a wrong SAR date window.

Each scene must contain `BAND*.tif` files; the first four (sorted by name) are used as Green, Red, NIR, SWIR.

### Step 2 - Organize optical scenes

```bash
python src/data_prep/organize_optical.py [--force]
```

Unzips each archive into `data/raw/optical/<scene_id>/`. If the scene already exists with `BAND*.tif` files, it is skipped unless `--force` is given. If the zip holds a single top-level folder, its contents are moved up one level. Stale temporary extraction folders are cleared before extraction.

### Step 3 - Extract scene footprints

```bash
python src/data_prep/extract_footprints.py
```

Validates scene folder naming format (`YYYY-MM-DD_Location`), reads the primary optical band (`BAND*.tif`), and writes its WGS84 bounding box to `data/processed/footprints.geojson` (and `footprints_summary.csv`). This file drives the SAR search in the next step.

### Step 4 - Download the matching Sentinel-1 SAR

```bash
python src/data_prep/download_matching_sar.py
```

For each footprint it searches ASF for Sentinel-1 GRD_HD scenes (IW beam mode, VV+VH polarization) within **+/-12 days** of the optical date that cover at least **25 %** of the optical footprint. Candidates are ranked by overlap descending, then by smallest date difference to the optical acquisition. It keeps the **2 best by overlap**.

Downloads are de-duplicated across scenes by product `fileID`. If a SAR product is shared between scenes, it is downloaded once into the first scene's folder and linked in `<scene>/links.txt` for subsequent scenes. Downloads are executed in parallel using multi-product batch downloads per directory with `MAX_CONCURRENT_DOWNLOADS`. Existing zips are skipped.

Requires `EARTHDATA_USER` / `EARTHDATA_PASS` in `.env`.

### Step 5 - Organize SAR scenes

```bash
python src/data_prep/organize_sar.py [--force]
```

Unzips each SAR archive in place, producing `data/raw/sar/<scene_id>/*.SAFE/`. Skips extraction if a matching `.SAFE` folder already exists, unless `--force` is provided. Uses `\\?\` long paths on Windows.

### Step 6 - Preprocess and co-register to a 24 m grid

```bash
python src/data_prep/preprocess_all_scenes.py [--force]
```

For every folder in `data/raw/optical/`:

- stacks the first 4 optical bands and resamples them directly to **24 m** (bilinear), written straight to `data/processed/scenes/<scene_id>/optical_24m.tif` without intermediate full-resolution files
- sorts `.SAFE` folders chronologically by the acquisition start timestamp in their name
- warps each `.SAFE` product (VV and VH) onto that scene's optical grid using the SAR GCPs with **average** resampling (`Resampling.average`)
- converts GRD amplitude values to intensity ($DN^2$) then `10*log10`, written as uncalibrated backscatter (dB-like) to `sar_1_24m.tif`, `sar_2_24m.tif`, ... (2 bands each: VV, VH; nodata `-9999`)
- prints dynamic range percentiles per scene

Skips processing for scenes where `optical_24m.tif` and all expected `sar_*_24m.tif` rasters already exist, unless `--force` is specified.

### Step 7 - Quality report

```bash
python src/data_prep/quality_check.py
```

Calculates valid overlapping pixels where all 4 optical bands are `> 0` and both SAR bands (VV, VH) are `> -9990`. Evaluates all existing `sar_*_24m.tif` rasters dynamically in a loop and writes `data/processed/quality_report.csv` containing `n_sar`, `valid_pct_best`, and per-SAR metrics (`sar1_valid_pct`, `sar2_valid_pct`, etc.).

Scenes qualify for patch tiling when at least `REQUIRED_SAR_PRODUCTS` (default 2) SAR products each have `>= MIN_VALID_PCT` (default 50.0%) valid overlap.

### Step 8 - Download the OpenStreetMap extract from Geofabrik (manual)

Download the southern India `.osm.pbf` from [Geofabrik - India](https://download.geofabrik.de/asia/india.html) (the "Southern Zone").

Save it as exactly:

```text
data/raw/vectors/south-india-latest.osm.pbf
```

The file name is set in `configs/paths.py` (`PBF_PATH`); rename your download to match, or change that constant.

### Step 9 - Create the road masks

```bash
python src/data_prep/create_masks_geofabrik.py
```

Selects scenes based on optical validity (`optical_24m.tif` exists). For each scene, clips the PBF to the scene bounds (Pyrosm, `network_type="all"`), parses multi-value tags (semicolon-separated), classifies roads, buffers them, and rasterizes onto the scene's optical grid -> `data/processed/masks/mask_<scene_id>.tif` (`uint8`, with no nodata tag so background class 0 is retained).

| Class | Meaning | OSM `highway` values | Buffer radius | Resulting mask width |
| :---: | --- | --- | :---: | :---: |
| 0 | Background | everything else | - | - |
| 1 | Local / minor | `residential`, `unclassified`, `service`, `living_street`, `track` | 11 m | ~22 m |
| 2 | Secondary | `secondary`, `tertiary` (+ `_link`) | 17 m | ~34 m |
| 3 | Major | `motorway`, `trunk`, `primary` (+ `_link`) | 26 m | ~52 m |

Ways shorter than 5 m are ignored. Where roads overlap, the higher class wins. Track inclusion is governed by `INCLUDE_TRACKS = True`.

### Step 9b - (Optional) Verify dataset readiness

```bash
python src/utils/verify_dataset_readiness.py
```

Audits qualifying scenes before tiling: checks file presence, verifies that optical, mask, and all SAR rasters match in shape, CRS, affine transform (within tolerance), and bounding box, confirms optical has 4 bands, checks road pixel presence, and validates SAR dB values against shared settings `[SAR_MIN_DB, SAR_MAX_DB]`.

### Step 10 - Create the patches

```bash
python src/data_prep/extract_patches.py [--seed 42]
```

Automatically removes any stale files from `data/processed/patches/` and recreates the directory. Selects scenes qualifying with at least `REQUIRED_SAR_PRODUCTS` meeting `MIN_VALID_PCT`.

Slides a `256 x 256` window with stride `128` over every selected scene and keeps a tile only if it passes all gates:

| Gate | Rule |
| --- | --- |
| Nodata | optical band 0 `<= 0` in at most 10 % of pixels, and SAR `<= -9990` in at most 10 % |
| Cloud proxy | relative scene threshold: visible brightness (mean of Green and Red) `> 0.8 * scene_max` AND low NIR/SWIR contrast (`|NIR - SWIR| < 0.1 * scene_max`) in at most 18 % of pixels (where `scene_max` is the 99.9th percentile of NIR band) |
| Road content | at least 80 road pixels, and road ratio at most 0.65 |

**Split:** scenes are partitioned by **geographic location** (`location = scene_id.split("_", 1)[1]`) using a seeded shuffle (`random.Random(seed)`):

- **1 location:** all scenes assigned to `train` (prints clear notice that val and test are empty)
- **2 locations:** 1 location assigned to `train`, 1 to `val` (prints clear notice that test is empty)
- **3+ locations:** partitioned across `train`, `val`, and `test` (at least one location each)

Outputs:

- `data/processed/patches/<split>/tileNNNNN.npz` containing `image` (`float32`, `(C, 256, 256)`) and `mask` (`uint8`, `(256, 256)`)
- `data/processed/patches/tiles_metadata.csv` (tile id, scene, split, `y`, `x`, road pixels, road ratio, channel count, CRS, scene bounds)

---

## Data format

**Channels per patch** = 4 optical + 2 per SAR product + 1 validity mask.

| Index | Content |
| :---: | --- |
| 0-3 | LISS-III Green, Red, NIR, SWIR (raw DN as `float32`, not normalized) |
| 4-5 | SAR product 1: VV, VH (uncalibrated backscatter dB-like, sentinels/NaN replaced with 0.0) |
| 6-7 | SAR product 2: VV, VH (uncalibrated backscatter dB-like, sentinels/NaN replaced with 0.0) |
| 8 | Validity mask: `1.0` where all SAR channels are valid and optical band 0 > 0, else `0.0` |

In a standard configuration requiring 2 SAR products, patches have **9 channels** (`C = 9`). SAR sentinels (`-9999`) and NaNs are replaced with `0.0`, while true validity is encoded in the final mask channel.

Loading a patch:

```python
import numpy as np

with np.load("data/processed/patches/train/tile00001.npz") as d:
    image, mask = d["image"], d["mask"]  # (C, 256, 256) float32, (256, 256) uint8
```

---

## Visualization tools

The visualization tools accept command-line arguments and automatically default to available processed scenes.

| Script | CLI arguments | Output |
| --- | --- | --- |
| `src/visualization/view_scene.py` | `[SCENE_ID]` (default: first available in `data/processed/scenes`), `--downsample` (default 4), `--force` | `previews/<scene>_preview.png` (cached) |
| `src/visualization/view_mask_validation.py` | `[SCENE_ID]` (default: first available in `data/processed/scenes`), `--downsample` (default 4) | `previews/<scene>_mask_validation.png` and on-screen |
| `src/visualization/view_patches.py` | `--split` (default: `train`), `--n` (default: 2) | `previews/patch_inspection_<split>.png` |

`view_patches.py` dynamically displays all SAR channels present (derived from channel count `C`) along with the validity mask channel and the target road mask.

---

## Configuration reference

Shared thresholds and paths are centralized across `configs/paths.py` and `configs/settings.py`.

**Shared Constants (`configs/settings.py`):**

| Constant | Value | Description |
| --- | --- | --- |
| `MIN_VALID_PCT` | `50.0` | Minimum optical/SAR valid overlap percentage required |
| `TARGET_RES` | `24.0` | Target spatial resolution in meters |
| `REQUIRED_SAR_PRODUCTS` | `2` | Number of qualifying SAR products required per scene for patch extraction |
| `SAR_CALIBRATION_NOTE` | `"uncalibrated backscatter (dB-like)"` | Radiometric representation note for SAR intensity logs |
| `SAR_MIN_DB` | `-50.0` | Minimum expected SAR decibel threshold for readiness audit |
| `SAR_MAX_DB` | `15.0` | Maximum expected SAR decibel threshold for readiness audit |

**Pipeline Parameters:**

| Setting | Where | Value |
| --- | --- | --- |
| Scene naming convention | `extract_footprints.py` | `^\d{4}-\d{2}-\d{2}_.+` (`YYYY-MM-DD_Location`) |
| OSM vector file path | `configs/paths.py` (`PBF_PATH`) | `data/raw/vectors/south-india-latest.osm.pbf` |
| SAR search window, count, overlap | `download_matching_sar.py` | +/-12 days, 2 scenes, 25 % |
| Road classes and buffers | `create_masks_geofabrik.py` | Class 1 (11m), Class 2 (17m), Class 3 (26m) |
| Track inclusion | `create_masks_geofabrik.py` (`INCLUDE_TRACKS`) | `True` (Class 1) |
| Minimum road length | `create_masks_geofabrik.py` (`MIN_ROAD_LENGTH_M`) | 5.0 m |
| Patch size, stride, nodata, cloud | `extract_patches.py` | 256x256, stride 128, max 10% nodata, max 18% cloud |
| Split strategy | `extract_patches.py` (`generate_location_split`) | Location-level split, default seed 42 |
