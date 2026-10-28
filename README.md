# Multi-Modal Satellite Road Extraction (LISS-III Optical + Sentinel-1 SAR + OSM)

A geospatial data-preparation pipeline that builds a training dataset for multi-class road extraction. It combines:

- **Optical:** ISRO Resourcesat-2 **LISS-III** scenes (4 bands: Green, Red, NIR, SWIR) downloaded manually from [Bhoonidhi](https://bhoonidhi.nrsc.gov.in/)
- **SAR:** Sentinel-1 **GRD (VV + VH)** scenes downloaded automatically from the [ASF archive](https://search.asf.alaska.edu/) (NASA Earthdata login required)
- **Labels:** road masks rasterized offline from an **OpenStreetMap** extract ([Geofabrik](https://download.geofabrik.de/asia/india.html))

The output is a set of `256 x 256` multi-channel patches (`.npz`) with a 4-class road mask, split into `train / val / test` by scene.

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
│   └── paths.py                       # All directory/file paths (anchored to the repo root)
├── src/
│   ├── data_prep/
│   │   ├── organize_optical.py        # [2]  unzip optical archives -> data/raw/optical/<scene>/
│   │   ├── extract_footprints.py      # [3]  scene bounding boxes -> footprints.geojson / .csv
│   │   ├── download_matching_sar.py   # [4]  search + download Sentinel-1 GRD from ASF
│   │   ├── organize_sar.py            # [5]  unzip SAR archives -> data/raw/sar/<scene>/*.SAFE
│   │   ├── preprocess_all_scenes.py   # [6]  resample optical to 24 m, warp SAR onto the same grid
│   │   ├── quality_check.py           # [7]  valid optical/SAR overlap -> quality_report.csv
│   │   ├── create_masks_geofabrik.py  # [9]  OSM roads -> 3-class buffered raster masks
│   │   └── extract_patches.py         # [10] 256x256 tiles, quality gates, scene-level split
│   ├── utils/
│   │   └── verify_dataset_readiness.py  # [9b] audit scenes + masks before tiling
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
    ├── scenes/<scene_id>/      # optical_full.tif, optical_24m.tif, sar_1_24m.tif, sar_2_24m.tif  (step 6)
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
python src/data_prep/organize_optical.py
```

Unzips each archive into `data/raw/optical/<scene_id>/`. If the zip holds a single top-level folder, its contents are moved up one level.

Re-running replaces any existing scene folder in `data/raw/optical/`.

### Step 3 - Extract scene footprints

```bash
python src/data_prep/extract_footprints.py
```

Reads one `.tif` per scene and writes its WGS84 bounding box to `data/processed/footprints.geojson` (and `footprints_summary.csv`). This file drives the SAR search in the next step.

### Step 4 - Download the matching Sentinel-1 SAR

```bash
python src/data_prep/download_matching_sar.py
```

For each footprint it searches ASF for Sentinel-1 GRD_HD scenes within **+/-12 days** of the optical date that cover at least **25 %** of the optical footprint, keeps the **2 best by overlap**, and downloads them to `data/raw/sar/<scene_id>/`. Existing zips are skipped.

Requires `EARTHDATA_USER` / `EARTHDATA_PASS` in `.env`. Each GRD zip is large (on the order of 1-2 GB), so check disk space first.

### Step 5 - Organize SAR scenes

```bash
python src/data_prep/organize_sar.py
```

Unzips each SAR archive in place, producing `data/raw/sar/<scene_id>/*.SAFE/`. Uses `\\?\` long paths on Windows. The zip files are kept after extraction.

### Step 6 - Preprocess and co-register to a 24 m grid

```bash
python src/data_prep/preprocess_all_scenes.py
```

For every folder in `data/raw/optical/`:

- stacks the first 4 optical bands and resamples them to **24 m** (bilinear), written to `data/processed/scenes/<scene_id>/optical_24m.tif`
- warps each `.SAFE` product (VV and VH) onto that scene's optical grid using the SAR GCPs, converts to a log scale, and writes `sar_1_24m.tif`, `sar_2_24m.tif`, ... (2 bands each: VV, VH; nodata `-9999`)

Each scene uses its **own** grid (the optical scene's CRS and bounds), not one shared global grid. It processes every scene folder found in `data/raw/optical/`, so large batches take a while.

### Step 7 - Quality report

```bash
python src/data_prep/quality_check.py
```

Writes `data/processed/quality_report.csv` with, for `sar_1` and `sar_2`, the percentage of pixels where optical band 1 and SAR band 1 (VV) are both valid (`sar1_valid_pct`, `sar2_valid_pct`, plus `*_opt_valid`, `*_sar_valid`, `*_shape_match`).

From here on, a scene is **selected** only if **both** `sar1_valid_pct` and `sar2_valid_pct` are `>= 50`.

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

For each selected scene it clips the PBF to the scene bounds (Pyrosm, `network_type="all"`), classifies roads, buffers them, and rasterizes onto the scene's optical grid -> `data/processed/masks/mask_<scene_id>.tif` (`uint8`).

| Class | Meaning | OSM `highway` values | Buffer radius | Resulting mask width |
| :---: | --- | --- | :---: | :---: |
| 0 | Background | everything else | - | - |
| 1 | Local / minor | `residential`, `unclassified`, `service`, `living_street` | 11 m | ~22 m |
| 2 | Secondary | `secondary`, `tertiary` (+ `_link`) | 17 m | ~34 m |
| 3 | Major | `motorway`, `trunk`, `primary` (+ `_link`) | 26 m | ~52 m |

The buffer is a **radius** on each side of the centreline, so the painted width is twice the number shown. These widths are deliberately exaggerated to be visible at 24 m/pixel; they are not real road widths. Ways shorter than 30 m are dropped. Where roads overlap, the higher class wins.

### Step 9b - (Optional) Verify dataset readiness

```bash
python src/utils/verify_dataset_readiness.py
```

For each selected scene it checks that the optical, SAR and mask files exist, that optical and mask have the same shape and CRS, that SAR rasters match the optical shape, and that the mask has road pixels.

### Step 10 - Create the patches

```bash
python src/data_prep/extract_patches.py
```

Slides a `256 x 256` window with stride `128` over every selected scene and keeps a tile only if it passes all gates:

| Gate | Rule |
| --- | --- |
| Nodata | optical band 1 `<= 0` in at most 10 % of pixels, and SAR `<= -9990` in at most 10 % |
| Cloud proxy | NIR band `> 3500` in at most 18 % of pixels |
| Road content | at least 80 road pixels, and road ratio at most 0.65 |

**Split:** whole scenes are assigned to `train / val / test` with a seeded shuffle (seed 42):

| Number of scenes `n` | val | test | train |
| --- | :---: | :---: | :---: |
| 1-5 | 1 (if `n >= 2`) | `n - train - val` | `max(1, n - 2)` |
| 6-12 | 2 | 2 | `n - 4` |
| 13+ | `max(2, 15 %)` | `max(2, 15 %)` | rest |

Outputs:

- `data/processed/patches/<split>/tileNNNNN.npz` containing `image` (`float32`, `(C, 256, 256)`) and `mask` (`uint8`, `(256, 256)`)
- `data/processed/patches/tiles_metadata.csv` (tile id, scene, split, `y`, `x`, road pixels, road ratio, channel count, CRS, scene bounds)

Before re-running with a different scene set or different thresholds, delete `data/processed/patches/` so old tiles do not remain.

---

## Data format

**Channels per patch** = 4 optical + 2 per SAR product.

| Index | Content |
| :---: | --- |
| 0-3 | LISS-III Green, Red, NIR, SWIR (raw DN as `float32`, not normalized) |
| 4-5 | SAR product 1: VV, VH |
| 6-7 | SAR product 2: VV, VH |

Because step 7 only selects scenes that have both `sar_1` and `sar_2`, patches from a normal run have **8 channels**. Check `num_channels` in `tiles_metadata.csv`. SAR product 1 and 2 are the `.SAFE` folders of the scene in sorted-name order.

**SAR values:** `10*log10` of the Sentinel-1 GRD values, stored as `float32`. Invalid pixels are `-9999`.

Loading a patch:

```python
import numpy as np

d = np.load("data/processed/patches/train/tile00001.npz")
image, mask = d["image"], d["mask"]     # (C, 256, 256) float32, (256, 256) uint8
```

---

## Visualization tools

These are plain scripts with a **hardcoded** scene or split at the top. Edit the constants before running.

| Script | What to edit | Output |
| --- | --- | --- |
| `src/visualization/view_scene.py` | `SCENE_ID` (default `2025-11-25_Amaravati`), `DOWNSAMPLE_FACTOR` | `previews/<scene>_preview.png` (cached) |
| `src/visualization/view_mask_validation.py` | `SCENE_ID` (default `2025-11-17_Amaravati`, or `None` for a random scene) | on-screen only |
| `src/visualization/view_patches.py` | `SPLIT`, `NUM_SAMPLES` | `previews/patch_inspection_<split>.png` |

The scripts that call `plt.show()` need a display.

---

## Configuration reference (what is dynamic and what is not)

There are no command-line arguments and no config file besides `configs/paths.py`; settings are constants at the top of each script.

**Dynamic (derived at run time):**

- All paths are relative to the repo root (`configs/paths.py` resolves it from `__file__`), so the project can live anywhere.
- Scenes are discovered from folder contents (`data/raw/optical/*`, `data/raw/sar/*`, `data/processed/scenes/*`).
- SAR search dates, scene metadata and footprints come from the folder names and the rasters.
- Scene selection comes from `quality_report.csv`; the train/val/test assignment is computed from however many scenes qualify.
- Mask shape, CRS and transform come from each scene's `optical_24m.tif`; the CRS is inherited from the optical product (not forced to one EPSG code).
- The channel count is computed from the files found and recorded in the metadata.

**Static (hardcoded; edit the source to change):**

| Setting | Where | Value |
| --- | --- | --- |
| Scene naming convention | `extract_footprints.py` (date from first 10 characters) | `YYYY-MM-DD_Location` |
| OSM file name | `configs/paths.py` `PBF_PATH` | `south-india-latest.osm.pbf` |
| Target resolution | `preprocess_all_scenes.py` `TARGET_RES` | 24 m |
| SAR search window, count, overlap | `download_matching_sar.py` | +/-12 days, 2 scenes, 25 % |
| Number of SAR products in QC | `quality_check.py` | exactly `sar_1` and `sar_2` |
| Scene quality threshold | `create_masks_geofabrik.py`, `extract_patches.py`, `verify_dataset_readiness.py` (three scripts) | 50 % |
| Road classes and buffers | `create_masks_geofabrik.py` | see step 9 |
| Patch size, stride, gates, seed | `extract_patches.py` | 256, 128, see step 10, 42 |
| Band order | `preprocess_all_scenes.py`, viewers | Green, Red, NIR, SWIR |
| Viewer targets | the three visualization scripts | fixed scene IDs / split |

---

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `EARTHDATA_USER or EARTHDATA_PASS not set in .env` | `.env` missing, or run outside the repo tree |
| Wrong SAR dates / no SAR found | optical zip not renamed to `YYYY-MM-DD_Location.zip` |
| `IndexError` in `process_one_sar` | a downloaded SAR scene has no VV or VH measurement file (single-polarization product) |
| `Less than 4 bands found` | optical scene folder lacks `BAND*.tif` files, or they sit in a sub-folder |
| `Geofabrik extract missing at ...` | PBF not saved under the hardcoded name/path (step 8) |
| `Selected 0 scenes` / empty split | no scene reached 50 % valid overlap for **both** SARs (step 7) |
| Viewer: `Optical or Mask file not found` | the hardcoded `SCENE_ID` does not exist in your data |
