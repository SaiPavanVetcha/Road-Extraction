from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import rasterio

# ============================================================
SCENE_ID = "2025-11-17_Amaravati"
# ============================================================

SCENE_DIR = Path(f"processed/{SCENE_ID}")
OPTICAL_PATH = SCENE_DIR / "optical_24m.tif"

# Find all generated SAR files dynamically (e.g. sar_1_24m.tif, sar_2_24m.tif)
SAR_PATHS = sorted(list(SCENE_DIR.glob("sar_*_24m.tif")))


def robust_stretch(band, lower=2, upper=98):
    """Normalizes array between 0 and 1 using percentiles."""
    band = band.astype(np.float32)
    valid = band[np.isfinite(band) & (band > 0)]
    if valid.size < 100:
        return np.zeros_like(band)
    p_low, p_high = np.percentile(valid, (lower, upper))
    return np.clip((band - p_low) / (p_high - p_low + 1e-6), 0, 1)


# ---------- 1. Read & Print Optical ----------
print(f"=== Loading Scene: {SCENE_ID} ===")
if not OPTICAL_PATH.exists():
    raise FileNotFoundError(f"Optical file not found: {OPTICAL_PATH}")

with rasterio.open(OPTICAL_PATH) as src:
    print(f"Optical Shape : {src.shape}")
    print(f"Optical CRS   : {src.crs}")
    print(f"Optical Bounds: {src.bounds}")

    green = src.read(1)
    red = src.read(2)
    nir = src.read(3)

rgb = np.stack(
    [robust_stretch(nir), robust_stretch(red), robust_stretch(green)], axis=-1
)

# ---------- 2. Read All SAR Files ----------
if not SAR_PATHS:
    raise FileNotFoundError(f"No SAR files found in {SCENE_DIR}")

print(f"\nFound {len(SAR_PATHS)} SAR file(s) for visualization")

sar_data = []
for sar_path in SAR_PATHS:
    print(f"\nLoading SAR: {sar_path.name}")
    with rasterio.open(sar_path) as src:
        vv = src.read(1)
        vh = src.read(2)

    vv_masked = np.ma.masked_equal(vv, -9999)
    vh_masked = np.ma.masked_equal(vh, -9999)

    valid_vv = vv_masked.compressed()
    valid_vh = vh_masked.compressed()

    if valid_vv.size > 0:
        print(f"  VV Range (dB) : {valid_vv.min():.1f} → {valid_vv.max():.1f}")
        vmin_vv, vmax_vv = np.percentile(valid_vv, (2, 98))
    else:
        vmin_vv, vmax_vv = -25.0, 0.0

    if valid_vh.size > 0:
        print(f"  VH Range (dB) : {valid_vh.min():.1f} → {valid_vh.max():.1f}")
        vmin_vh, vmax_vh = np.percentile(valid_vh, (2, 98))
    else:
        vmin_vh, vmax_vh = -30.0, -5.0

    sar_data.append({
        "name": sar_path.stem,
        "vv": vv_masked,
        "vh": vh_masked,
        "vv_bounds": (vmin_vv, vmax_vv),
        "vh_bounds": (vmin_vh, vmax_vh),
    })

# ---------- 3. Dynamic Plot Grid ----------
# Total subplots = 1 Optical + (2 channels * N SAR files)
total_plots = 1 + (len(sar_data) * 2)
fig, axes = plt.subplots(1, total_plots, figsize=(5 * total_plots, 5))

if total_plots == 1:
    axes = [axes]

# Plot Optical
axes[0].imshow(rgb)
axes[0].set_title(f"{SCENE_ID}\nOptical (NIR-R-G)", fontsize=11)
axes[0].axis("off")

# Plot each SAR file (VV and VH)
col_idx = 1
for sar in sar_data:
    # VV Channel
    axes[col_idx].imshow(
        sar["vv"],
        cmap="gray",
        vmin=sar["vv_bounds"][0],
        vmax=sar["vv_bounds"][1],
    )
    axes[col_idx].set_title(f"{sar['name']}\nVV (dB)", fontsize=11)
    axes[col_idx].axis("off")
    col_idx += 1

    # VH Channel
    axes[col_idx].imshow(
        sar["vh"],
        cmap="gray",
        vmin=sar["vh_bounds"][0],
        vmax=sar["vh_bounds"][1],
    )
    axes[col_idx].set_title(f"{sar['name']}\nVH (dB)", fontsize=11)
    axes[col_idx].axis("off")
    col_idx += 1

plt.tight_layout()
plt.show()