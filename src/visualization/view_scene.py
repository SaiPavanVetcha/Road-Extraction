import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import rasterio

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

from configs.paths import PROCESSED_DIR, PROCESSED_SCENES_DIR

# ============================================================
SCENE_ID = "2025-11-25_Amaravati"
DOWNSAMPLE_FACTOR = 4  # Downsample x4 for safe rendering (~1787x1937)
OVERWRITE_EXISTING = False  # Set to True if you want to force re-rendering

# Previews Directory
PREVIEWS_DIR = PROCESSED_DIR / "previews"
PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
# ============================================================

SCENE_DIR = PROCESSED_SCENES_DIR / SCENE_ID
OPTICAL_PATH = SCENE_DIR / "optical_24m.tif"
SAR_PATHS = sorted(list(SCENE_DIR.glob("sar_*_24m.tif")))


def robust_stretch(band, lower=2, upper=98):
    """Normalizes array between 0.0 and 1.0 float32 using percentiles."""
    band = band.astype(np.float32)
    valid = band[np.isfinite(band) & (band > 0)]
    if valid.size < 100:
        return np.zeros_like(band, dtype=np.float32)
    p_low, p_high = np.percentile(valid, (lower, upper))
    return np.clip((band - p_low) / (p_high - p_low + 1e-6), 0, 1).astype(
        np.float32
    )


def process_sar_channel(sar_arr):
    """Ensures SAR channel is in proper dB scale and normalizes for display."""
    sar_arr = sar_arr.astype(np.float32)
    valid_mask = sar_arr > -9990

    if not np.any(valid_mask):
        return np.zeros_like(sar_arr, dtype=np.float32)

    if np.max(sar_arr[valid_mask]) > 5.0:
        sar_arr[valid_mask] = 10.0 * np.log10(
            np.maximum(sar_arr[valid_mask], 1e-6)
        )

    valid_vals = sar_arr[valid_mask]
    p2, p98 = np.percentile(valid_vals, (2, 98))
    sar_norm = np.clip((sar_arr - p2) / (p98 - p2 + 1e-6), 0, 1)
    sar_norm[~valid_mask] = 0
    return sar_norm.astype(np.float32)

def main():
    out_preview = PREVIEWS_DIR / f"{SCENE_ID}_preview.png"

    # If preview already exists, load and display it directly without re-reading rasters
    if out_preview.exists() and not OVERWRITE_EXISTING:
        print(f"✓ Cached preview found at → {out_preview}")
        print("  Displaying existing preview window...\n")

        # Read saved PNG image and display
        img = plt.imread(out_preview)
        fig, ax = plt.subplots(figsize=(16, 5))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Cached Scene Preview: {SCENE_ID}", fontsize=12)
        plt.tight_layout()
        plt.show()
        return

    print(f"=== Loading Scene: {SCENE_ID} ===")
    if not OPTICAL_PATH.exists():
        raise FileNotFoundError(f"Optical file not found: {OPTICAL_PATH}")

    # 1. Read Optical with Downsampling
    with rasterio.open(OPTICAL_PATH) as src:
        out_h, out_w = src.height // DOWNSAMPLE_FACTOR, src.width // DOWNSAMPLE_FACTOR
        print(f"Original Shape   : {src.shape}")
        print(f"Downsampled Shape: ({out_h}, {out_w})")

        green = src.read(1, out_shape=(out_h, out_w))
        red = src.read(2, out_shape=(out_h, out_w))
        nir = src.read(3, out_shape=(out_h, out_w))

    rgb = np.stack(
        [robust_stretch(nir), robust_stretch(red), robust_stretch(green)],
        axis=-1,
    ).astype(np.float32)

    # 2. Read SAR Files with Downsampling
    if not SAR_PATHS:
        raise FileNotFoundError(f"No SAR files found in {SCENE_DIR}")

    print(f"\nFound {len(SAR_PATHS)} SAR file(s) for visualization")

    sar_data = []
    for sar_path in SAR_PATHS:
        print(f"Loading SAR: {sar_path.name}")
        with rasterio.open(sar_path) as src:
            vv = src.read(1, out_shape=(out_h, out_w))
            vh = src.read(2, out_shape=(out_h, out_w))

        vv_norm = process_sar_channel(vv)
        vh_norm = process_sar_channel(vh)

        sar_data.append(
            {
                "name": sar_path.stem,
                "vv": vv_norm,
                "vh": vh_norm,
            }
        )

    # 3. Render Side-by-Side Plot Grid
    total_plots = 1 + (len(sar_data) * 2)
    fig, axes = plt.subplots(1, total_plots, figsize=(4 * total_plots, 4.5))

    if total_plots == 1:
        axes = [axes]

    # Plot Optical False-Color
    axes[0].imshow(rgb)
    axes[0].set_title(f"{SCENE_ID}\nOptical (NIR-R-G)", fontsize=10)
    axes[0].axis("off")

    # Plot SAR VV and VH
    col_idx = 1
    for sar in sar_data:
        axes[col_idx].imshow(sar["vv"], cmap="gray", vmin=0, vmax=1)
        axes[col_idx].set_title(f"{sar['name']}\nVV Channel", fontsize=10)
        axes[col_idx].axis("off")
        col_idx += 1

        axes[col_idx].imshow(sar["vh"], cmap="gray", vmin=0, vmax=1)
        axes[col_idx].set_title(f"{sar['name']}\nVH Channel", fontsize=10)
        axes[col_idx].axis("off")
        col_idx += 1

    # Save to dedicated previews directory
    plt.tight_layout()
    plt.savefig(out_preview, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved preview to → {out_preview}")
    plt.show()


if __name__ == "__main__":
    main()