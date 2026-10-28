import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
import random

# ============================================================
# CONFIGURATION
# ============================================================
PROCESSED_ROOT = Path("processed")
MASK_ROOT = Path("processed/masks")

# Put the scene you want to check here
SCENE_ID = "2025-11-17_Amaravati"      # ← change this

# Or set to None to pick a random scene
# SCENE_ID = None
# ============================================================

colors = [
    "#000000",  # 0 - Background
    "#1f77b4",  # 1 - Local / Minor
    "#ffbb78",  # 2 - Secondary
    "#d62728",  # 3 - Major
]
cmap = ListedColormap(colors)
labels = ["Background", "Local", "Secondary", "Major"]

def robust_stretch(band, lower=2, upper=98):
    band = band.astype(np.float32)
    valid = band[np.isfinite(band) & (band > 0)]
    if valid.size < 50:
        return np.zeros_like(band)
    p_low, p_high = np.percentile(valid, (lower, upper))
    return np.clip((band - p_low) / (p_high - p_low + 1e-6), 0, 1)


def main():
    global SCENE_ID

    if SCENE_ID is None:
        mask_files = list(MASK_ROOT.glob("mask_*.tif"))
        if not mask_files:
            print("No masks found!")
            return
        SCENE_ID = random.choice(mask_files).stem.replace("mask_", "")
        print(f"Randomly selected: {SCENE_ID}")

    opt_path = PROCESSED_ROOT / SCENE_ID / "optical_24m.tif"
    mask_path = MASK_ROOT / f"mask_{SCENE_ID}.tif"

    if not opt_path.exists() or not mask_path.exists():
        print("Optical or Mask file not found!")
        return

    # Read Optical
    with rasterio.open(opt_path) as src:
        green = src.read(1)
        red   = src.read(2)
        nir   = src.read(3)
        print(f"Optical shape: {src.shape}")

    rgb = np.stack([
        robust_stretch(nir),
        robust_stretch(red),
        robust_stretch(green)
    ], axis=-1)

    # Read Mask
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        print(f"Mask shape   : {src.shape}")

    unique, counts = np.unique(mask, return_counts=True)
    print("\nClass distribution:")
    for u, c in zip(unique, counts):
        print(f"  {labels[u] if u < len(labels) else u}: {c:,} pixels")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    axes[0].imshow(rgb)
    axes[0].set_title(f"{SCENE_ID}\nOptical (NIR-R-G)", fontsize=14)
    axes[0].axis("off")

    im = axes[1].imshow(mask, cmap=cmap, vmin=0, vmax=3)
    axes[1].set_title("Multi-class Road Mask", fontsize=14)
    axes[1].axis("off")

    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(labels)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()