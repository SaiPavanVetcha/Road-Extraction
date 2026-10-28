import sys
import random
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.paths import PATCHES_DIR, PROCESSED_DIR

# ============================================================
# CONFIGURATION
# ============================================================
SPLIT = "train"        # "train", "val", or "test"
NUM_SAMPLES = 2       # Number of random tiles to inspect
# ============================================================

PREVIEWS_DIR = PROCESSED_DIR / "previews"
PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

# 4 Discrete classes: Background (0), Local (1), Secondary (2), Major (3)
MASK_COLORS = ["#000000", "#1f77b4", "#ffbb78", "#d62728"]
MASK_CMP = ListedColormap(MASK_COLORS)


def robust_stretch(band, lower=2, upper=98):
    """Normalizes optical channel between 0.0 and 1.0 using percentiles."""
    band = band.astype(np.float32)
    valid = band[np.isfinite(band) & (band > 0)]
    if valid.size < 50:
        return np.zeros_like(band, dtype=np.float32)
    p_low, p_high = np.percentile(valid, (lower, upper))
    return np.clip((band - p_low) / (p_high - p_low + 1e-6), 0, 1).astype(np.float32)


def process_sar(sar_band):
    """Normalizes SAR decibel channel for display."""
    sar_band = sar_band.astype(np.float32)
    valid_mask = sar_band > -9990
    if not np.any(valid_mask):
        return np.zeros_like(sar_band, dtype=np.float32)

    if np.max(sar_band[valid_mask]) > 5.0:
        sar_band[valid_mask] = 10.0 * np.log10(np.maximum(sar_band[valid_mask], 1e-6))

    valid_vals = sar_band[valid_mask]
    p2, p98 = np.percentile(valid_vals, (2, 98))
    norm = np.clip((sar_band - p2) / (p98 - p2 + 1e-6), 0, 1)
    norm[~valid_mask] = 0
    return norm.astype(np.float32)


def main():
    split_dir = PATCHES_DIR / SPLIT
    if not split_dir.exists():
        print(f"❌ Patch directory missing at: {split_dir}")
        print("   Please run src/dataset/extract_patches.py first.")
        return

    patch_files = list(split_dir.glob("*.npz"))
    if not patch_files:
        print(f"❌ No .npz patches found in {split_dir}")
        return

    print(f"Found {len(patch_files):,} patches in [{SPLIT.upper()}] set.")
    selected = random.sample(patch_files, min(NUM_SAMPLES, len(patch_files)))

    fig, axes = plt.subplots(len(selected), 4, figsize=(16, 4 * len(selected)))
    if len(selected) == 1:
        axes = [axes]

    for idx, p_file in enumerate(selected):
        data = np.load(p_file)
        img = data["image"]  # Shape: (C, 256, 256)
        mask = data["mask"]  # Shape: (256, 256)

        # 1. Optical False-Color RGB (NIR=index 2, Red=index 1, Green=index 0)
        nir, red, green = img[2], img[1], img[0]
        rgb = np.stack(
            [robust_stretch(nir), robust_stretch(red), robust_stretch(green)],
            axis=-1,
        )

        # 2. SAR VV & VH Channels (Assuming indices 4 and 5)
        vv_norm = process_sar(img[4])
        vh_norm = process_sar(img[5])

        # Plot Optical False-Color
        axes[idx][0].imshow(rgb)
        axes[idx][0].set_title(f"{p_file.stem}\nOptical (NIR-R-G)", fontsize=10)
        axes[idx][0].axis("off")

        # Plot SAR VV
        axes[idx][1].imshow(vv_norm, cmap="gray")
        axes[idx][1].set_title("SAR (VV dB)", fontsize=10)
        axes[idx][1].axis("off")

        # Plot SAR VH
        axes[idx][2].imshow(vh_norm, cmap="gray")
        axes[idx][2].set_title("SAR (VH dB)", fontsize=10)
        axes[idx][2].axis("off")

        # Plot Target Road Mask
        im = axes[idx][3].imshow(mask, cmap=MASK_CMP, vmin=0, vmax=3)
        axes[idx][3].set_title(f"Road Mask Target\n(Pixels > 0: {np.sum(mask > 0):,})", fontsize=10)
        axes[idx][3].axis("off")

    out_preview = PREVIEWS_DIR / f"patch_inspection_{SPLIT}.png"
    plt.tight_layout()
    plt.savefig(out_preview, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved patch inspection grid to → {out_preview}")
    plt.show()


if __name__ == "__main__":
    main()