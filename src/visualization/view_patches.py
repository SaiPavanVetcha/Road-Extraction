import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# Handle Windows console encoding for Unicode checkmarks and symbols
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

from configs.paths import PATCHES_DIR, PREVIEWS_DIR
from src.utils.viz import process_sar_channel, robust_stretch

PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

# 4 Discrete classes: Background (0), Local (1), Secondary (2), Major (3)
MASK_COLORS = ["#000000", "#1f77b4", "#ffbb78", "#d62728"]
MASK_CMP = ListedColormap(MASK_COLORS)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect random extracted patches."
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to inspect (default: train)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=2,
        help="Number of random tiles to inspect (default: 2)",
    )
    args = parser.parse_args()

    split_dir = PATCHES_DIR / args.split
    if not split_dir.exists():
        print(f"❌ Patch directory missing at: {split_dir}")
        print("   Please run src/data_prep/extract_patches.py first.")
        return

    patch_files = sorted(list(split_dir.glob("*.npz")))
    if not patch_files:
        print(f"❌ No .npz patches found in {split_dir}")
        return

    print(f"Found {len(patch_files):,} patches in [{args.split.upper()}] set.")
    num_samples = min(max(1, args.n), len(patch_files))
    selected = random.sample(patch_files, num_samples)

    # Inspect first tile to determine channel layout
    sample_data = np.load(selected[0])
    sample_img = sample_data["image"]
    C = sample_img.shape[0]

    sar_indices = list(range(4, C - 1)) if C > 5 else []
    has_validity_mask = C >= 5

    # Columns: Optical, all SAR channels, validity mask (if present), road mask
    total_cols = 1 + len(sar_indices) + (1 if has_validity_mask else 0) + 1

    fig, axes = plt.subplots(
        len(selected),
        total_cols,
        figsize=(3.5 * total_cols, 4 * len(selected)),
        squeeze=False,
    )

    for row_idx, p_file in enumerate(selected):
        data = np.load(p_file)
        img = data["image"]  # Shape: (C, 256, 256)
        mask = data["mask"]  # Shape: (256, 256)

        col_idx = 0

        # 1. Optical False-Color RGB (NIR=index 2, Red=index 1, Green=index 0)
        nir, red, green = img[2], img[1], img[0]
        rgb = np.stack(
            [robust_stretch(nir), robust_stretch(red), robust_stretch(green)],
            axis=-1,
        )
        axes[row_idx][col_idx].imshow(rgb)
        axes[row_idx][col_idx].set_title(
            f"{p_file.stem}\nOptical (NIR-R-G)", fontsize=9
        )
        axes[row_idx][col_idx].axis("off")
        col_idx += 1

        # 2. All SAR Channels
        for s_pos, ch in enumerate(sar_indices):
            sar_norm = process_sar_channel(img[ch])
            sar_num = (s_pos // 2) + 1
            pol = "VV" if (s_pos % 2 == 0) else "VH"
            axes[row_idx][col_idx].imshow(sar_norm, cmap="gray", vmin=0, vmax=1)
            axes[row_idx][col_idx].set_title(
                f"SAR {sar_num} ({pol} dB)", fontsize=9
            )
            axes[row_idx][col_idx].axis("off")
            col_idx += 1

        # 3. Validity Mask (last channel)
        if has_validity_mask:
            val_ch = img[-1]
            axes[row_idx][col_idx].imshow(
                val_ch, cmap="gray", vmin=0.0, vmax=1.0
            )
            axes[row_idx][col_idx].set_title("Validity Mask", fontsize=9)
            axes[row_idx][col_idx].axis("off")
            col_idx += 1

        # 4. Target Road Mask
        axes[row_idx][col_idx].imshow(mask, cmap=MASK_CMP, vmin=0, vmax=3)
        axes[row_idx][col_idx].set_title(
            f"Road Mask Target\n(Road px: {int(np.sum(mask > 0)):,})",
            fontsize=9,
        )
        axes[row_idx][col_idx].axis("off")

    out_preview = PREVIEWS_DIR / f"patch_inspection_{args.split}.png"
    plt.tight_layout()
    plt.savefig(out_preview, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved patch inspection grid to → {out_preview}")
    plt.show()


if __name__ == "__main__":
    main()
