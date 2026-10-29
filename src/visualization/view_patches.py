import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.paths import PATCHES_DIR
from configs.settings import CHANNELS_PER_TILE
from src.utils.viz import robust_stretch

# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Visualize random extracted patches")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=4, help="Number of patches to show")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    split_dir = PATCHES_DIR / args.split
    if not split_dir.exists():
        print(f"No patches found in {split_dir}")
        print("Run: python -m src.data_prep.extract_patches")
        return

    files = sorted(split_dir.glob("tile*.npz"))
    if not files:
        print(f"No .npz tiles in {split_dir}")
        return

    rng = random.Random(args.seed)
    selected = rng.sample(files, min(args.n, len(files)))

    colors = ["#000000", "#1f77b4", "#ffbb78", "#d62728"]
    cmap = ListedColormap(colors)

    n = len(selected)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, fpath in enumerate(selected):
        data = np.load(fpath)
        image = data["image"]  # (7, 256, 256)
        mask = data["mask"]

        assert image.shape[0] == CHANNELS_PER_TILE, (
            f"{fpath.name}: expected {CHANNELS_PER_TILE} channels, got {image.shape[0]}"
        )

        # Optical false colour (NIR-R-G)
        nir, red, green = image[2], image[1], image[0]
        rgb = np.stack([
            robust_stretch(nir),
            robust_stretch(red),
            robust_stretch(green),
        ], axis=-1)

        vv = image[4]
        vh = image[5]
        validity = image[6]

        road_pixels = int(np.sum(mask > 0))

        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"{fpath.stem}\nOptical (NIR-R-G)", fontsize=10)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(vv, cmap="gray")
        axes[row, 1].set_title("SAR VV", fontsize=10)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(vh, cmap="gray")
        axes[row, 2].set_title("SAR VH", fontsize=10)
        axes[row, 2].axis("off")

        axes[row, 3].imshow(mask, cmap=cmap, vmin=0, vmax=3)
        axes[row, 3].set_title(f"Mask (roads: {road_pixels})", fontsize=10)
        axes[row, 3].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()