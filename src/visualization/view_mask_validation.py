import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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

from configs.paths import MASK_DIR, PREVIEWS_DIR, PROCESSED_SCENES_DIR
from src.utils.viz import robust_stretch

PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

colors = [
    "#000000",  # 0 - Background
    "#1f77b4",  # 1 - Local / Minor
    "#ffbb78",  # 2 - Secondary
    "#d62728",  # 3 - Major
]
cmap = ListedColormap(colors)
labels = ["Background", "Local", "Secondary", "Major"]


def main():
    parser = argparse.ArgumentParser(
        description="Visualize optical imagery alongside multi-class road mask."
    )
    parser.add_argument(
        "scene_id",
        nargs="?",
        default=None,
        help="Scene ID to inspect (default: first available in data/processed/scenes)",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=4,
        help="Downsample factor for rendering (default: 4)",
    )
    args = parser.parse_args()

    scene_id = args.scene_id
    if scene_id is None:
        available_scenes = (
            sorted([d.name for d in PROCESSED_SCENES_DIR.iterdir() if d.is_dir()])
            if PROCESSED_SCENES_DIR.exists()
            else []
        )
        if not available_scenes:
            print(f"No processed scenes found in {PROCESSED_SCENES_DIR}")
            return
        scene_id = available_scenes[0]
        print(f"No scene specified; defaulting to: {scene_id}")

    opt_path = PROCESSED_SCENES_DIR / scene_id / "optical_24m.tif"
    mask_path = MASK_DIR / f"mask_{scene_id}.tif"

    if not opt_path.exists() or not mask_path.exists():
        print(f"Optical or Mask file not found for scene {scene_id}!")
        return

    downsample = max(1, args.downsample)
    with rasterio.open(opt_path) as src:
        out_h, out_w = src.height // downsample, src.width // downsample
        print(f"Original shape   : {src.shape}")
        print(f"Downsampled shape: ({out_h}, {out_w})")
        green = src.read(1, out_shape=(out_h, out_w))
        red = src.read(2, out_shape=(out_h, out_w))
        nir = src.read(3, out_shape=(out_h, out_w))

    rgb = np.stack(
        [robust_stretch(nir), robust_stretch(red), robust_stretch(green)],
        axis=-1,
    )

    with rasterio.open(mask_path) as src:
        mask = src.read(1, out_shape=(out_h, out_w))

    unique, counts = np.unique(mask, return_counts=True)
    print("\nClass distribution:")
    for u, c in zip(unique, counts):
        print(f"  {labels[u] if u < len(labels) else u}: {c:,} pixels")

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    axes[0].imshow(rgb)
    axes[0].set_title(f"{scene_id}\nOptical (NIR-R-G)", fontsize=14)
    axes[0].axis("off")

    im = axes[1].imshow(mask, cmap=cmap, vmin=0, vmax=3)
    axes[1].set_title("Multi-class Road Mask", fontsize=14)
    axes[1].axis("off")

    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(labels)

    plt.tight_layout()
    out_preview = PREVIEWS_DIR / f"{scene_id}_mask_validation.png"
    plt.savefig(out_preview, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved mask validation preview to → {out_preview}")
    plt.show()


if __name__ == "__main__":
    main()
