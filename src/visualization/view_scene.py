import argparse
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

from configs.paths import PREVIEWS_DIR, PROCESSED_SCENES_DIR
from src.utils.viz import process_sar_channel, robust_stretch

PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize optical false-color and SAR channels for a scene."
    )
    parser.add_argument(
        "scene_id",
        nargs="?",
        default=None,
        help="Scene ID to visualize (default: first available in data/processed/scenes)",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=4,
        help="Downsample factor for rendering (default: 4)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-rendering even if cached preview exists",
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

    scene_dir = PROCESSED_SCENES_DIR / scene_id
    optical_path = scene_dir / "optical_24m.tif"
    sar_paths = sorted(list(scene_dir.glob("sar_*_24m.tif")))

    out_preview = PREVIEWS_DIR / f"{scene_id}_preview.png"

    # If preview already exists and not forced, display cached
    if out_preview.exists() and not args.force:
        print(f"✓ Cached preview found at → {out_preview}")
        print("  Displaying existing preview window...\n")
        img = plt.imread(out_preview)
        fig, ax = plt.subplots(figsize=(16, 5))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Cached Scene Preview: {scene_id}", fontsize=12)
        plt.tight_layout()
        plt.show()
        return

    print(f"=== Loading Scene: {scene_id} ===")
    if not optical_path.exists():
        raise FileNotFoundError(f"Optical file not found: {optical_path}")

    # 1. Read Optical with Downsampling
    downsample = max(1, args.downsample)
    with rasterio.open(optical_path) as src:
        out_h, out_w = src.height // downsample, src.width // downsample
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
    if not sar_paths:
        raise FileNotFoundError(f"No SAR files found in {scene_dir}")

    print(f"\nFound {len(sar_paths)} SAR file(s) for visualization")

    sar_data = []
    for sar_path in sar_paths:
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
    axes[0].set_title(f"{scene_id}\nOptical (NIR-R-G)", fontsize=10)
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

    # Save to previews directory
    plt.tight_layout()
    plt.savefig(out_preview, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved preview to → {out_preview}")
    plt.show()


if __name__ == "__main__":
    main()
