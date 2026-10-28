import argparse
import sys
import zipfile
from pathlib import Path

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

from configs.paths import RAW_SAR_DIR

# ============================================================
# CONFIGURATION
# ============================================================

SAR_ROOT = RAW_SAR_DIR.resolve()  # Get full absolute path
# ============================================================


def get_windows_long_path(path: Path) -> Path:
    r"""Prepends \\?\ to Windows absolute paths to bypass the 260-character MAX_PATH limit."""
    abs_str = str(path.resolve())
    if sys.platform == "win32" and not abs_str.startswith("\\\\?\\"):
        return Path(f"\\\\?\\{abs_str}")
    return path


def organize_sar(force=False):
    # Find all scene folders (e.g. 2025-11-17_Amaravati)
    if not SAR_ROOT.exists():
        print(f"SAR root directory does not exist: {SAR_ROOT}")
        return

    scene_folders = sorted([f for f in SAR_ROOT.iterdir() if f.is_dir()])
    print(f"Found {len(scene_folders)} optical-matched folders\n")

    for scene_dir in scene_folders:
        print(f"Processing: {scene_dir.name}")

        # Find all zip files inside this folder
        zip_files = sorted(list(scene_dir.glob("*.zip")))

        if not zip_files:
            print("  → No zip files found\n")
            continue

        for zip_path in zip_files:
            safe_name = (
                zip_path.stem
                if zip_path.stem.endswith(".SAFE")
                else f"{zip_path.stem}.SAFE"
            )
            matching_safe = scene_dir / safe_name

            if not force and matching_safe.exists() and matching_safe.is_dir():
                print(f"  [Skipped] Already extracted: {safe_name}")
                continue

            print(f"  Unzipping: {zip_path.name}")

            # Extract directly to scene_dir (Sentinel zips contain their own .SAFE root folder)
            target_extract_dir = get_windows_long_path(scene_dir)

            try:
                # Open zip with long path support
                with zipfile.ZipFile(
                    get_windows_long_path(zip_path), "r"
                ) as zf:
                    zf.extractall(target_extract_dir)

                print("    → Successfully extracted .SAFE product")

            except Exception as e:
                print(f"    ❌ Extraction failed: {e}")

        print()

    print("=== SAR organization completed ===")


def main():
    parser = argparse.ArgumentParser(
        description="Extract raw Sentinel-1 SAR zips into .SAFE folders."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction even if matching .SAFE folders exist.",
    )
    args = parser.parse_args()
    organize_sar(force=args.force)


if __name__ == "__main__":
    main()
