import argparse
import shutil
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

from configs.paths import DOWNLOADS_DIR, RAW_OPTICAL_DIR

# ============================================================
# CONFIGURATION
# ============================================================

# Folder where all the .zip files are currently present
ZIP_FOLDER = DOWNLOADS_DIR

# Final destination
OUTPUT_FOLDER = RAW_OPTICAL_DIR
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# ============================================================


def organize(force=False):
    zip_files = sorted(list(Path(ZIP_FOLDER).glob("*.zip")))
    print(f"Found {len(zip_files)} zip files\n")

    for zip_path in zip_files:
        # Get clean name without .zip
        folder_name = zip_path.stem  # e.g. 2025-11-17_Amaravati

        print(f"Processing: {folder_name}")

        final_path = Path(OUTPUT_FOLDER) / folder_name
        temp_dir = Path(ZIP_FOLDER) / f"temp_{folder_name}"

        # Remove any stale temp_* folder before extracting
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

        if not force and final_path.exists() and list(final_path.glob("BAND*.tif")):
            print(f"  [Skipped] {folder_name}: already organized (use --force to rebuild)\n")
            continue

        temp_dir.mkdir(parents=True, exist_ok=True)

        # Extract
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        # Find the actual data folder inside (usually one main folder)
        extracted_items = list(temp_dir.iterdir())

        if final_path.exists():
            shutil.rmtree(final_path)

        # If there is only one folder inside, move its contents
        if len(extracted_items) == 1 and extracted_items[0].is_dir():
            shutil.move(str(extracted_items[0]), str(final_path))
        else:
            final_path.mkdir(parents=True, exist_ok=True)
            for item in extracted_items:
                shutil.move(str(item), str(final_path / item.name))

        # Clean temp
        shutil.rmtree(temp_dir, ignore_errors=True)

        print(f"  → Saved to: {final_path}\n")

    print("=== All done ===")


def main():
    parser = argparse.ArgumentParser(
        description="Organize downloaded optical zip files into raw directories."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction and overwrite existing organized scenes.",
    )
    args = parser.parse_args()
    organize(force=args.force)


if __name__ == "__main__":
    main()
