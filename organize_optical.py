import os
import zipfile
import shutil
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Folder where all the .zip files are currently present
ZIP_FOLDER = BASE_DIR / "downloads"

# Final destination
OUTPUT_FOLDER = BASE_DIR / "raw" / "optical"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# ============================================================

def organize():
    zip_files = list(Path(ZIP_FOLDER).glob("*.zip"))
    print(f"Found {len(zip_files)} zip files\n")

    for zip_path in zip_files:
        # Get clean name without .zip
        folder_name = zip_path.stem   # e.g. 2025-11-17_Amaravati

        print(f"Processing: {folder_name}")

        # Temporary extract location
        temp_dir = Path(ZIP_FOLDER) / f"temp_{folder_name}"
        temp_dir.mkdir(exist_ok=True)

        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Find the actual data folder inside (usually one main folder)
        extracted_items = list(temp_dir.iterdir())
        
        # Final destination
        final_path = Path(OUTPUT_FOLDER) / folder_name

        if final_path.exists():
            shutil.rmtree(final_path)

        # If there is only one folder inside, move its contents
        if len(extracted_items) == 1 and extracted_items[0].is_dir():
            shutil.move(str(extracted_items[0]), str(final_path))
        else:
            # Move everything
            final_path.mkdir(exist_ok=True)
            for item in extracted_items:
                shutil.move(str(item), str(final_path / item.name))

        # Clean temp
        shutil.rmtree(temp_dir, ignore_errors=True)

        print(f"  → Saved to: {final_path}\n")

    print("=== All done ===")

if __name__ == "__main__":
    organize()