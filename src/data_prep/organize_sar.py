import os
import sys
import shutil
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
    if os.name == "nt" and not abs_str.startswith("\\\\?\\"):
        return Path(f"\\\\?\\{abs_str}")
    return path


def organize_sar():
    # Find all scene folders (e.g. 2025-11-17_Amaravati)
    scene_folders = [f for f in SAR_ROOT.iterdir() if f.is_dir()]
    print(f"Found {len(scene_folders)} optical-matched folders\n")

    for scene_dir in scene_folders:
        print(f"Processing: {scene_dir.name}")

        # Find all zip files inside this folder
        zip_files = list(scene_dir.glob("*.zip"))

        if not zip_files:
            print("  → No zip files found")
            continue

        for zip_path in zip_files:
            print(f"  Unzipping: {zip_path.name}")

            # Extract directly to scene_dir (Sentinel zips contain their own .SAFE root folder)
            target_extract_dir = get_windows_long_path(scene_dir)

            try:
                # Open zip with long path support
                with zipfile.ZipFile(get_windows_long_path(zip_path), "r") as zf:
                    zf.extractall(target_extract_dir)

                print(f"    → Successfully extracted .SAFE product")

                # Optional: Delete zip file after extraction to free up disk space
                # zip_path.unlink()
                # print(f"    → Deleted zip")

            except Exception as e:
                print(f"    ❌ Extraction failed: {e}")

        print()

    print("=== SAR organization completed ===")


if __name__ == "__main__":
    organize_sar()