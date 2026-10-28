import sys
from pathlib import Path
import pandas as pd

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.paths import QUALITY_CSV
from configs.settings import MIN_VALID_PCT


def get_qualifying_scenes(min_valid=MIN_VALID_PCT, min_sar=1):
    """Reads quality report CSV and returns scenes that have at least min_sar SAR products each meeting min_valid."""
    if not QUALITY_CSV.exists():
        raise FileNotFoundError(f"Quality report not found: {QUALITY_CSV}")

    df = pd.read_csv(QUALITY_CSV)
    if df.empty:
        return []

    sar_cols = sorted([c for c in df.columns if c.startswith("sar") and c.endswith("_valid_pct")])
    qualifying = []
    for _, row in df.iterrows():
        valid_sar_count = sum(1 for c in sar_cols if pd.notna(row.get(c)) and row[c] >= min_valid)
        if valid_sar_count >= min_sar:
            qualifying.append(str(row["scene_id"]))

    return sorted(qualifying)

