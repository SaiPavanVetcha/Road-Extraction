import numpy as np


def robust_stretch(band, lower=2, upper=98):
    """Normalizes array between 0.0 and 1.0 float32 using percentiles."""
    band = band.astype(np.float32)
    valid = band[np.isfinite(band) & (band > 0)]
    if valid.size < 50:
        return np.zeros_like(band, dtype=np.float32)
    p_low, p_high = np.percentile(valid, (lower, upper))
    return np.clip((band - p_low) / (p_high - p_low + 1e-6), 0, 1).astype(
        np.float32
    )


def process_sar_channel(sar_arr):
    """Normalizes SAR decibel channel between 0.0 and 1.0 for display using percentiles."""
    sar_arr = sar_arr.astype(np.float32)
    valid_mask = (sar_arr > -9990) & np.isfinite(sar_arr)

    if not np.any(valid_mask):
        return np.zeros_like(sar_arr, dtype=np.float32)

    valid_vals = sar_arr[valid_mask]
    p2, p98 = np.percentile(valid_vals, (2, 98))
    sar_norm = np.clip((sar_arr - p2) / (p98 - p2 + 1e-6), 0, 1)
    sar_norm[~valid_mask] = 0
    return sar_norm.astype(np.float32)

