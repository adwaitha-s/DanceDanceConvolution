"""NaN-aware moving averages shared by normalization, the composite, tracking and plots."""

from __future__ import annotations

import numpy as np


def nan_moving_mean(x: np.ndarray, win: int) -> np.ndarray:
    """Centered moving mean along axis 0 that ignores NaNs.

    out[i] = nanmean(x[i - win//2 : i + win//2 + 1]) (window clipped at the ends),
    NaN where the window holds no finite value. Same result as the per-frame loop it
    replaces, computed with cumulative sums so it's O(F) instead of a Python loop.
    """
    x = np.asarray(x, float)
    h = win // 2
    if h <= 0 or len(x) == 0:
        return x.copy()
    ok = np.isfinite(x)
    pad = [(h + 1, h)] + [(0, 0)] * (x.ndim - 1)
    s = np.cumsum(np.pad(np.where(ok, x, 0.0), pad), axis=0)
    n = np.cumsum(np.pad(ok.astype(np.int64), pad), axis=0)
    L = 2 * h + 1
    tot, cnt = s[L:] - s[:-L], n[L:] - n[:-L]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, tot / cnt, np.nan)
