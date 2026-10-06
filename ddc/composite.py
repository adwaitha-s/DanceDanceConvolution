"""Per-frame consensus pose: confidence-weighted geometric median across dancers."""

from __future__ import annotations

import numpy as np


def geometric_median(pts: np.ndarray, w: np.ndarray, iters: int = 50, eps: float = 1e-6):
    """Weighted geometric median of (N,2) points (Weiszfeld). N>=1."""
    if len(pts) == 1:
        return pts[0].copy()
    if len(pts) == 2:   # median of two points is any point on the segment; use the midpoint
        return pts.mean(0)
    x = (pts * w[:, None]).sum(0) / w.sum()
    for _ in range(iters):
        d = np.linalg.norm(pts - x, axis=1)
        inv = w / np.maximum(d, eps)
        nx = (pts * inv[:, None]).sum(0) / inv.sum()
        if np.linalg.norm(nx - x) < 1e-7:
            x = nx
            break
        x = nx
    return x


def _smooth_gaps(c: np.ndarray, strength: int, max_gap: int = 8) -> np.ndarray:
    """Interpolate short NaN gaps then apply a centered moving average. c: (F,17,2)."""
    out = c.copy()
    F = len(c)
    idx = np.arange(F)
    for j in range(c.shape[1]):
        for k in range(2):
            v = out[:, j, k]
            ok = np.isfinite(v)
            if ok.sum() < 2:
                continue
            filled = np.interp(idx, idx[ok], v[ok])
            # only keep interpolation for gaps <= max_gap
            gap = np.zeros(F, int)
            run = 0
            for i in range(F):
                run = run + 1 if not ok[i] else 0
                gap[i] = run
            for i in range(F - 2, -1, -1):
                if not ok[i] and gap[i + 1] > gap[i]:
                    gap[i] = gap[i + 1]
            v2 = np.where(ok | (gap <= max_gap), filled, np.nan)
            if strength > 1:
                h = strength // 2
                sm = np.full(F, np.nan)
                for i in range(F):
                    seg = v2[max(0, i - h): i + h + 1]
                    if np.isfinite(seg).any():
                        sm[i] = np.nanmean(seg)
                v2 = np.where(np.isfinite(v2), sm, np.nan)
            out[:, j, k] = v2
    return out


def composite_frame(norm_f: np.ndarray, exclude: int | None = None, min_conf: float = 0.3):
    """norm_f: (T,17,3) normalized poses at one frame -> (17,2) composite (NaN if none)."""
    T, J = norm_f.shape[:2]
    out = np.full((J, 2), np.nan)
    for j in range(J):
        pts, w = [], []
        for t in range(T):
            if t == exclude:
                continue
            p = norm_f[t, j]
            if np.isfinite(p[:2]).all() and p[2] >= min_conf:
                pts.append(p[:2])
                w.append(p[2])
        if pts:
            out[j] = geometric_median(np.array(pts), np.array(w))
    return out


def build_composite(norm: np.ndarray, smooth: int = 5, leave_one_out: bool = False):
    """norm: (F,T,17,3). Returns composite (F,17,2), or (F,T,17,2) leave-one-out."""
    F, T = norm.shape[:2]
    if not leave_one_out:
        c = np.stack([composite_frame(norm[f]) for f in range(F)])
        return _smooth_gaps(c, smooth)
    loo = np.full((F, T, norm.shape[2], 2), np.nan)
    for f in range(F):
        for t in range(T):
            if np.isfinite(norm[f, t, :, 0]).any():
                loo[f, t] = composite_frame(norm[f], exclude=t)
    for t in range(T):
        loo[:, t] = _smooth_gaps(loo[:, t], smooth)
    return loo
