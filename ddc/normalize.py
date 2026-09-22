"""Body-frame normalization: translate to hip center, scale by torso length."""

from __future__ import annotations

import numpy as np

from .skeleton import CONF_THR, J


def _smooth_nan(x: np.ndarray, win: int) -> np.ndarray:
    """Centered moving average along axis 0 ignoring NaNs; fills short gaps."""
    out = np.full_like(x, np.nan)
    h = win // 2
    for i in range(len(x)):
        seg = x[max(0, i - h): i + h + 1]
        if np.isfinite(seg).any():
            out[i] = np.nanmean(seg, axis=0)
    return out


def normalize(dense: np.ndarray, smooth_win: int = 15, rotate: bool = False):
    """dense: (F, T, 17, 3). Returns (norm (F,T,17,3), origin (F,T,2), scale (F,T), rot (F,T)).

    Normalized xy = R(-rot) @ (xy - origin) / scale.  Joints under CONF_THR are NaN.
    """
    F, T = dense.shape[:2]
    xy = dense[..., :2].copy()
    conf = dense[..., 2]
    xy[conf < CONF_THR] = np.nan

    lh, rh = xy[:, :, J["left_hip"]], xy[:, :, J["right_hip"]]
    ls, rs = xy[:, :, J["left_shoulder"]], xy[:, :, J["right_shoulder"]]
    hip, sho = (lh + rh) / 2, (ls + rs) / 2
    # Fall back to a single visible hip/shoulder if only one is detected.
    hip = np.where(np.isfinite(hip), hip, np.where(np.isfinite(lh), lh, rh))
    sho = np.where(np.isfinite(sho), sho, np.where(np.isfinite(ls), ls, rs))

    torso = sho - hip
    scale = np.linalg.norm(torso, axis=-1)
    origin = hip.copy()
    rot = np.zeros((F, T))
    for t in range(T):
        scale[:, t] = _smooth_nan(scale[:, t], smooth_win)
        origin[:, t] = _smooth_nan(origin[:, t], 3)
        if rotate:
            ang = np.arctan2(torso[:, t, 0], -torso[:, t, 1])  # 0 when torso points up
            s, c = _smooth_nan(np.sin(ang), smooth_win), _smooth_nan(np.cos(ang), smooth_win)
            rot[:, t] = np.arctan2(s, c)

    rel = (xy - origin[:, :, None, :]) / scale[:, :, None, None]
    if rotate:
        c, s = np.cos(-rot)[..., None], np.sin(-rot)[..., None]
        x, y = rel[..., 0], rel[..., 1]
        rel = np.stack([c * x - s * y, s * x + c * y], axis=-1)
    norm = np.concatenate([rel, conf[..., None]], axis=-1)
    norm[~np.isfinite(norm[..., 0])] = np.nan
    return norm, origin, scale, rot


def denormalize(pts: np.ndarray, origin, scale, rot) -> np.ndarray:
    """Inverse transform for one frame. pts (...,17,2); origin (2,), scale/rot scalars."""
    x, y = pts[..., 0], pts[..., 1]
    c, s = np.cos(rot), np.sin(rot)
    xr, yr = c * x - s * y, s * x + c * y
    return np.stack([xr, yr], axis=-1) * scale + origin
