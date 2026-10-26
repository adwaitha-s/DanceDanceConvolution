"""Per-frame consensus pose: confidence-weighted geometric median across dancers."""

from __future__ import annotations

import os

import numpy as np

from .smoothing import nan_moving_mean


def geometric_median(pts: np.ndarray, w: np.ndarray, iters: int = 8, eps: float = 1e-6):
    """Weighted geometric median of (N,2) points. N>=1. See `_batch_median`."""
    if len(pts) == 1:
        return pts[0].copy()
    if len(pts) == 2:   # median of two points is any point on the segment; use the midpoint
        return pts.mean(0)
    return _batch_median(pts[None], np.ones((1, len(pts)), bool), np.asarray(w, float)[None],
                         iters, eps)[0]


def _batch_median(pts: np.ndarray, valid: np.ndarray, w: np.ndarray, iters: int = 8,
                  eps: float = 1e-6, x0: np.ndarray | None = None) -> np.ndarray:
    """`geometric_median` for many point sets at once.

    pts (B,N,2), valid (B,N) bool, w (B,N) weights; each set uses only its valid points.
    1 point -> that point, 2 -> their midpoint, otherwise a few Weiszfeld steps (`iters`)
    to get close, then damped Newton steps to converge, then `_snap_to_point` for medians
    that sit exactly on a data point. Plain Weiszfeld crawls near data points and could
    stop up to ~0.3 torso lengths short of the median, which showed up as deviation.

    `x0` (B,2) optionally starts nearer the answer (NaN rows start from the weighted
    mean). A start on one of the points is ignored: Weiszfeld can't move off a data
    point, so it would stop there even when that point isn't the median.
    Returns (B,2), NaN for sets with no valid point."""
    B = len(pts)
    out = np.full((B, 2), np.nan)
    n = valid.sum(1)
    p = np.where(valid[..., None], pts, 0.0)
    w = np.where(valid, w, 0.0)
    one, two, many = n == 1, n == 2, n >= 3
    out[one] = p[one].sum(1)                  # the single valid point
    out[two] = p[two].sum(1) / 2              # midpoint (unweighted, like geometric_median)
    if not many.any():
        return out
    idx = np.flatnonzero(many)
    P, W, V = p[idx], w[idx], valid[idx]
    x = (P * W[..., None]).sum(1) / W.sum(1)[:, None]
    if x0 is not None:
        start = x0[idx]
        near = np.where(V, np.linalg.norm(P - start[:, None], axis=-1), np.inf).min(1)
        use = np.isfinite(start).all(1) & (near > 1e-3)
        x = np.where(use[:, None], start, x)
        iters = min(iters, 2)                 # already close: Newton takes it from here
    for _ in range(iters):
        x = _weiszfeld_step(x, P, W, V, eps)
    out[idx] = _newton(x, P, W, V, eps)
    return out


def _weiszfeld_step(x, P, W, V, eps):
    d = np.linalg.norm(P - x[:, None], axis=-1)
    inv = np.where(V, W / np.maximum(d, eps), 0.0)
    return (P * inv[..., None]).sum(1) / inv.sum(1)[:, None]


def _objective(x, P, W, V):
    return np.where(V, W * np.linalg.norm(P - x[:, None], axis=-1), 0.0).sum(1)


def _newton(x, P, W, V, eps, steps: int = 30, tol: float = 1e-8):
    """Damped Newton on sum_i w_i |x - p_i|. Each step halves until the objective drops;
    if no step size helps, a Weiszfeld step is taken instead. Sets whose median is one
    of their points (`_snap_to_point`) finish as soon as that's detected."""
    x = x.copy()
    active = np.arange(len(x))
    for it in range(steps):
        if it >= 2:      # by now only sets crawling toward a point are left; settle those
            snapped, on_point = _snap_to_point(x[active], P[active], W[active], V[active], True)
            x[active[on_point]] = snapped[on_point]
            active = active[~on_point]
            if not len(active):
                break
        Pa, Wa, Va, xa = P[active], W[active], V[active], x[active]
        diff = xa[:, None] - Pa
        d = np.linalg.norm(diff, axis=-1)
        m = Va & (d > eps)
        with np.errstate(invalid="ignore", divide="ignore"):
            u = np.where(m[..., None], diff / d[..., None], 0.0)
            c = np.where(m, Wa / d, 0.0)
        g = (Wa[..., None] * u).sum(1)
        hxx = (c * (1 - u[..., 0] ** 2)).sum(1)
        hyy = (c * (1 - u[..., 1] ** 2)).sum(1)
        hxy = -(c * u[..., 0] * u[..., 1]).sum(1)
        reg = 1e-9 * (hxx + hyy) + 1e-300
        hxx, hyy = hxx + reg, hyy + reg
        det = hxx * hyy - hxy ** 2
        step = -np.stack([hyy * g[:, 0] - hxy * g[:, 1], hxx * g[:, 1] - hxy * g[:, 0]], 1) / det[:, None]
        tiny = np.linalg.norm(step, axis=-1) <= tol   # converged: take the step and stop
        x[active[tiny]] = xa[tiny] + step[tiny]
        active, step = active[~tiny], step[~tiny]
        if not len(active):
            break
        Pa, Wa, Va, xa = Pa[~tiny], Wa[~tiny], Va[~tiny], xa[~tiny]
        f0 = _objective(xa, Pa, Wa, Va)
        new = xa.copy()
        pending = np.ones(len(active), bool)
        t = 1.0
        for _ in range(20):
            cand = xa[pending] + t * step[pending]
            better = _objective(cand, Pa[pending], Wa[pending], Va[pending]) < f0[pending]
            rows = np.flatnonzero(pending)[better]
            new[rows] = cand[better]
            pending[rows] = False
            if not pending.any():
                break
            t *= 0.5
        if pending.any():                      # Newton direction didn't help: fall back
            new[pending] = _weiszfeld_step(xa[pending], Pa[pending], Wa[pending], Va[pending], eps)
        moved = np.linalg.norm(new - xa, axis=-1)
        x[active] = new
        active = active[moved > tol]
        if not len(active):
            break
    return _snap_to_point(x, P, W, V)


def _snap_to_point(x: np.ndarray, P: np.ndarray, W: np.ndarray, V: np.ndarray,
                   return_mask: bool = False):
    """Weiszfeld only creeps toward a median that is one of the data points, so it can stop
    well short of it. Point k is the exact median when the weighted pull of the others,
    |sum_i w_i (p_i - p_k)/|p_i - p_k||, is no more than w_k; check the point nearest each
    estimate and snap to it when that holds."""
    d = np.where(V, np.linalg.norm(P - x[:, None], axis=-1), np.inf)
    k = d.argmin(1)
    rows = np.arange(len(P))
    pk, wk = P[rows, k], W[rows, k]
    diff = P - pk[:, None]
    dist = np.linalg.norm(diff, axis=-1)
    others = V & (dist > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pull = np.where(others[..., None], W[..., None] * diff / dist[..., None], 0.0).sum(1)
    snap = np.linalg.norm(pull, axis=-1) <= wk
    out = np.where(snap[:, None], pk, x)
    return (out, snap) if return_mask else out


def _gap_lengths(ok: np.ndarray) -> np.ndarray:
    """For a 1-D finite-mask, the length of the NaN run each element sits in (0 if finite)."""
    nan = ~ok
    run_id = np.cumsum(ok)                    # constant within a run of NaNs
    counts = np.bincount(run_id[nan], minlength=run_id[-1] + 1 if len(run_id) else 1)
    return np.where(nan, counts[run_id], 0)


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
            v2 = np.where(ok | (_gap_lengths(ok) <= max_gap), filled, np.nan)
            if strength > 1:
                v2 = np.where(np.isfinite(v2), nan_moving_mean(v2, strength), np.nan)
            out[:, j, k] = v2
    return out


def _composite(norm: np.ndarray, exclude: int | None = None, min_conf: float = 0.3,
               start: np.ndarray | None = None) -> np.ndarray:
    """norm (F,T,17,3) -> (F,17,2) per-frame composite, optionally leaving dancer `exclude` out.
    `start` (F,17,2) is an initial guess for the median (see `_batch_median`)."""
    F, T, J = norm.shape[:3]
    pts = norm[..., :2].transpose(0, 2, 1, 3).reshape(F * J, T, 2)
    conf = norm[..., 2].transpose(0, 2, 1).reshape(F * J, T)
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(pts).all(-1) & (conf >= min_conf)
    if exclude is not None:
        valid[:, exclude] = False
    x0 = None if start is None else start.reshape(F * J, 2)
    return _parallel_median(pts, valid, conf, x0).reshape(F, J, 2)


_CHUNK = 8192   # point sets per task; numpy releases the GIL, so chunks run on separate cores


def _parallel_median(pts, valid, w, x0=None):
    """`_batch_median` over chunks of the batch on a thread pool (same result, faster)."""
    B = len(pts)
    if B <= _CHUNK:
        return _batch_median(pts, valid, w, x0=x0)
    from concurrent.futures import ThreadPoolExecutor
    bounds = [(a, min(a + _CHUNK, B)) for a in range(0, B, _CHUNK)]
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as ex:
        parts = ex.map(lambda ab: _batch_median(pts[ab[0]:ab[1]], valid[ab[0]:ab[1]], w[ab[0]:ab[1]],
                                                x0=None if x0 is None else x0[ab[0]:ab[1]]), bounds)
        return np.concatenate(list(parts))


def composite_frame(norm_f: np.ndarray, exclude: int | None = None, min_conf: float = 0.3):
    """norm_f: (T,17,3) normalized poses at one frame -> (17,2) composite (NaN if none)."""
    return _composite(norm_f[None], exclude, min_conf)[0]


def build_composite(norm: np.ndarray, smooth: int = 5, leave_one_out: bool = False):
    """norm: (F,T,17,3). Returns composite (F,17,2), or (F,T,17,2) leave-one-out."""
    F, T = norm.shape[:2]
    if not leave_one_out:
        return _smooth_gaps(_composite(norm), smooth)
    loo = np.full((F, T, norm.shape[2], 2), np.nan)
    present = np.isfinite(norm[..., 0]).any(-1)          # (F,T)
    full = _composite(norm)   # dropping one dancer barely moves the median: start each from it
    for t in range(T):
        f = np.flatnonzero(present[:, t])
        if len(f):
            loo[f, t] = _composite(norm[f], exclude=t, start=full[f])
        loo[:, t] = _smooth_gaps(loo[:, t], smooth)
    return loo
