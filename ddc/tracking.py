"""Stabilize per-frame detections into consistent dancer tracks.

RTMW ids are per-frame indices and swap between frames, so we re-associate
detections across frames with Hungarian matching.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .skeleton import CONF_THR, J


def _size(p: np.ndarray) -> float:
    """Person scale in px: torso length if available, else bbox diagonal."""
    ok = p[:, 2] >= CONF_THR
    if ok[J["left_shoulder"]] and ok[J["right_shoulder"]] and ok[J["left_hip"]] and ok[J["right_hip"]]:
        sh = p[[J["left_shoulder"], J["right_shoulder"]], :2].mean(0)
        hp = p[[J["left_hip"], J["right_hip"]], :2].mean(0)
        return max(float(np.linalg.norm(sh - hp)), 1.0)
    pts = p[ok, :2]
    if len(pts) < 2:
        return 1.0
    return max(float(np.linalg.norm(pts.max(0) - pts.min(0))) * 0.4, 1.0)


def _cost(a: np.ndarray, b: np.ndarray) -> float:
    """Mean joint distance (in units of a's size) over jointly-visible joints."""
    both = (a[:, 2] >= CONF_THR) & (b[:, 2] >= CONF_THR)
    if both.sum() < 3:
        return np.inf
    d = np.linalg.norm(a[both, :2] - b[both, :2], axis=1).mean()
    return d / _size(a)


def _dedupe(frame: np.ndarray, dup_cost: float = 0.4) -> np.ndarray:
    """Boolean keep-mask: drop near-identical detections of the same person.

    The detector sometimes emits two overlapping skeletons for one dancer; keep
    the one with more visible joints / higher mean confidence.
    """
    n = len(frame)
    keep = np.ones(n, bool)
    quality = [((p[:, 2] >= CONF_THR).sum(), np.nanmean(p[:, 2])) for p in frame]
    for i in range(n):
        for j in range(i + 1, n):
            if keep[i] and keep[j] and min(_cost(frame[i], frame[j]),
                                           _cost(frame[j], frame[i])) < dup_cost:
                keep[j if quality[i] >= quality[j] else i] = False
    return keep


def stabilize(kps: list, max_cost: float = 2.0, max_gap: int = 10,
              min_track_len: int = 15, dup_cost: float = 0.4):
    """Assign stable track ids.

    Returns (track_ids, n_tracks) where track_ids[f] is an int array aligned
    with kps[f] (-1 = dropped as a short-lived spurious track).
    """
    tracks: dict[int, dict] = {}   # id -> {"pose": last pose, "last": frame}
    next_id = 0
    out = []
    for f, frame in enumerate(kps):
        active = [tid for tid, s in tracks.items() if f - s["last"] <= max_gap]
        keep = _dedupe(frame, dup_cost) if len(frame) else np.zeros(0, bool)
        assigned = np.full(len(frame), -1, int)
        if keep.any() and active:
            C = np.full((len(frame), len(active)), np.inf)
            for i, p in enumerate(frame):
                if not keep[i]:
                    continue
                for j, tid in enumerate(active):
                    C[i, j] = _cost(tracks[tid]["pose"], p)
            Cf = np.where(np.isfinite(C) & (C <= max_cost), C, 1e6)
            r, c = linear_sum_assignment(Cf)
            for i, j in zip(r, c):
                if Cf[i, j] < 1e6:
                    assigned[i] = active[j]
        for i in range(len(frame)):
            if not keep[i]:
                continue
            if assigned[i] < 0:
                assigned[i] = next_id
                next_id += 1
            tracks[assigned[i]] = {"pose": frame[i], "last": f}
        out.append(assigned)

    # Drop tracks that live only briefly (likely false detections).
    counts: dict[int, int] = {}
    for a in out:
        for tid in a:
            counts[tid] = counts.get(tid, 0) + 1
    keep = sorted(t for t, n in counts.items() if n >= min_track_len)
    remap = {t: i for i, t in enumerate(keep)}
    out = [np.array([remap.get(t, -1) for t in a], int) for a in out]
    return out, len(keep)


def to_dense(kps: list, track_ids: list, n_tracks: int) -> np.ndarray:
    """(F, T, 17, 3) array with NaN where a track is absent."""
    dense = np.full((len(kps), n_tracks, kps[0].shape[1] if len(kps) else 17, 3), np.nan)
    for f, (frame, ids) in enumerate(zip(kps, track_ids)):
        for p, tid in zip(frame, ids):
            if tid >= 0:
                dense[f, tid] = p
    return dense
