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


def _center(p: np.ndarray) -> np.ndarray:
    """Hip midpoint (or the one visible hip / centroid of visible joints); NaN if none."""
    ok = p[:, 2] >= CONF_THR
    hips = [J["left_hip"], J["right_hip"]]
    if ok[hips].any():
        return p[hips][ok[hips], :2].mean(0)
    return p[ok, :2].mean(0) if ok.any() else np.full(2, np.nan)


def _smoothed_centers(cs: np.ndarray, win: int = 5) -> np.ndarray:
    """Moving nanmean of (n,2) centers so one flailing frame doesn't decide a link."""
    out = np.full_like(cs, np.nan)
    h = win // 2
    for i in range(len(cs)):
        seg = cs[max(0, i - h): i + h + 1]
        if np.isfinite(seg).any():
            out[i] = np.nanmean(seg, axis=0)
    return out


def _make_group(occ: list, kps: list, members: list) -> dict:
    """Arrays describing a set of detections of (what may be) one person, by frame."""
    occ = sorted(occ)
    cs = np.array([_center(kps[f][i]) for f, i in occ])
    return {"occ": occ, "members": members,
            "frames": np.array([f for f, _ in occ]),
            "centers": _smoothed_centers(cs),
            "sizes": np.array([_size(kps[f][i]) for f, i in occ])}


def _link_cost(a: dict, b: dict, fps: float, max_gap: int, max_overlap: int = 3,
               dup_dist: float = 0.6, k: int = 8) -> float:
    """Cost of `a` and `b` being the same person (inf = impossible).

    Two detections can't be one person if they're in the same frame, so groups
    sharing frames never link -- except a few frames where both detections sit
    on top of each other (a duplicate the frame-level dedupe missed). Otherwise
    the cost looks at the `k` closest-in-time pairs of detections (this also
    covers groups whose time spans interleave): how much farther apart they are
    than someone could plausibly travel in that gap, plus a torso-size mismatch
    and a small preference for short gaps.
    """
    fa, fb = a["frames"], b["frames"]
    shared = np.intersect1d(fa, fb)
    if len(shared):
        if len(shared) > max_overlap:
            return np.inf
        for f in shared:
            ia, ib = np.searchsorted(fa, f), np.searchsorted(fb, f)
            d = np.linalg.norm(a["centers"][ia] - b["centers"][ib])
            if not d / min(a["sizes"][ia], b["sizes"][ib]) < dup_dist:
                return np.inf
    # nearest `a` frame for each `b` frame, skipping shared frames
    idx = np.searchsorted(fa, fb)
    left, right = np.maximum(idx - 1, 0), np.minimum(idx, len(fa) - 1)
    pick = np.where(np.abs(fb - fa[left]) <= np.abs(fa[right] - fb), left, right)
    gap = np.abs(fb - fa[pick])
    ok = gap >= 1
    if not ok.any():
        return np.inf
    sel = np.flatnonzero(ok)
    sel = sel[np.argsort(gap[sel], kind="stable")[:k]]
    gaps = gap[sel]
    if gaps.min() > max_gap:
        return np.inf
    size = np.minimum(a["sizes"][pick[sel]], b["sizes"][sel])
    dist = np.linalg.norm(a["centers"][pick[sel]] - b["centers"][sel], axis=1) / size
    if not np.isfinite(dist).any():
        return np.inf
    excess = float(np.nanmedian(np.maximum(0.0, dist - (0.5 + gaps / fps))))
    size_pen = abs(float(np.log(np.median(b["sizes"]) / np.median(a["sizes"]))))
    return excess + size_pen + 0.1 * float(np.median(gaps)) / fps


def stitch_tracklets(tracklets: list, kps: list, max_stitch_gap: int = 150,
                     stitch_cost: float = 1.0, fps: float = 30.0) -> list:
    """Greedily join tracklets that look like one person who dropped out and came back.

    `tracklets`: list of [(frame, detection_idx), ...]. Merges the cheapest
    pair first (see `_link_cost`), re-scoring the merged group against the rest,
    until no pair is under `stitch_cost`. Returns groups: lists of indices into
    `tracklets`, ordered by first appearance.
    """
    import heapq

    groups = {i: _make_group(occ, kps, [i]) for i, occ in enumerate(tracklets)}
    version = {i: 0 for i in groups}
    heap: list = []

    def push(x: int, y: int):
        c = _link_cost(groups[x], groups[y], fps, max_stitch_gap)
        if c <= stitch_cost:
            heapq.heappush(heap, (c, x, y, version[x], version[y]))

    ids = sorted(groups)
    for n, x in enumerate(ids):
        for y in ids[n + 1:]:
            push(x, y)

    next_id = len(tracklets)
    while heap:
        _, x, y, vx, vy = heapq.heappop(heap)
        if x not in groups or y not in groups or version[x] != vx or version[y] != vy:
            continue
        a, b = groups.pop(x), groups.pop(y)
        groups[next_id] = _make_group(a["occ"] + b["occ"], kps, a["members"] + b["members"])
        version[next_id] = 0
        for other in list(groups):
            if other != next_id:
                push(next_id, other)
        next_id += 1
    return [g["members"] for g in sorted(groups.values(), key=lambda g: (g["frames"][0], g["members"][0]))]


def stabilize(kps: list, max_cost: float = 2.0, max_gap: int = 10,
              min_track_len: int = 15, dup_cost: float = 0.4, stitch: bool = True,
              max_stitch_gap: int = 150, stitch_cost: float = 1.0, fps: float = 30.0,
              return_info: bool = False):
    """Assign stable track ids.

    Returns (track_ids, n_tracks) where track_ids[f] is an int array aligned
    with kps[f] (-1 = dropped as a short-lived spurious track).

    Frame-to-frame matching forgets a person after `max_gap` frames, so anyone
    occluded or undetected for longer comes back as a new id. With `stitch`
    those fragments are re-linked by `stitch_tracklets` *before* short tracks
    are discarded, so a fragment can be absorbed into a real person rather than
    lost. `return_info=True` adds a third value: {"raw": pre-stitch ids,
    "fragments": number of raw fragments in each final track}.
    """
    raw, n_raw = _track_frames(kps, max_cost, max_gap, dup_cost)

    occs: list[list[tuple[int, int]]] = [[] for _ in range(n_raw)]
    for f, a in enumerate(raw):
        for i, tid in enumerate(a):
            if tid >= 0:
                occs[tid].append((f, i))
    groups = (stitch_tracklets(occs, kps, max_stitch_gap, stitch_cost, fps)
              if stitch else [[t] for t in range(n_raw)])

    # Drop tracks that live only briefly (likely false detections).
    kept = [g for g in groups if sum(len(occs[t]) for t in g) >= min_track_len]
    kept.sort(key=lambda g: (occs[g[0]][0][0], g[0]))
    remap = {t: n for n, g in enumerate(kept) for t in g}
    out = [np.array([remap.get(t, -1) for t in a], int) for a in raw]
    for f, a in enumerate(out):    # stitched duplicates sharing a frame: keep the better one
        for tid in {t for t in a if t >= 0 and (a == t).sum() > 1}:
            dup = np.flatnonzero(a == tid)
            best = max(dup, key=lambda i: np.nanmean(kps[f][i][:, 2]))
            a[[i for i in dup if i != best]] = -1
    if return_info:
        return out, len(kept), {"raw": raw, "fragments": [len(g) for g in kept]}
    return out, len(kept)


def _track_frames(kps: list, max_cost: float, max_gap: int, dup_cost: float):
    """Frame-to-frame Hungarian tracking. Returns (raw ids per frame, n raw ids)."""
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
    return out, next_id


def merge_track_ids(kps: list, track_ids: list, groups) -> tuple[list, int]:
    """Manually merge tracks: each group (a list of track ids) becomes one track.

    The merged track takes the id/position of its lowest member and the rest
    are renumbered in order, so ids stay contiguous. If two merged tracks have
    a detection in the same frame, the higher-confidence one is kept.
    Returns (track_ids, n_tracks) like `stabilize`.
    """
    n = 1 + max((int(t) for a in track_ids for t in a), default=-1)
    groups = [sorted({int(t) for t in g}) for g in (groups or []) if len(g) > 1]
    flat = [t for g in groups for t in g]
    if len(flat) != len(set(flat)):
        raise ValueError("a person can only be in one merge group")
    if any(t < 0 or t >= n for t in flat):
        raise ValueError(f"merge ids {sorted(flat)} don't match this run's {n} detected people")
    target = {t: t for t in range(n)}
    for g in groups:
        for t in g:
            target[t] = g[0]
    kept = sorted(set(target.values()))
    renum = {t: i for i, t in enumerate(kept)}
    out = []
    for f, a in enumerate(track_ids):
        b = np.array([renum[target[int(t)]] if t >= 0 else -1 for t in a], int)
        for tid in {t for t in b if t >= 0 and (b == t).sum() > 1}:
            dup = np.flatnonzero(b == tid)
            best = max(dup, key=lambda i: np.nanmean(kps[f][i][:, 2]))
            b[[i for i in dup if i != best]] = -1
        out.append(b)
    return out, len(kept)


def to_dense(kps: list, track_ids: list, n_tracks: int) -> np.ndarray:
    """(F, T, 17, 3) array with NaN where a track is absent."""
    dense = np.full((len(kps), n_tracks, kps[0].shape[1] if len(kps) else 17, 3), np.nan)
    for f, (frame, ids) in enumerate(zip(kps, track_ids)):
        for p, tid in zip(frame, ids):
            if tid >= 0:
                dense[f, tid] = p
    return dense


def stable_track_ids(det, merge_groups=None) -> tuple[list, int]:
    """Final per-frame track ids for a loaded run: automatic stitching, then any manual merges.

    This is the one place that defines "who is person N" for a run.  The Find people
    list, the composite analysis and the dancer-filtering features must all use it, or
    labels and predictions attach to different people than the ones on screen.
    """
    t = det.t
    fps = float((len(t) - 1) / (t[-1] - t[0])) if len(t) > 1 and t[-1] > t[0] else 30.0
    ids, n = stabilize(det.kps, fps=fps)
    groups = [sorted(set(g)) for g in (merge_groups or []) if len(set(g)) > 1]
    if groups:
        ids, n = merge_track_ids(det.kps, ids, groups)
    return ids, n
