"""Stabilize per-frame detections into consistent dancer tracks.

RTMW ids are per-frame indices and swap between frames. Associate those
detections using torso motion, size and pose; preserve explicit tracker ids.
"""

from __future__ import annotations

from collections import OrderedDict
import json

import numpy as np
from scipy.optimize import linear_sum_assignment

from .skeleton import CONF_THR, J
from .appearance import appearance_distance, combine_appearances, load_appearances, source_stamp


TRACKING_VERSION = 2
_TRACK_CACHE = OrderedDict()


def baseline_stamp(path):
    """Inputs to the motion/appearance tracker, before learned and manual joins."""
    from pathlib import Path
    stat = Path(path).stat()
    return [TRACKING_VERSION, stat.st_size, stat.st_mtime_ns, source_stamp(path)]


def tracking_stamp(path):
    """Include the frozen identity model decisions in all downstream cache keys."""
    from .identity_learning import prediction_stamp
    stamp = baseline_stamp(path)
    identity = prediction_stamp(path)
    return stamp + [identity] if identity is not None else stamp


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


def _dedupe(frame: np.ndarray, dup_cost: float = 0.15, appearances=None) -> np.ndarray:
    """Boolean keep-mask: drop near-identical detections of the same person.

    The detector sometimes emits two overlapping skeletons for one dancer; keep
    the one with more visible joints / higher mean confidence.
    """
    n = len(frame)
    keep = np.ones(n, bool)
    quality = [((p[:, 2] >= CONF_THR).sum(), np.nanmean(p[:, 2])) for p in frame]
    for i in range(n):
        for j in range(i + 1, n):
            color = appearance_distance(appearances[i], appearances[j]) if appearances is not None else None
            if color is not None and color > .25:
                continue
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


def _box_size(box):
    if box is None:
        return np.nan
    box = np.asarray(box, float)
    return float(np.linalg.norm(box[2:] - box[:2])) if np.isfinite(box).all() else np.nan


def _make_group(occ: list, kps: list, members: list, appearances=None, boxes=None) -> dict:
    """Arrays describing a set of detections of (what may be) one person, by frame."""
    occ = sorted(occ)
    cs = np.array([_center(kps[f][i]) for f, i in occ])
    return {"occ": occ, "members": members,
            "frames": np.array([f for f, _ in occ]),
            "centers": _smoothed_centers(cs),
            "sizes": np.array([_size(kps[f][i]) for f, i in occ]),
            "appearance": combine_appearances([appearances[f][i] for f, i in occ])
                          if appearances is not None else None,
            "extents": np.array([_box_size(boxes[f][i]) for f, i in occ]) if boxes is not None else None}


def _velocity(frames, centers) -> np.ndarray:
    """Robust translation per frame, without extrapolating limb movement."""
    if len(frames) < 2:
        return np.zeros(2)
    steps = np.diff(centers, axis=0) / np.diff(frames)[:, None]
    valid = np.isfinite(steps).all(axis=1)
    return np.median(steps[valid], axis=0) if valid.any() else np.zeros(2)


def _prediction(state: dict, f: int) -> dict:
    frames, centers, sizes = zip(*state["history"])
    gap = f - state["last"]
    return {"pose": state["pose"], "center": centers[-1], "gap": gap,
            "appearance": state["appearance"],
            "extent": float(np.median(state["extents"])) if state["extents"] else np.nan,
            "size": float(np.median(sizes)),
            "predicted": centers[-1] + _velocity(np.asarray(frames), np.asarray(centers)) * gap}


def _association_cost(state: dict, p: np.ndarray, center: np.ndarray,
                      size: float, max_cost: float, appearance=None, extent=np.nan) -> float:
    """Match the predicted torso, allowing pose changes without changing identity."""
    gap = state["gap"]
    distance = float(np.linalg.norm(center - state["predicted"])) / state["size"]
    # An old track must not grab a neighbour simply because its limbs look similar.
    if not distance <= min(max_cost, 0.8 + 0.1 * (gap - 1)):
        return np.inf
    size_change = abs(float(np.log(size / state["size"])))
    if size_change > np.log(2.5):
        return np.inf
    color = appearance_distance(state["appearance"], appearance)
    if color is not None and color > .8:
        return np.inf
    extent_change = abs(float(np.log(extent / state["extent"]))) if extent > 0 and state["extent"] > 0 else 0.0
    if extent_change > np.log(2.5):
        return np.inf
    previous = state["pose"]
    both = (previous[:, 2] >= CONF_THR) & (p[:, 2] >= CONF_THR)
    if both.sum() < 3:
        return np.inf
    delta = (previous[both, :2] - state["center"]) - (p[both, :2] - center)
    pose_cost = np.linalg.norm(delta, axis=1).mean() / state["size"]
    return (distance + 0.15 * pose_cost + 0.5 * size_change + extent_change
            + 0.03 * (gap - 1) + 1.5 * (color or 0))


def _link_cost(a: dict, b: dict, fps: float, max_gap: int, k: int = 8) -> float:
    """Cost of `a` and `b` being the same person (inf = impossible).

    Two detections can't be one person if they're in the same frame, so groups
    with overlapping time spans never link. Otherwise the cost looks at the
    `k` closest-in-time pairs of detections: how much farther apart they are
    than someone could plausibly travel in that gap, plus a torso-size mismatch
    and a small preference for short gaps.
    """
    fa, fb = a["frames"], b["frames"]
    # Frame-level deduplication already removed duplicate skeletons. Joining
    # simultaneous or interleaved tracks can combine two different people.
    if fa[-1] >= fb[0] and fb[-1] >= fa[0]:
        return np.inf
    color = appearance_distance(a["appearance"], b["appearance"])
    if color is not None and color > .6:
        return np.inf
    # Nearest `a` frame for each `b` frame.
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
    if a["extents"] is not None and b["extents"] is not None:
        extent = np.minimum(a["extents"][pick[sel]], b["extents"][sel])
        size = np.where(np.isfinite(extent), np.maximum(size, extent * .25), size)
    dist = np.linalg.norm(a["centers"][pick[sel]] - b["centers"][sel], axis=1) / size
    if not np.isfinite(dist).any():
        return np.inf
    excess = float(np.nanmedian(np.maximum(0.0, dist - (0.5 + gaps / fps))))
    size_pen = abs(float(np.log(np.median(b["sizes"]) / np.median(a["sizes"]))))
    if a["extents"] is not None and b["extents"] is not None:
        ea, eb = a["extents"], b["extents"]
        ea, eb = ea[np.isfinite(ea) & (ea > 0)], eb[np.isfinite(eb) & (eb > 0)]
        if len(ea) and len(eb):
            # A floor move can foreshorten the torso dramatically while the
            # full-body diagonal still distinguishes foreground from audience.
            size_pen = .2 * size_pen + abs(float(np.log(np.median(eb) / np.median(ea))))
    return excess + size_pen + 0.1 * float(np.median(gaps)) / fps + (color or 0)


def stitch_tracklets(tracklets: list, kps: list, max_stitch_gap: int = 150,
                     stitch_cost: float = 1.0, fps: float = 30.0, appearances=None, boxes=None) -> list:
    """Greedily join tracklets that look like one person who dropped out and came back.

    `tracklets`: list of [(frame, detection_idx), ...]. Merges the cheapest
    pair first (see `_link_cost`), re-scoring the merged group against the rest,
    until no pair is under `stitch_cost`. Returns groups: lists of indices into
    `tracklets`, ordered by first appearance.
    """
    import heapq

    groups = {i: _make_group(occ, kps, [i], appearances, boxes) for i, occ in enumerate(tracklets)}
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
        groups[next_id] = _make_group(a["occ"] + b["occ"], kps, a["members"] + b["members"], appearances, boxes)
        version[next_id] = 0
        for other in list(groups):
            if other != next_id:
                push(next_id, other)
        next_id += 1
    return [g["members"] for g in sorted(groups.values(), key=lambda g: (g["frames"][0], g["members"][0]))]


def stabilize(kps: list, max_cost: float = 2.0, max_gap: int = 10,
              min_track_len: int = 15, dup_cost: float = 0.15, stitch: bool = True,
              max_stitch_gap: int = 150, stitch_cost: float = 1.0, fps: float = 30.0,
              return_info: bool = False, source_ids: list | None = None, appearances=None, boxes=None):
    """Assign stable track ids.

    Returns (track_ids, n_tracks) where track_ids[f] is an int array aligned
    with kps[f] (-1 = dropped as a short-lived spurious track).

    Frame-to-frame matching forgets a person after `max_gap` frames, so anyone
    occluded or undetected for longer comes back as a new id. With `stitch`
    those fragments are re-linked by `stitch_tracklets` *before* short tracks
    are discarded, so a fragment can be absorbed into a real person rather than
    lost. `return_info=True` adds a third value: {"raw": pre-stitch ids,
    "fragments": number of raw fragments in each final track}.

    Explicit `source_ids` preserve an upstream tracker's identities. Optional
    clothing `appearances` disambiguate overlapping people without rerunning
    pose detection; pose-only files use the motion/geometry fallback.
    """
    raw, n_raw = (_source_tracks(source_ids) if source_ids is not None else
                  _track_frames(kps, max_cost, max_gap, dup_cost, appearances, boxes))

    occs: list[list[tuple[int, int]]] = [[] for _ in range(n_raw)]
    for f, a in enumerate(raw):
        for i, tid in enumerate(a):
            if tid >= 0:
                occs[tid].append((f, i))
    groups = (stitch_tracklets(occs, kps, max_stitch_gap, stitch_cost, fps, appearances, boxes)
              if stitch and source_ids is None else [[t] for t in range(n_raw)])

    # Drop tracks that live only briefly (likely false detections).
    kept = [g for g in groups if sum(len(occs[t]) for t in g) >= min_track_len]
    kept.sort(key=lambda g: min((occs[t][0][0], t) for t in g))
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


def _track_frames(kps: list, max_cost: float, max_gap: int, dup_cost: float, appearances=None, boxes=None):
    """Frame-to-frame Hungarian tracking. Returns (raw ids per frame, n raw ids)."""
    tracks: dict[int, dict] = {}
    next_id = 0
    out = []
    for f, frame in enumerate(kps):
        active = [tid for tid, s in tracks.items() if f - s["last"] <= max_gap]
        predictions = {tid: _prediction(tracks[tid], f) for tid in active}
        centers = [_center(p) for p in frame]
        sizes = [_size(p) for p in frame]
        extents = [_box_size(boxes[f][i]) if boxes is not None else np.nan for i in range(len(frame))]
        keep = (_dedupe(frame, dup_cost, appearances[f] if appearances is not None else None)
                if len(frame) else np.zeros(0, bool))
        assigned = np.full(len(frame), -1, int)
        if keep.any() and active:
            C = np.full((len(frame), len(active)), np.inf)
            for i, p in enumerate(frame):
                if not keep[i]:
                    continue
                for j, tid in enumerate(active):
                    color = appearances[f][i] if appearances is not None else None
                    C[i, j] = _association_cost(predictions[tid], p, centers[i], sizes[i], max_cost, color, extents[i])
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
            tid = assigned[i]
            history = tracks[tid]["history"] if tid in tracks else []
            history = (history + [(f, centers[i], sizes[i])])[-5:]
            appearance = appearances[f][i] if appearances is not None else None
            if tid in tracks and tracks[tid]["appearance"] is not None:
                # Keep a clothing reference over time; one occluded frame must
                # not replace it with the person standing in front.
                appearance = combine_appearances([tracks[tid]["appearance"]] * 9 + [appearance])
            extent_history = tracks[tid]["extents"] if tid in tracks else []
            if np.isfinite(extents[i]) and extents[i] > 0:
                extent_history = (extent_history + [extents[i]])[-5:]
            tracks[tid] = {"pose": frame[i], "last": f, "history": history,
                           "appearance": appearance, "extents": extent_history}
        out.append(assigned)
    return out, next_id


def _source_tracks(source_ids: list) -> tuple[list, int]:
    """Renumber persistent detector ids by first appearance, never by position."""
    mapping = {}
    out = []
    for frame_ids in source_ids:
        assigned = np.full(len(frame_ids), -1, int)
        for i, tid in enumerate(frame_ids):
            if tid >= 0:
                if tid not in mapping:
                    mapping[tid] = len(mapping)
                assigned[i] = mapping[tid]
        out.append(assigned)
    return out, len(mapping)


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


def stable_track_ids(det, merge_groups=None, return_info: bool = False):
    """Final track ids: automatic stitching, frozen learned joins, then manual merges.

    This is the one place that defines "who is person N" for a run.  The Find people
    list, the composite analysis and the dancer-filtering features must all use it, or
    labels and predictions attach to different people than the ones on screen.
    """
    t = det.t
    fps = float((len(t) - 1) / (t[-1] - t[0])) if len(t) > 1 and t[-1] > t[0] else 30.0
    source_ids = det.ids if getattr(det, "ids_persistent", False) else None
    path = getattr(det, "path", None)
    key = (str(path), json.dumps(baseline_stamp(path))) if path is not None else None
    if key is not None and key in _TRACK_CACHE:
        _TRACK_CACHE.move_to_end(key)
        ids, n, info = _TRACK_CACHE[key]
    else:
        appearances = load_appearances(det) if source_ids is None else None
        boxes = [[p.get("bbox") for p in frame] for frame in det.raw]
        ids, n, info = stabilize(det.kps, fps=fps, source_ids=source_ids, appearances=appearances,
                                boxes=boxes, return_info=True)
        if key is not None:
            _TRACK_CACHE[key] = (ids, n, info)
            while len(_TRACK_CACHE) > 8:
                _TRACK_CACHE.popitem(last=False)
    ids = [a.copy() for a in ids]
    info = {"raw": [a.copy() for a in info["raw"]], "fragments": list(info["fragments"])}
    from .identity_learning import prediction_groups
    learned = prediction_groups(path) if path is not None else []
    for merge_stage in (learned, merge_groups):
        groups = [sorted(set(g)) for g in (merge_stage or []) if len(set(g)) > 1]
        if groups:
            ids, n = merge_track_ids(det.kps, ids, groups)
            target = {i: i for i in range(len(info["fragments"]))}
            for group in groups:
                for tid in group:
                    target[tid] = group[0]
            info["fragments"] = [sum(info["fragments"][t] for t in target if target[t] == k)
                                 for k in sorted(set(target.values()))]
    return (ids, n, info) if return_info else (ids, n)
