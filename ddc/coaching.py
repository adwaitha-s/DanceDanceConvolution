"""Deterministic coaching from the existing normalized joint distances.

No language model or extra inference is required. Missing comparisons stay NaN.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .deviation import Deviation
from .skeleton import BODY, CONF_THR, GROUPS

MIN_JOINTS = 6
NOTICE_DISTANCE = 0.15  # torso lengths; a UI heuristic, not a validated tolerance
SCORE_EXPLANATION = (
    "Movement similarity (0–100) is a heuristic, not a scientifically validated accuracy "
    "percentage. Score = 100 / (1 + mean deviation in torso lengths). Each valid frame "
    "uses the mean of its comparable joint distances; dancer scores use the mean across "
    "valid frames. Overall is the equal-weight mean of available dancer scores. "
    "At least 6 of 17 joints must be comparable per frame. Missing data is excluded, "
    "never treated as a perfect match or a zero score; reference dancers are not scored. "
    "Coverage counts valid frames out of all clip frames. Sparse coverage can inflate scores. "
    "Composite joints need another dancer visible at that joint; a lone dancer cannot "
    "score against themselves. With two dancers, composite differences are symmetric. "
    "Scores depend on reference choice, smoothing, camera angle, tracking, timing and "
    "normalization, and do not measure rhythm, artistry or 3D technique. Directions describe "
    "relative 2D image height after hip/scale normalization, not depth; directions are "
    "omitted when torso rotation is normalized. Differences below 0.15 torso lengths "
    "are described as close (an adjustable design heuristic)."
)


@dataclass
class Coaching:
    joint: np.ndarray       # comparable distances only (F,T,17)
    deviation: np.ndarray   # mean comparable joint distance (F,T)
    frame_score: np.ndarray
    dancers: list[dict]
    overall: float | None


def similarity(distance):
    """Monotone bounded mapping; undefined/negative distances remain undefined."""
    x = np.asarray(distance, dtype=float)
    out = np.full_like(x, np.nan)
    ok = np.isfinite(x) & (x >= 0)
    out[ok] = 100.0 / (1.0 + x[ok])
    return out


def build_coaching(dev: Deviation, norm: np.ndarray) -> Coaching:
    joint = dev.joint.copy()
    observed = np.isfinite(norm[..., :2]).all(-1) & np.isfinite(norm[..., 2]) & (norm[..., 2] >= CONF_THR)
    comparable = observed.copy()
    if dev.mode == "composite":
        # Require a real peer now, even if smoothing filled a reference dropout.
        comparable &= observed.sum(axis=1)[:, None, :] >= 2
    elif dev.reference_track is not None:
        comparable &= observed[:, dev.reference_track, None, :]
        comparable[:, dev.reference_track] = False
    joint[~comparable | ~np.isfinite(joint)] = np.nan
    count = np.isfinite(joint).sum(-1)
    distance = np.full(count.shape, np.nan, dtype=float)
    np.divide(np.nansum(joint, axis=-1), count, out=distance, where=count >= MIN_JOINTS)
    # Feedback uses the same eligibility as scores.
    joint[count < MIN_JOINTS] = np.nan
    dancers = []
    for d in range(norm.shape[1]):
        valid = np.isfinite(distance[:, d])
        mean = float(distance[valid, d].mean()) if valid.any() else None
        groups = {}
        for name, indices in GROUPS.items():
            values = joint[:, d, indices]
            finite = values[np.isfinite(values)]
            if finite.size:
                groups[name] = float(finite.mean())
        closest = min(groups, key=groups.get) if groups else None
        furthest = max(groups, key=groups.get) if groups else None
        counts = np.isfinite(joint[:, d]).sum(0)
        means = np.full(len(BODY), np.nan)
        np.divide(np.nansum(joint[:, d], axis=0), counts, out=means, where=counts > 0)
        worst = int(np.nanargmax(means)) if np.isfinite(means).any() else None
        dancers.append(dict(
            score=float(similarity(mean)) if mean is not None else None,
            valid_frames=int(valid.sum()), total_frames=len(norm),
            coverage=float(valid.mean()) if len(norm) else 0.0,
            reference=d == dev.reference_track,
            strength=(f"Closest region: {closest} ({groups[closest]:.2f} torso lengths)"
                      if closest else "No comparable pose data"),
            improvement=(f"Review {BODY[worst].replace('_', ' ')} ({means[worst]:.2f} torso lengths); "
                         f"largest region deviation: {furthest}"
                         if worst is not None and means[worst] >= NOTICE_DISTANCE else
                         "Comparable regions are close" if groups else "More pose data needed"),
        ))
    scores = [s["score"] for s in dancers if s["score"] is not None and not s["reference"]]
    return Coaching(joint, distance, similarity(distance), dancers,
                    float(np.mean(scores)) if scores else None)


def joint_feedback(dev, norm, coaching, frame, dancer, joint, t, rotate=False):
    """Only use signed coordinates when available and aligned with image axes."""
    distance = coaching.joint[frame, dancer, joint]
    if not np.isfinite(distance):
        return "Insufficient comparable pose data."
    name = BODY[joint].replace("_", " ")
    stamp = f"at {t:.2f} seconds"
    if distance < NOTICE_DISTANCE:
        return f"Your {name} is close to the reference {stamp}."
    ref = dev.composite[frame, dancer, joint] if dev.leave_one_out else dev.composite[frame, joint]
    delta = norm[frame, dancer, joint, :2] - ref
    if not rotate and np.isfinite(delta).all() and abs(delta[1]) >= NOTICE_DISTANCE and abs(delta[1]) >= 1.5 * abs(delta[0]):
        direction = "lower" if delta[1] > 0 else "higher"
        return (f"Your {name} is {direction} than the reference {stamp} "
                "in the normalized camera view. Review its height relative to your hips.")
    return (f"Your {name} differs from the reference {stamp} by {distance:.2f} torso lengths. "
            "Review this joint in the overlay.")


def feedback_moments(dev, norm, coaching, t, rotate=False, top_n=3, min_sep_s=1.0):
    """Separated peaks of eligible deviations, with full-precision seek times."""
    rows = []
    for d in range(norm.shape[1]):
        picks = []
        values = coaching.deviation[:, d]
        for f in np.argsort(-np.nan_to_num(values, nan=-1)):
            if not np.isfinite(values[f]) or len(picks) >= top_n:
                break
            if np.nanmax(coaching.joint[f, d]) < NOTICE_DISTANCE:
                continue
            if any(abs(t[f] - t[p]) < min_sep_s for p in picks):
                continue
            picks.append(f)
            j = int(np.nanargmax(coaching.joint[f, d]))
            rows.append([d, float(t[f]), joint_feedback(dev, norm, coaching, f, d, j, t[f], rotate)])
    return sorted(rows, key=lambda row: row[1])


@dataclass(frozen=True)
class KeyMoment:
    dancers: tuple[int, int]
    start: float
    end: float
    peak: float
    mean_deviation: float
    comparable_joints: int
    region: str
    joint: str


def key_moments(norm, t, top_n=5, min_duration_s=0.25, min_sep_s=1.0):
    """Rank sustained pairwise pose mismatches, independent of reference choice.

    A candidate needs >= MIN_JOINTS confident joints shared by both dancers and
    at least one joint differing by NOTICE_DISTANCE. Runs must last 0.25 s;
    missing comparisons and large timestamp gaps break runs. Rank by the mean
    shared-joint distance at each run's peak, then keep separated peaks.
    These are simultaneous 2D pose differences, not evidence of a timing lag.
    """
    t = np.asarray(t, dtype=float)
    if len(t) != len(norm) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("key-moment timestamps must be finite, increasing and match pose frames")
    if min_duration_s < 0 or min_sep_s < 0:
        raise ValueError("key-moment durations and separation must be nonnegative")
    if len(t) < 2 or norm.shape[1] < 2 or top_n <= 0:
        return []
    observed = (np.isfinite(norm[..., :2]).all(-1) & np.isfinite(norm[..., 2])
                & (norm[..., 2] >= CONF_THR))
    gap_limit = max(0.25, 2 * float(np.median(np.diff(t))))
    candidates = []
    for a, b in combinations(range(norm.shape[1]), 2):
        shared = observed[:, a] & observed[:, b]
        # Mask coordinates before subtraction so invalid observations cannot overflow.
        xy_a = np.where(shared[..., None], norm[:, a, :, :2], 0.0)
        xy_b = np.where(shared[..., None], norm[:, b, :, :2], 0.0)
        distance = np.linalg.norm(xy_a - xy_b, axis=-1)
        distance[~shared] = np.nan
        counts = shared.sum(-1)
        mean = np.full(len(t), np.nan)
        np.divide(np.nansum(distance, axis=-1), counts, out=mean, where=counts >= MIN_JOINTS)
        mismatch = ((counts >= MIN_JOINTS)
                    & np.any(np.where(shared, distance, 0.0) >= NOTICE_DISTANCE, axis=-1))
        start = None
        for f in range(len(t) + 1):
            gap = 0 < f < len(t) and t[f] - t[f - 1] > gap_limit
            if start is not None and (f == len(t) or not mismatch[f] or gap):
                end = f - 1
                if t[end] - t[start] >= min_duration_s:
                    peak = start + int(np.argmax(mean[start:end + 1]))
                    j = int(np.nanargmax(distance[peak]))
                    regions = {}
                    for name, indices in GROUPS.items():
                        values = distance[peak, indices]
                        valid = values[np.isfinite(values)]
                        if valid.size:
                            regions[name] = float(valid.mean())
                    candidates.append(KeyMoment(
                        (a, b), float(t[start]), float(t[end]), float(t[peak]),
                        float(mean[peak]), int(counts[peak]),
                        max(regions, key=regions.get), BODY[j]))
                start = None
            if f < len(t) and mismatch[f] and start is None:
                start = f
    selected = []
    for moment in sorted(candidates, key=lambda m: (-m.mean_deviation, m.peak, m.dancers)):
        if all(abs(moment.peak - other.peak) >= min_sep_s for other in selected):
            selected.append(moment)
            if len(selected) >= top_n:
                break
    return selected
