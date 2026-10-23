"""Per-dancer deviation from the composite pose."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .composite import build_composite
from .skeleton import GROUPS, LIMB_IDX


@dataclass
class Deviation:
    joint: np.ndarray        # (F,T,17) distance in torso lengths (NaN where undefined)
    angle: np.ndarray        # (F,T,8) limb angle difference in degrees
    score: np.ndarray        # (F,T) frame score (mean joint distance, torso lengths)
    group: dict              # name -> (F,T) mean joint distance for that body group
    composite: np.ndarray    # (F,17,2) or (F,T,17,2) if leave-one-out
    n_present: np.ndarray    # (F,) dancers visible per frame
    leave_one_out: bool
    mode: str = "composite"  # "composite" | "dancer" | "solo"
    reference_track: int | None = None   # dancer index used as reference (mode "dancer")
    tolerance: int = 0       # frames each dancer may lead/lag the reference without penalty
    lag: np.ndarray | None = None      # (F,T) offset s of the matched reference frame f+s (- = dancer behind)
    matched: np.ndarray | None = None  # (F,T,17,2) reference pose each dancer frame was scored against

    def ref_pose(self, f: int, d: int) -> np.ndarray:
        """(17,2) reference pose dancer `d` was compared with at frame `f`."""
        if self.matched is not None:
            return self.matched[f, d]
        return self.composite[f, d] if self.leave_one_out else self.composite[f]


def _limb_angles(p: np.ndarray) -> np.ndarray:
    """p (...,17,2) -> (...,8) limb direction angles (radians)."""
    out = []
    for a, b in LIMB_IDX:
        d = p[..., b, :] - p[..., a, :]
        out.append(np.arctan2(d[..., 1], d[..., 0]))
    return np.stack(out, axis=-1)


def _shift(ref: np.ndarray, s: int) -> np.ndarray:
    """out[f] = ref[f + s], NaN where that falls outside the clip."""
    out = np.full_like(ref, np.nan)
    F = len(ref)
    if s >= 0:
        out[:F - s] = ref[s:]
    else:
        out[-s:] = ref[:F + s]
    return out


def _match_in_window(xy: np.ndarray, ok: np.ndarray, ref: np.ndarray, tolerance: int):
    """For each (frame, dancer) pick the reference frame within +-`tolerance` frames with
    the smallest mean joint distance. Ties go to the smaller offset, so 0 wins when equal.
    Returns (matched (F,T,17,2), lag (F,T))."""
    F, T = xy.shape[:2]
    best = np.full((F, T), np.inf)
    matched = np.full_like(ref, np.nan)
    lag = np.zeros((F, T), int)
    for s in sorted(range(-tolerance, tolerance + 1), key=lambda s: (abs(s), s)):
        cand = _shift(ref, s)
        dist = np.linalg.norm(xy - cand, axis=-1)
        dist[~ok] = np.nan
        with np.errstate(all="ignore"):
            score = np.nanmean(dist, axis=-1)
        better = np.isfinite(score) & (score < best)
        best[better] = score[better]
        matched[better] = cand[better]
        lag[better] = s
    return matched, lag


def compute_deviation(norm: np.ndarray, leave_one_out: bool | None = None,
                      smooth: int = 5, reference: np.ndarray | None = None,
                      reference_track: int | None = None, tolerance: int = 0) -> Deviation:
    """`reference` (F,17,2, normalized space) replaces the composite as the target pose.
    `reference_track` marks which dancer it came from; that dancer is not scored.

    `tolerance` (frames) relaxes timing: each dancer frame is scored against the
    closest-matching reference frame within +-`tolerance`, so being slightly early or
    late isn't counted as a pose error. 0 compares simultaneous frames only."""
    tolerance = max(0, int(tolerance))
    F, T = norm.shape[:2]
    present = np.isfinite(norm[:, :, :, 0]).any(-1)          # (F,T)
    if reference is None:
        if leave_one_out is None:
            leave_one_out = T >= 3
        comp = build_composite(norm, smooth=smooth, leave_one_out=leave_one_out)
        mode = "composite"
    else:
        if reference_track is not None:
            present[:, reference_track] = False
        leave_one_out, comp = False, reference
        mode = "dancer" if reference_track is not None else "solo"
    n_present = present.sum(1)

    ref = comp if leave_one_out else np.broadcast_to(comp[:, None], (F, T) + comp.shape[1:])
    matched = lag = None
    if tolerance:
        matched, lag = _match_in_window(norm[..., :2], norm[..., 2] >= 0.3, ref, tolerance)
        ref = matched
    diff = norm[..., :2] - ref
    joint = np.linalg.norm(diff, axis=-1)
    joint[norm[..., 2] < 0.3] = np.nan
    if reference_track is not None:
        joint[:, reference_track] = np.nan

    d_ang = np.degrees(np.abs(np.angle(np.exp(1j * (_limb_angles(norm[..., :2]) - _limb_angles(ref))))))
    if reference_track is not None:
        d_ang[:, reference_track] = np.nan
    with np.errstate(all="ignore"):
        score = np.nanmean(joint, axis=-1)
        group = {g: np.nanmean(joint[..., idx], axis=-1) for g, idx in GROUPS.items()}
    return Deviation(joint, d_ang, score, group, comp, n_present, leave_one_out,
                     mode, reference_track, tolerance, lag, matched)


def worst_moments(dev: Deviation, t: np.ndarray, top_n: int = 3, min_sep_s: float = 1.0):
    """Per dancer: list of (time, score, worst_joint_idx) for the top_n separated peaks."""
    out = {}
    for d in range(dev.score.shape[1]):
        s = np.nan_to_num(dev.score[:, d], nan=-1)
        picks = []
        for f in np.argsort(-s):
            if s[f] < 0:
                break
            if all(abs(t[f] - pt) >= min_sep_s for pt, _, _ in picks):
                jr = np.nan_to_num(dev.joint[f, d], nan=-1)
                picks.append((float(t[f]), float(s[f]), int(np.argmax(jr))))
            if len(picks) >= top_n:
                break
        out[d] = picks
    return out
