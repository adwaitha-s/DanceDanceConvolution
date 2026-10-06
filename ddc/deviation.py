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


def _limb_angles(p: np.ndarray) -> np.ndarray:
    """p (...,17,2) -> (...,8) limb direction angles (radians)."""
    out = []
    for a, b in LIMB_IDX:
        d = p[..., b, :] - p[..., a, :]
        out.append(np.arctan2(d[..., 1], d[..., 0]))
    return np.stack(out, axis=-1)


def compute_deviation(norm: np.ndarray, leave_one_out: bool | None = None,
                      smooth: int = 5) -> Deviation:
    F, T = norm.shape[:2]
    present = np.isfinite(norm[:, :, :, 0]).any(-1)          # (F,T)
    n_present = present.sum(1)
    if leave_one_out is None:
        leave_one_out = T >= 3
    comp = build_composite(norm, smooth=smooth, leave_one_out=leave_one_out)

    ref = comp if leave_one_out else np.broadcast_to(comp[:, None], (F, T) + comp.shape[1:])
    diff = norm[..., :2] - ref
    joint = np.linalg.norm(diff, axis=-1)
    joint[norm[..., 2] < 0.3] = np.nan

    d_ang = np.degrees(np.abs(np.angle(np.exp(1j * (_limb_angles(norm[..., :2]) - _limb_angles(ref))))))
    with np.errstate(all="ignore"):
        score = np.nanmean(joint, axis=-1)
        group = {g: np.nanmean(joint[..., idx], axis=-1) for g, idx in GROUPS.items()}
    return Deviation(joint, d_ang, score, group, comp, n_present, leave_one_out)


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
