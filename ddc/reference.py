"""Reference poses for deviation: one dancer from the clip, or a separate solo video.

Both return a (F,17,2) pose sequence in the same hip-centered, torso-scaled space
as `normalize()`, so it can stand in for the composite in `compute_deviation`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import io as ddc_io
from .composite import _smooth_gaps
from .normalize import normalize
from .tracking import stabilize, to_dense


def reference_from_track(norm: np.ndarray, idx: int, smooth: int = 5) -> np.ndarray:
    """norm: (F,T,17,3). One dancer's normalized pose as the reference, (F,17,2)."""
    return _smooth_gaps(norm[:, idx, :, :2].copy(), smooth)


def _resample(ref: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Linear-interpolate (Fs,17,2) from t_src onto t_dst. NaN outside t_src's range
    and wherever either neighbouring source sample is NaN."""
    out = np.full((len(t_dst),) + ref.shape[1:], np.nan)
    inside = (t_dst >= t_src[0]) & (t_dst <= t_src[-1])
    hi = np.clip(np.searchsorted(t_src, t_dst), 1, len(t_src) - 1)
    lo = hi - 1
    span = np.maximum(t_src[hi] - t_src[lo], 1e-9)
    w = np.clip((t_dst - t_src[lo]) / span, 0.0, 1.0)[:, None, None]
    out[inside] = (ref[lo] * (1 - w) + ref[hi] * w)[inside]
    return out


def reference_from_solo(solo_run_dir, t_group: np.ndarray, offset: float = 0.0,
                        rotate: bool = False, smooth: int = 5) -> np.ndarray:
    """Pose sequence of the solo video's main dancer, resampled onto the group
    video's timestamps. `offset` is seconds into the solo that lines up with t=0
    of the group video (positive = solo starts earlier / group is delayed)."""
    det = ddc_io.load_jsonl(Path(solo_run_dir) / "tracking.jsonl")
    if det.n_frames < 2:
        raise ValueError("solo reference video has too few frames")
    ids, n_tracks = stabilize(det.kps, fps=(det.n_frames - 1) / max(det.t[-1] - det.t[0], 1e-9))
    if n_tracks == 0:
        raise ValueError("no persistent person found in the solo reference video")
    dense = to_dense(det.kps, ids, n_tracks)
    # The solo dancer is whoever is tracked longest (bystanders are shorter-lived).
    main = int(np.argmax(np.isfinite(dense[:, :, 0, 0]).sum(0)))
    norm, *_ = normalize(dense[:, [main]], rotate=rotate)
    ref = _smooth_gaps(norm[:, 0, :, :2], smooth)
    return _resample(ref, det.t, np.asarray(t_group, float) + offset)
