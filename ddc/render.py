"""Overlay video: each dancer's skeleton colored by deviation, with the composite as a ghost."""

from __future__ import annotations

import contextlib
from pathlib import Path

import cv2
import numpy as np

from .normalize import denormalize
from .skeleton import CONF_THR, EDGE_IDX

DEV_MAX = 0.5   # torso lengths at which a joint is drawn fully red
TRACK_COLORS = [(255, 170, 60), (90, 200, 255), (200, 120, 255), (120, 255, 160),
                (255, 120, 160), (200, 200, 90)]  # BGR, for dancer labels


def dev_color(d: float) -> tuple:
    """Green -> yellow -> red (BGR). NaN -> gray."""
    if not np.isfinite(d):
        return (140, 140, 140)
    x = float(np.clip(d / DEV_MAX, 0, 1))
    if x < 0.5:
        return (60, int(220), int(60 + 2 * x * 195))      # green -> yellow
    return (60, int(220 - (x - 0.5) * 2 * 160), 255)     # yellow -> red


def _frame_source(video_path, n_frames):
    """Yield BGR frames from video_path (dimmed if it is an overlay), else None forever."""
    cap = cv2.VideoCapture(str(video_path)) if video_path else None
    try:
        for _ in range(n_frames):
            ok, fr = (cap.read() if cap is not None else (False, None))
            yield fr if ok else None
    finally:
        if cap is not None:
            cap.release()


def render_overlay(dense, dev, origin, scale, rot, t, out_path, video_path=None,
                   dim: float = 1.0, fps: float | None = None):
    """Write an mp4. `dim`<1 darkens the background (use ~0.35 for the pre-drawn overlay).

    Background frames are read frame-index-aligned with `dense`/`dev` (frame f of
    the video == frame f of the tracking data). If video_path is a variable-frame-
    rate source that isn't already the run's own (CFR) overlay.mp4, that alignment
    drifts over time -- the same issue `_cfr_source` fixes for the analysis pass --
    so we resample it to CFR first via the same helper.
    """
    from video_pipeline import _cfr_source, _reencode_h264

    F, T = dense.shape[:2]
    with _cfr_source(str(video_path)) if video_path else contextlib.nullcontext(None) as vp:
        size = None
        if vp:
            cap = cv2.VideoCapture(str(vp))
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            vfps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            fps = fps or (vfps if vfps > 0 else None)
        if size is None or size[0] == 0:
            xs, ys = dense[..., 0], dense[..., 1]
            size = (int(np.nanmax(xs)) + 80, int(np.nanmax(ys)) + 80)
        if not fps:
            fps = (F - 1) / (t[-1] - t[0]) if F > 1 and t[-1] > t[0] else 30.0
        w, h = size
        thick = max(2, h // 360)

        raw = Path(out_path).with_suffix(".raw.mp4")
        vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f, bg in enumerate(_frame_source(vp, F)):
            canvas = np.zeros((h, w, 3), np.uint8) if bg is None else cv2.resize(bg, (w, h))
            if bg is not None and dim < 1.0:
                canvas = (canvas * dim).astype(np.uint8)
            for d in range(T):
                if not np.isfinite(dense[f, d, :, 0]).any() or not np.isfinite(origin[f, d]).all() \
                        or not np.isfinite(scale[f, d]):
                    continue
                ref = dev.composite[f, d] if dev.leave_one_out else dev.composite[f]
                ghost = denormalize(ref, origin[f, d], scale[f, d], rot[f, d])
                pts = dense[f, d, :, :2]
                ok = dense[f, d, :, 2] >= CONF_THR
                # composite ghost (behind)
                for a, b in EDGE_IDX:
                    if np.isfinite(ghost[[a, b]]).all():
                        cv2.line(canvas, tuple(ghost[a].astype(int)), tuple(ghost[b].astype(int)),
                                 (235, 235, 235), max(1, thick - 1), cv2.LINE_AA)
                # deviation vectors
                for j in range(len(pts)):
                    if ok[j] and np.isfinite(ghost[j]).all():
                        cv2.line(canvas, tuple(pts[j].astype(int)), tuple(ghost[j].astype(int)),
                                 dev_color(dev.joint[f, d, j]), 1, cv2.LINE_AA)
                # dancer skeleton, edges colored by the worse endpoint
                for a, b in EDGE_IDX:
                    if ok[a] and ok[b]:
                        c = dev_color(np.nanmax([dev.joint[f, d, a], dev.joint[f, d, b]]))
                        cv2.line(canvas, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)),
                                 c, thick + 1, cv2.LINE_AA)
                for j in range(len(pts)):
                    if ok[j]:
                        cv2.circle(canvas, tuple(pts[j].astype(int)), thick + 2,
                                   dev_color(dev.joint[f, d, j]), -1, cv2.LINE_AA)
                # label
                head = pts[ok][:, :2]
                top = head[np.argmin(head[:, 1])].astype(int)
                s = dev.score[f, d]
                label = f"D{d + 1}  {s:.2f}" if np.isfinite(s) else f"D{d + 1}"
                cv2.putText(canvas, label, (int(top[0]) - 20, max(int(top[1]) - 16, 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, max(0.5, h / 1400), TRACK_COLORS[d % len(TRACK_COLORS)],
                            2, cv2.LINE_AA)
            cv2.putText(canvas, f"t={t[f]:.2f}s  dancers={int(dev.n_present[f])}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            vw.write(canvas)
        vw.release()

    final = _reencode_h264(raw, Path(out_path))
    return str(final)
