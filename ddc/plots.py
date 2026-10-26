"""Timeline figure: deviation over time per dancer, plus a body-group heatmap."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .deviation import Deviation
from .palette import dancer_color
from .render import DEV_MAX
from .smoothing import nan_moving_mean

MAX_POINTS = 2000   # timeline samples sent to the browser; longer runs are binned


def _bin(x: np.ndarray, stride: int) -> np.ndarray:
    """Mean over consecutive bins of `stride` samples (NaN-aware) along axis 0."""
    if stride <= 1:
        return x
    n = -(-len(x) // stride) * stride
    pad = np.full((n - len(x),) + x.shape[1:], np.nan)
    b = np.concatenate([x, pad]).reshape((n // stride, stride) + x.shape[1:])
    ok = np.isfinite(b)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ok.any(1), np.where(ok, b, 0).sum(1) / ok.sum(1), np.nan)


def _js(v: np.ndarray) -> list:
    return [None if not np.isfinite(x) else round(float(x), 4) for x in v]


def _smooth(x: np.ndarray, win: int = 9) -> np.ndarray:
    return nan_moving_mean(x, win)


def tracks_figure(tracks) -> go.Figure:
    """Gantt of when each detected person is visible (gaps > 0.5 s break the bar).

    `tracks`: `analyze.TrackInfo` list. Lets you see where fragments were joined
    and which people overlap in time (overlapping people can't be the same one)."""
    fig = go.Figure()
    for k, tr in enumerate(tracks):
        xs, ys = [], []
        for t0, t1 in tr.segments:
            xs += [t0, t1, None]
            ys += [tr.track_id + 1] * 3
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=f"Person {tr.track_id + 1}", showlegend=False,
            line=dict(color=dancer_color(k), width=10), connectgaps=False,
            hovertemplate=f"Person {tr.track_id + 1}: {tr.frames_present} frames, "
                          f"{tr.fragments} fragment(s) joined<extra></extra>"))
    n = max(len(tracks), 1)
    fig.update_yaxes(autorange="reversed", dtick=1, title_text="person",
                     range=[n + 0.5, 0.5])
    fig.update_xaxes(title_text="time (s)")
    fig.update_layout(height=max(160, 40 * n + 80), margin=dict(l=60, r=20, t=20, b=40),
                      title=None)
    return fig


def timeline_figure(dev: Deviation, t: np.ndarray, height: int | None = None) -> go.Figure:
    """`height` (px) defaults to a size that grows with the number of heatmap rows, clamped
    so the figure still fits next to the video on a laptop screen."""
    T = dev.score.shape[1]
    source = {"composite": "composite", "solo": "solo reference",
              "dancer": f"Dancer {(dev.reference_track or 0) + 1} (reference)"}[dev.mode]
    scored = [d for d in range(T) if d != dev.reference_track]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.08,
        subplot_titles=(f"Deviation from {source} (torso lengths)", "Deviation by body group"),
    )
    stride = max(1, -(-len(t) // MAX_POINTS))
    tx = _bin(np.asarray(t, float), stride).tolist()
    for d in scored:
        fig.add_trace(go.Scatter(
            x=tx, y=_js(_bin(_smooth(dev.score[:, d]), stride)), mode="lines", name=f"Dancer {d + 1}",
            line=dict(color=dancer_color(d), width=2), connectgaps=False,
            hovertemplate="t=%{x:.2f}s  dev=%{y:.2f}<extra>Dancer " + str(d + 1) + "</extra>"),
            row=1, col=1)

    labels, rows = [], []
    for d in scored:
        for g, v in dev.group.items():
            labels.append(f"D{d + 1} {g}")
            rows.append(_bin(_smooth(v[:, d]), stride))
    fig.add_trace(go.Heatmap(
        x=tx, y=labels, z=[_js(r) for r in rows], zmin=0, zmax=DEV_MAX,
        colorscale=[[0, "#2fbf5b"], [0.5, "#f2d02e"], [1, "#e5383b"]],
        colorbar=dict(title="dev", len=0.45, y=0.2), hoverongaps=False,
        hovertemplate="t=%{x:.2f}s  %{y}: %{z:.2f}<extra></extra>"), row=2, col=1)

    # Shade spans where deviation is uninformative: the composite has < 2 contributors,
    # or (dancer/solo reference) no one is being scored against the reference.
    low = dev.n_present < (2 if dev.mode == "composite" else 1)
    if dev.mode != "composite":
        low = low | ~np.isfinite(dev.composite[:, :, 0]).any(-1)
    start = None
    for i, flag in enumerate(list(low) + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            fig.add_vrect(x0=t[start], x1=t[min(i, len(t) - 1)], fillcolor="gray",
                          opacity=0.2, line_width=0, row="all", col=1)
            start = None

    fig.update_yaxes(autorange="reversed", row=2, col=1)
    fig.update_xaxes(title_text="time (s)", row=2, col=1)
    if height is None:
        height = int(min(720, max(420, 300 + 16 * len(labels))))
    fig.update_layout(height=height, margin=dict(l=70, r=10, t=50, b=40), hovermode="x unified",
                      legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))
    return fig
