"""Timeline figure: deviation over time per dancer, plus a body-group heatmap."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .deviation import Deviation
from .render import DEV_MAX

COLORS = ["#f5a03c", "#3cc8ff", "#a078ff", "#3cff9e", "#ff789e", "#c8c85a"]


def _smooth(x: np.ndarray, win: int = 9) -> np.ndarray:
    out = np.full_like(x, np.nan)
    h = win // 2
    for i in range(len(x)):
        seg = x[max(0, i - h): i + h + 1]
        if np.isfinite(seg).any():
            out[i] = np.nanmean(seg)
    return out


def timeline_figure(dev: Deviation, t: np.ndarray) -> go.Figure:
    T = dev.score.shape[1]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.08,
        subplot_titles=("Deviation from composite (torso lengths)", "Deviation by body group"),
    )
    for d in range(T):
        fig.add_trace(go.Scatter(
            x=t.tolist(), y=[None if not np.isfinite(v) else round(float(v), 4) for v in _smooth(dev.score[:, d])], mode="lines", name=f"Dancer {d + 1}",
            line=dict(color=COLORS[d % len(COLORS)], width=2), connectgaps=False,
            hovertemplate="t=%{x:.2f}s  dev=%{y:.2f}<extra>Dancer " + str(d + 1) + "</extra>"),
            row=1, col=1)

    labels, rows = [], []
    for d in range(T):
        for g, v in dev.group.items():
            labels.append(f"D{d + 1} {g}")
            rows.append(_smooth(v[:, d]))
    fig.add_trace(go.Heatmap(
        x=t.tolist(), y=labels,
        z=[[None if not np.isfinite(v) else round(float(v), 4) for v in r] for r in rows], zmin=0, zmax=DEV_MAX,
        colorscale=[[0, "#2fbf5b"], [0.5, "#f2d02e"], [1, "#e5383b"]],
        colorbar=dict(title="dev", len=0.45, y=0.2), hoverongaps=False,
        hovertemplate="t=%{x:.2f}s  %{y}: %{z:.2f}<extra></extra>"), row=2, col=1)

    # Shade spans where the composite has < 2 contributors (deviation is uninformative).
    low = dev.n_present < 2
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
    fig.update_layout(height=560, margin=dict(l=70, r=20, t=50, b=40), hovermode="x unified",
                      legend=dict(orientation="h", y=1.12))
    return fig
