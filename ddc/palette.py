"""One color per dancer, shared by the timeline plots and the overlay video labels.

30 entries so a full team gets distinct colors; the first six are the original palette."""

from __future__ import annotations

DANCER_COLORS = [
    "#f5a03c", "#3cc8ff", "#a078ff", "#3cff9e", "#ff789e", "#c8c85a",
    "#ff5a5a", "#5a8cff", "#ffd23c", "#2ad4c0", "#e05aff", "#9ee05a",
    "#ff9e6e", "#6ee0ff", "#c89eff", "#b4ff3c", "#ff3cc8", "#3c96c8",
    "#e0b48c", "#8cffd2", "#ffb4e0", "#8c8cff", "#d2a03c", "#50c878",
    "#ff6e3c", "#a0d2ff", "#d25aa0", "#78a03c", "#f0f0a0", "#b46ee0",
]


def dancer_color(i: int) -> str:
    """Hex color for dancer/track index `i` (wraps after 30)."""
    return DANCER_COLORS[i % len(DANCER_COLORS)]


def dancer_bgr(i: int) -> tuple[int, int, int]:
    """Same color as `dancer_color`, as an OpenCV BGR tuple."""
    h = dancer_color(i).lstrip("#")
    r, g, b = (int(h[k:k + 2], 16) for k in (0, 2, 4))
    return (b, g, r)
