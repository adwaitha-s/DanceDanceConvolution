"""Small plotting helpers shared by final_selection.py (matplotlib only)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

NAVY, ORANGE, GRAY, INK, INK2 = "#13294B", "#FF5F05", "#B8BDC6", "#1B1B1B", "#4A4F57"


def style():
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelcolor": INK2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#E3E5E8", "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def bar_rank(scores, k, title, xlabel, path, n=10):
    """Horizontal bars for the top-n scores; the first k are drawn in orange."""
    s = scores.head(n)[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 0.34 * len(s) + 1.0))
    ax.barh(s.index, s.values, color=[ORANGE if i >= len(s) - k else NAVY for i in range(len(s))], height=0.7)
    for i, v in enumerate(s.values):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=9, color=INK2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, s.max() * 1.15)
    ax.grid(axis="y", visible=False)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def corr_heatmap(X, cols, path):
    """Pearson correlation among the given columns (redundancy check)."""
    c = X[cols].corr().values
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="left", fontsize=9)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(cols)), cols, fontsize=9)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{c[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(c[i, j]) > 0.7 else INK)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
