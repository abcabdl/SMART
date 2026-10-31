from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt


FACE_COLOR = "#fcfbf7"
GRID_COLOR = "#ddd8ca"
SPINE_COLOR = "#8a8478"
TEXT_COLOR = "#1f2428"
MUTED_TEXT = "#5e5a52"

BLUE = "#184e77"
ORANGE = "#c07a2c"
GREEN = "#4c956c"
PURPLE = "#7a5c99"
RED = "#b2573f"
GRAY = "#9d978c"
LIGHT_GRAY = "#cfc8ba"


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
            "axes.titlesize": 15,
            "axes.labelsize": 12.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
            "figure.facecolor": FACE_COLOR,
            "axes.facecolor": FACE_COLOR,
            "savefig.facecolor": FACE_COLOR,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
        }
    )


def style_axes(ax, *, grid_axis: Literal["x", "y", "both", "none"] = "y") -> None:
    ax.set_facecolor(FACE_COLOR)
    ax.grid(False)
    if grid_axis != "none":
        ax.grid(
            True,
            axis=grid_axis,
            color=GRID_COLOR,
            linewidth=0.9,
            alpha=0.92,
        )
    for side, spine in ax.spines.items():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(1.0)
        if side in {"top", "right"}:
            spine.set_alpha(0.75)


def style_legend(legend) -> None:
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor(FACE_COLOR)
    frame.set_edgecolor(LIGHT_GRAY)
    frame.set_linewidth(0.9)
    frame.set_alpha(0.97)
