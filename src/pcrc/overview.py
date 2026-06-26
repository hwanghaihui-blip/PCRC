"""Static overview artifacts for the manuscript front matter."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from pcrc.constants import FIGURES_MAIN_DIR
from pcrc.contract import dataset_sources_frame, method_sources_frame
from pcrc.reporting import export_numbered_table
from pcrc.utils import ensure_parent


def _box(ax, xy: tuple[float, float], width: float, height: float, text: str, *, facecolor: str) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.2,
        edgecolor="#263238",
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2.0, xy[1] + height / 2.0, text, ha="center", va="center", fontsize=10)


def _arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color="#37474f",
        )
    )


def write_overview_artifacts() -> None:
    export_numbered_table(method_sources_frame(), "table1_1", "method_sources", display_method_labels=False)
    export_numbered_table(dataset_sources_frame(), "table1_2", "dataset_sources")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, (0.05, 0.62), 0.18, 0.18, "Pre-deployment\nData", facecolor="#d0f0c0")
    _box(ax, (0.30, 0.62), 0.20, 0.18, "Predictive Model\nand Set", facecolor="#ffe082")
    _box(ax, (0.57, 0.62), 0.18, 0.18, "Candidate Action\nRisk", facecolor="#ffccbc")
    _box(ax, (0.80, 0.62), 0.15, 0.18, "Decision Rule", facecolor="#c5cae9")

    _box(ax, (0.20, 0.18), 0.20, 0.20, "Deployment\nAction", facecolor="#b2dfdb")
    _box(ax, (0.47, 0.18), 0.18, 0.20, "Observed Outcome\n+ Post Coverage", facecolor="#f8bbd0")
    _box(ax, (0.72, 0.18), 0.18, 0.20, "Threshold Update\nScore and Risk", facecolor="#d1c4e9")

    _arrow(ax, (0.23, 0.71), (0.30, 0.71))
    _arrow(ax, (0.50, 0.71), (0.57, 0.71))
    _arrow(ax, (0.75, 0.71), (0.80, 0.71))
    _arrow(ax, (0.875, 0.62), (0.875, 0.40))
    _arrow(ax, (0.80, 0.28), (0.65, 0.28))
    _arrow(ax, (0.47, 0.28), (0.40, 0.28))
    _arrow(ax, (0.30, 0.38), (0.30, 0.62))
    _arrow(ax, (0.56, 0.38), (0.56, 0.62))
    _arrow(ax, (0.72, 0.28), (0.90, 0.28))
    _arrow(ax, (0.81, 0.38), (0.81, 0.62))

    ax.text(0.50, 0.92, "Closed-loop post-decision calibration", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(0.30, 0.49, "calibrate", ha="center", va="center", fontsize=9, color="#455a64")
    ax.text(0.56, 0.49, "action-conditioned risk", ha="center", va="center", fontsize=9, color="#455a64")
    ax.text(0.81, 0.49, "closed-loop update", ha="center", va="center", fontsize=9, color="#455a64")

    pdf_path = ensure_parent(FIGURES_MAIN_DIR / "fig1_1_pcrc_closed_loop_overview.pdf")
    png_path = ensure_parent(FIGURES_MAIN_DIR / "fig1_1_pcrc_closed_loop_overview.png")
    fig.tight_layout()
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
