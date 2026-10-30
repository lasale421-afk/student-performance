"""Figures for the notebook and the report (needs the optional "notebook" extra: matplotlib)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

# Colours from a colour-blind-checked palette; text stays in neutral inks.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def _style(ax: plt.Axes) -> None:
    """Make axes recessive: light grid, no top/right frame, muted tick labels."""
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def plot_precision_recall(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float, path: Path
) -> None:
    """Draw the precision-recall curve and mark the chosen threshold.

    Args:
        y_true: True 0/1 labels.
        y_proba: Predicted probability of class 1 (out-of-fold in this project).
        threshold: The chosen decision threshold, shown as a point.
        path: PNG file to write (its folder is created if needed).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    chosen = np.searchsorted(thresholds, threshold)  # index of the chosen threshold
    base_rate = float(np.mean(y_true))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(recall, precision, color=BLUE, linewidth=2, label="chosen model (out-of-fold)")
    ax.axhline(base_rate, color=MUTED, linewidth=1, linestyle="--", label="random guess")
    ax.scatter(
        recall[chosen],
        precision[chosen],
        s=64,
        color=ORANGE,
        zorder=3,
        edgecolor="white",
        linewidth=2,
        label=f"threshold {threshold:.3f}",
    )
    ax.annotate(
        f"recall {recall[chosen]:.2f}, precision {precision[chosen]:.2f}",
        (recall[chosen], precision[chosen]),
        xytext=(-10, -22),
        textcoords="offset points",
        ha="right",
        color=TEXT,
        fontsize=9,
    )
    ax.set_xlabel("Recall (share of failing students flagged)", color=TEXT)
    ax.set_ylabel("Precision (share of flagged who fail)", color=TEXT)
    ax.set_title("Precision-recall trade-off", color=TEXT, loc="left")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=9)
    _style(ax)
    _save(fig, path)


def plot_confusion_matrix(scores: dict, path: Path) -> None:
    """Draw a 2 x 2 confusion matrix from the counts tn, fp, fn, tp.

    Args:
        scores: Dict with the keys tn, fp, fn, tp (e.g. from final_scores).
        path: PNG file to write.
    """
    counts = np.array([[scores["tn"], scores["fp"]], [scores["fn"], scores["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(counts, cmap="Blues", vmin=0, vmax=counts.max() * 1.6)
    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                str(counts[row, column]),
                ha="center",
                va="center",
                fontsize=16,
                color=TEXT,
            )
    ax.set_xticks([0, 1], ["predicted pass", "predicted fail"])
    ax.set_yticks([0, 1], ["true pass", "true fail"])
    ax.tick_params(colors=MUTED, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Test set confusion matrix", color=TEXT, loc="left")
    _save(fig, path)


def plot_coefficients(table: pd.DataFrame, path: Path, top: int = 15) -> None:
    """Draw the largest coefficients as horizontal bars (orange raises risk, blue lowers it).

    Args:
        table: Output of coefficient_table (sorted by absolute coefficient).
        path: PNG file to write.
        top: Number of features to show.
    """
    shown = table.head(top).iloc[::-1]  # largest at the top of the chart
    colours = [ORANGE if value > 0 else BLUE for value in shown["coefficient"]]
    fig, ax = plt.subplots(figsize=(6.5, 0.35 * top + 1))
    ax.barh(shown["feature"], shown["coefficient"], color=colours, height=0.6)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_xlabel("Coefficient (log-odds of failing)", color=TEXT)
    ax.set_title(
        f"Top {top} coefficients: orange raises risk, blue lowers it", color=TEXT, loc="left"
    )
    _style(ax)
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def _save(fig: plt.Figure, path: Path) -> None:
    """Write the figure as a PNG, creating the folder if needed, and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
