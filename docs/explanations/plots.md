# `src/student_risk/plots.py`: explained line by line

**Role:** draw the three figures of the notebook and report:

- `precision_recall_oof.png`;
- `confusion_matrix_test.png`;
- `coefficients.png`.

All are saved in `reports/figures/`.

**Why a separate module:** it needs matplotlib, which is in the **optional** `notebook` extra
(`pip install -e ".[notebook]"`). The CLI, the tests and the rest of the package never import
`plots.py`. So the main install (`pip install -e .`) stays light and works without matplotlib.
We checked this in fresh environments: matplotlib was absent, and all 19 tests and the CLI
passed. If `plots.py` were merged into `evaluate.py`, every import of `evaluate` would require
matplotlib.

The module computes **no metric of its own**, except the curve points from scikit-learn's
`precision_recall_curve` (the same function `choose_threshold` uses). It only draws.

---

## Imports and constants

```python
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
```

- `matplotlib.pyplot` is the plotting library.
- **The colours:** blue and orange are the first two colours of a palette checked for
  colour-blind readers, taken from the assistant's chart guidelines. Blue/orange stays
  distinguishable for the common forms of colour blindness, unlike red/green **[from prior
  knowledge, to verify]**.
- Text and axes use neutral dark and grey, so that colour only carries meaning on the data.
- **No backend setting:** when there is no screen (for example in `nbconvert`), matplotlib picks
  a file-only backend by itself **[from prior knowledge, to verify]**. The figures were produced
  this way in our runs.

## `_style`

```python
def _style(ax: plt.Axes) -> None:
    """Make axes recessive: light grid, no top/right frame, muted tick labels."""
```

This applies a light grid behind the data (`set_axisbelow`), removes the top and right frame
lines, and greys the remaining axes. The data then stands out and the decoration fades.

The leading underscore means "internal helper, not part of the public interface". ruff therefore
does not require the full Google docstring, but it still has a one-line docstring.

## `plot_precision_recall`

```python
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    chosen = np.searchsorted(thresholds, threshold)  # index of the chosen threshold
    base_rate = float(np.mean(y_true))
```

- These are the same curve points as in `choose_threshold`, from the out-of-fold probabilities.
- `np.searchsorted` finds the position of our threshold in the sorted list of thresholds. The
  threshold *is* one of them, because `choose_threshold` returned an element of that list. So
  `precision[chosen]` and `recall[chosen]` are exactly its values: 0.31 and 0.80 on the figure,
  the same as `threshold_report.csv`.
- `base_rate`, the fail rate, is the precision of a random guess. It is drawn as a dashed line,
  so the reader sees how far above chance the model is.

```python
    ax.plot(recall, precision, ...)                  # the curve
    ax.axhline(base_rate, ..., linestyle="--")       # random-guess level
    ax.scatter(recall[chosen], precision[chosen], ...)  # the chosen threshold, in orange
    ax.annotate(f"recall {...:.2f}, precision {...:.2f}", ..., xytext=(-10, -22), ...)
```

- The chosen point gets a white ring (`edgecolor="white"`), so it stays visible on top of the
  curve.
- The label is placed below and to the left of the point. A first version placed it above, where
  it crossed the curve: we looked at the rendered figure and moved it.
- The axes go from 0 to 1, with full-sentence labels ("share of failing students flagged").

## `plot_confusion_matrix`

```python
    counts = np.array([[scores["tn"], scores["fp"]], [scores["fn"], scores["tp"]]])
    ax.imshow(counts, cmap="Blues", vmin=0, vmax=counts.max() * 1.6)
```

- It builds the 2 × 2 table from the counts returned by `final_scores`, in the same layout as
  the CLI: rows are the truth, columns are the decision.
- The shade is a single blue scale (light to dark = few to many).
- `vmax = max × 1.6` keeps even the largest cell medium-blue, so the black numbers stay readable
  on every cell.
- Each count is written in its cell, and the tick labels name the rows and columns.

## `plot_coefficients`

```python
    shown = table.head(top).iloc[::-1]  # largest at the top of the chart
    colours = [ORANGE if value > 0 else BLUE for value in shown["coefficient"]]
    ax.barh(shown["feature"], shown["coefficient"], color=colours, height=0.6)
    ax.axvline(0, color=MUTED, linewidth=1)
```

- It takes the `top` largest coefficients from `coefficient_table`, which is already sorted by
  absolute value. It reverses them, because `barh` draws the first row at the bottom.
- **The colour encodes the sign:** orange means the feature raises the risk of failing, blue means
  it lowers it. The title says so, so the colour is never the only cue: the bar's side of the
  zero line says the same.
- Horizontal bars leave room for long feature names such as `activities_yes`.

## `_save`

```python
def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
```

- It creates `reports/figures/` if needed.
- `tight_layout` prevents labels from being cut off.
- It saves at 150 dpi, sharp enough for the PDF report.
- `plt.close(fig)` frees the memory and stops the notebook from also displaying the figure a
  second time. The notebook shows the saved PNG through a Markdown image link instead.

**No tests for this module:** the figures are checked by eye; we rendered and looked at all
three. Testing them would require matplotlib in the main install, which is exactly what the
optional extra avoids.
