# `tests/conftest.py`: explained line by line

**Role:** shared *fixtures* for all tests. A fixture is a function that prepares an object. Any
test that names it as an argument receives that object automatically. pytest finds
`conftest.py` on its own, so no import is needed.

**Key design choice:** the tests use a small **synthetic** (fake) table and a config whose cache
points to a temporary folder. They never download anything, so they run in under a second, work
offline, and do not depend on UCI being online.

---

## Imports

```python
"""Shared test fixtures: a small synthetic dataset and a test config (no network)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from student_risk.config import Config, load_config
```

- `Path` is used for the type of `tmp_path` and of the config file.
- `numpy` generates random fake values, and `pandas` builds the fake DataFrame.
- `pytest` provides the `@pytest.fixture` decorator.
- `yaml` writes a temporary `config.yaml` (Phase 3).
- `Config` is used for the return type, and `load_config` is **the real loader**, used on the
  temporary file.

## Fixture `config_path` (Phase 3)

```python
@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    """Write a config.yaml like the real one into a temporary folder and return its path."""
    settings = {
        "random_seed": 42,
        "data": {
            "uci_id": 320,
            "cache_path": "cache.csv",
            "grade_column": "G3",
            "pass_mark": 10,
            "leakage_columns": ["G1", "G2"],
        },
        "split": {"test_size": 0.25},
        "cv": {"n_splits": 3},
        "threshold": {"target_recall": 0.80, "tradeoff_recalls": [0.70, 0.80, 0.90]},
        "paths": {"reports_dir": "reports", "model_path": "models/model.joblib"},
        "grids": {  # tiny grids so that the tests stay fast
            "logreg_l2": {"clf__C": [0.1, 1.0]},
            "logreg_l1": {"clf__C": [0.1, 1.0]},
            "linear_svm": {"clf__estimator__C": [0.1, 1.0]},
            "knn": {"clf__n_neighbors": [3, 5]},
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    return path
```

- `@pytest.fixture` registers the function as a fixture. `tmp_path` is a fixture built into
  pytest: a new, empty temporary folder for each test.
- `settings` has **the same structure as the real `config.yaml`**. It is written to
  `tmp_path/config.yaml` with `yaml.safe_dump`.
- **Every path is relative**, so `load_config` resolves it inside `tmp_path`:
  - the cache is `tmp_path/cache.csv`;
  - the model is `tmp_path/models/model.joblib`;
  - the reports go to `tmp_path/reports`.

  **No test can write into the real `data/`, `models/` or `reports/`.**
- **The values that differ from the real config, and why:**
  - `test_size: 0.25`: with 40 fake rows, this gives exactly 10 test rows.
  - `n_splits: 3`: fewer folds, because the fake data is tiny.
  - **Tiny grids**, with 2 values per model: the tests check the *mechanics* (shapes, train-only
    fitting, reproducibility, CLI), not the quality of the model. This keeps the 19 tests at
    about 9 seconds.
  - **`n_neighbors` 3 and 5, not the real 5 to 51:** each CV training fold has only 20 fake rows,
    and k = 31 or 51 would be more neighbours than rows. We checked that scikit-learn then raises
    `ValueError: Expected n_neighbors <= n_samples_fit`. Inside GridSearchCV, such fits would
    only give NaN scores and warnings, which hides the problem **[from prior knowledge, to
    verify]**.
- **Why a real file (Phase 3 change):** before Phase 3 this fixture built a `Config(...)` object
  directly. The CLI tests need a *file* to pass to `--config`, exactly as a user would. Writing
  the file and reading it back with `load_config` serves both kinds of test. As a bonus, every
  test now also exercises the real `load_config`.

## Fixture `config`

```python
@pytest.fixture
def config(config_path: Path) -> Config:
    """The test settings, read with the real loader. Every path is inside tmp_path."""
    return load_config(config_path)
```

- A fixture can use another fixture: pytest first runs `config_path`, then passes its result
  here.
- Tests that need the settings object ask for `config`. Tests that need the file (the CLI) ask
  for `config_path`. In the same test, both refer to the **same** temporary folder.
- **Without these fixtures:** every test would read the real `config.yaml` and the real `data/`.
  It would depend on the network, and could overwrite the real model.

## Fixture `raw_df`

```python
@pytest.fixture
def raw_df() -> pd.DataFrame:
    """40 fake students with the same kinds of columns as the real data."""
    rng = np.random.default_rng(0)
    n = 40
```

- `default_rng(0)`: a random generator with a fixed seed, so the "random" fake data is the same
  every run. Tests must be deterministic: a test that sometimes fails is worse than no test.
- `n = 40`: big enough to stratify and split, and small enough to read.

```python
    g3 = np.array([5, 8, 9, 12, 14, 15, 10, 11] * 5)  # 3 fails out of 8 -> 15 fails
```

- These are the final grades: the pattern of 8 values is repeated 5 times.
- Three values (5, 8, 9) are below 10, so there are 15 fails out of 40, which gives both classes
  for stratification.
- The values 10 and 11 are included on purpose: 10 is the exact boundary, and it must count as a
  **pass**.

```python
    return pd.DataFrame(
        {
            "school": rng.choice(["GP", "MS"], size=n),
            "sex": rng.choice(["F", "M"], size=n),
            "age": rng.integers(15, 22, size=n),
            "studytime": rng.integers(1, 5, size=n),
            "absences": rng.integers(0, 30, size=n),
            "higher": rng.choice(["yes", "no"], size=n),
            "G1": g3 + 1,
            "G2": g3 - 1,
            "G3": g3,
        }
    )
```

- The table imitates the real one:
  - 3 text columns: `school`, `sex`, `higher`.
  - 3 numeric columns: `age`, `studytime`, `absences`.
  - The three grade columns.
- It uses real column names, so the code under test sees the same structure as in production.
- `G1`/`G2` are close to `G3` on purpose, like in the real data. They exist so that we can check
  they are removed.
