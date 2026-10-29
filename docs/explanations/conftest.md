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

from student_risk.config import Config
```

- `Path`: used for the type of `tmp_path`.
- `numpy`: to generate random fake values.
- `pandas`: to build the fake DataFrame.
- `pytest`: for the `@pytest.fixture` decorator.
- `Config`: we build a config by hand, with no YAML file.

## Fixture `config`

```python
@pytest.fixture
def config(tmp_path: Path) -> Config:
    """Config like the real one, but with the cache in a temporary folder."""
    return Config(
        random_seed=42,
        uci_id=320,
        cache_path=tmp_path / "cache.csv",
        grade_column="G3",
        pass_mark=10,
        leakage_columns=["G1", "G2"],
        test_size=0.25,
        cv_splits=3,
        target_recall=0.80,
    )
```

- `@pytest.fixture` registers the function as a fixture named `config`.
- `tmp_path` is a fixture built into pytest: a new, empty temporary folder for each test.
  - **Why:** if a test ever wrote the cache, it would write it there, never into the real
    `data/` folder.
- The values are the same as `config.yaml`, except for two:
  - `test_size=0.25`: with only 40 fake rows, this gives exactly 10 test rows.
  - `cv_splits=3`: fewer folds, because the fake data is tiny.
- **Without this fixture:** every test would read the real `config.yaml` and depend on its
  values, so changing the project settings could break tests for no reason.

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
