# `src/student_risk/data.py`: explained line by line

**Role:** everything about the *data*: download it (or read the local copy), clean it, build the
target `fail`, remove the leaking columns, and split train/test **once**. Its single class,
`DatasetLoader`, is one of the two main classes of the project. The other is `Trainer`, which
comes in a later phase.

---

## Imports and constant

```python
"""Load the UCI Student Performance data, clean it, build the target and split it."""
```

The module docstring, required by ruff.

```python
import pandas as pd
from sklearn.model_selection import train_test_split

from student_risk.config import Config
```

- `pandas`: the data is a table, so a DataFrame is the natural type.
- `train_test_split`: scikit-learn's function for a (stratified) random split.
- `Config`: imported only to use in type hints, so the reader knows what `config` contains.

```python
TARGET_NAME = "fail"
```

The name given to the target Series. It is a constant, so the name is written in one place only.

## The class

```python
class DatasetLoader:
    """Download, clean, build the target and split the Student Performance data. ..."""

    def __init__(self, config: Config) -> None:
        """Store the settings used by every step."""
        self.config = config
```

- **Why a class:** the four steps all need the same settings (cache path, pass mark, seed...).
  We store the `Config` once in `self.config` instead of passing it to every function.
- `-> None`: `__init__` returns nothing. This is the type hint the course rules ask for.
- **Each method does one step**, so each step can be tested alone with a small fake DataFrame.

## `load_raw`: download or read the cache

```python
    def load_raw(self) -> pd.DataFrame:
        if self.config.cache_path.exists():
            return pd.read_csv(self.config.cache_path)
```

- If the CSV copy already exists, we read it. There is no network call, it is fast, and the data
  is always the same between runs.

```python
        # Imported here so that tests and cached runs never need the network library.
        from ucimlrepo import fetch_ucirepo
```

- **Why a local import:** `ucimlrepo` is only needed the first time. Importing it inside the
  function means that reading the cache, and all the unit tests, never touch it. Ruff accepts
  this, and the comment explains it.

```python
        dataset = fetch_ucirepo(id=self.config.uci_id)
        raw = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
```

- `fetch_ucirepo(id=320)` downloads the dataset. It returns an object where `data.features` holds
  the 30 input columns and `data.targets` holds G1, G2 and G3.
- `pd.concat(..., axis=1)` puts them side by side (`axis=1` means *columns*, not rows), giving
  one table of 649 rows × 33 columns. We want one table because cleaning (for example removing a
  row) must act on the features and the grades together, so they stay aligned.

```python
        self.config.cache_path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(self.config.cache_path, index=False)
        return raw
```

- `mkdir(parents=True, exist_ok=True)` creates the `data/` folder if needed, and does not crash if
  it already exists.
- `to_csv(..., index=False)` saves the table without the pandas row numbers. Otherwise, reading
  it back would add a useless `Unnamed: 0` column.
- **On a fresh machine:** `data/` is git-ignored, so the first run downloads and creates the
  cache, and later runs read it.

## `clean`

```python
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
```

- We work on a copy, so the caller's DataFrame is never modified by surprise. Pandas functions
  that modify in place are a classic source of bugs.

```python
        text_columns = df.select_dtypes(include="object").columns
        df[text_columns] = df[text_columns].apply(lambda column: column.str.strip())
```

- We find the text columns (dtype `object`) and remove spaces at the start and end of every
  value.
- **Why:** `"yes"` and `"yes "` would otherwise become two different one-hot columns. Typing
  errors like this are common in data entered by hand, such as a new case given in JSON.

```python
        df = df.dropna(subset=[self.config.grade_column])
```

- We remove students with no final grade. Without G3 we cannot build the target, and we must
  never guess a label.
- We do **not** remove rows with missing *features*: the imputer inside the pipeline handles them.

```python
        df = df.drop_duplicates()
        return df.reset_index(drop=True)
```

- `drop_duplicates()` removes exact copies of a row. A duplicate could end up in both train and
  test, which is a hidden form of leakage, and it would count one student twice.
- `reset_index(drop=True)` renumbers the rows 0..n-1 after removals, so the indexes are clean.
- **Facts about the real data:** it has 0 missing values and 0 duplicates (we checked: 649 rows
  before and after cleaning). The cleaning is a *safety net*, for example for future data or a
  corrupted cache.
- **Is cleaning before the split a leak?** No. Every step decides about a row by looking only at
  that row. Nothing is *learned* from the whole dataset, such as a mean. Anything that learns
  (imputation, scaling, encoding) is inside the Pipeline (see `features.md`).

## `build_target`

```python
        y = (df[self.config.grade_column] < self.config.pass_mark).astype(int)
        y.name = TARGET_NAME
```

- `df["G3"] < 10` gives a column of True/False values. `.astype(int)` turns it into 1/0. So
  `fail = 1` means G3 < 10, and **10 itself counts as a pass**.
- **Why binary:** the school needs a yes/no flag ("help this student"), not a predicted grade.
- **Imbalance:** about 15 % of students fail, so `fail = 1` is the minority, positive class. This
  is why we use stratification and metrics such as recall, precision and average precision, not
  just accuracy. A model that always says "pass" would already be about 85 % accurate.

```python
        columns_to_drop = [self.config.grade_column, *self.config.leakage_columns]
        X = df.drop(columns=columns_to_drop)
        return X, y
```

- `[G3, *[G1, G2]]` gives `["G3", "G1", "G2"]` (the `*` unpacks the list).
- We remove G3, because it *is* the answer, and G1/G2, because they leak the answer and are not
  known early. See `config_yaml.md` for the full reasoning.
- We return a **tuple** `(X, y)`, the scikit-learn convention.

## `split`

```python
        return train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            stratify=y,
            random_state=self.config.random_seed,
        )
```

- `test_size=0.2`: 20 % of the students go to the test set.
- `stratify=y`: train and test get the same fail rate (about 15 %). Without it, by bad luck the
  test set could get, say, 8 % fails, and the test scores would not represent reality.
- `random_state=42`: the same split every time, which makes results reproducible.
- It returns `X_train, X_test, y_train, y_test`, in this order. The order is a common source of
  mistakes, so it is written in the docstring.
- **This is called once.** The test set is then left untouched until the final evaluation.

## `load_and_split`

```python
        df = self.clean(self.load_raw())
        X, y = self.build_target(df)
        return self.split(X, y)
```

This is a shortcut that runs the whole data chain in the correct order. Callers (the baseline,
and later the `Trainer` and the CLI) use this one method and cannot forget a step.
