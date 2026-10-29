# `tests/test_features.py`: explained line by line

**Role:** check the preprocessing step built by `build_preprocessor()`:

- **no leakage:** it learns its statistics from the training data only;
- **correct output:** the right shape, no missing values, numbers scaled, and unknown categories
  handled.

---

## Imports

```python
"""Tests for the preprocessing step: no leakage and correct output."""

import numpy as np

from student_risk.data import DatasetLoader
from student_risk.features import build_preprocessor
```

- `numpy` gives `np.allclose` (to compare decimals), `np.isnan` and `np.nan`.
- `DatasetLoader` prepares X and the split in the same way as the real code.
- `build_preprocessor` is the function under test.

---

## Test 4: `test_scaler_is_fitted_on_train_only`

**What it checks:** when the preprocessor is fitted on the training set, the scaler's learned
means are those of the **training rows**, not those of the whole dataset. In other words, the
test rows did not influence preprocessing.

```python
    loader = DatasetLoader(config)
    X, y = loader.build_target(loader.clean(raw_df))
    X_train, X_test, _, _ = loader.split(X, y)
    numeric_columns = ["age", "studytime", "absences"]
```

- This prepares the data exactly like the real code does.
- `_` means we ignore `y_train` and `y_test`, because preprocessing does not use the target.
- `numeric_columns` lists the three numeric columns of the fake data, in table order.

```python
    preprocessor = build_preprocessor().fit(X_train)
    scaler = preprocessor.named_transformers_["numeric"].named_steps["scale"]
```

- We fit on the training rows only, as the Pipeline does.
- Then we take the fitted `StandardScaler` out of it. `named_transformers_` (with a trailing `_`)
  only exists after fitting, and gives the fitted branch called `"numeric"`. `named_steps["scale"]`
  gives the step called `"scale"` inside that branch.

```python
    assert np.allclose(scaler.mean_, X_train[numeric_columns].mean())
    assert not np.allclose(scaler.mean_, X[numeric_columns].mean())  # not the full data
```

- `scaler.mean_` holds the means it learned.
- Line 1: they are equal to the training means.
- Line 2: they are **not** equal to the means of all the data, train plus test. This line proves
  that the test passes for the right reason: with this data, the two kinds of mean really differ,
  so the test can tell them apart.
- `np.allclose` compares decimal numbers with a tiny tolerance, because `==` on floats is
  unreliable.

**Make it fail on purpose:** in the test, replace `.fit(X_train)` with `.fit(X)`. This is exactly
the mistake of preprocessing before splitting. Both asserts then fail.

---

## Test 5: `test_preprocessor_output_shape_and_values`

**What it checks:**

- the output has the expected number of columns;
- a missing number is filled in;
- numeric columns are centred;
- a category never seen during training does not crash.

```python
    X, _ = DatasetLoader(config).build_target(raw_df)
    X.loc[0, "age"] = np.nan  # a missing number must be imputed
```

We take the features and put one missing value in on purpose.

```python
    output = build_preprocessor().fit_transform(X)
```

`fit_transform` learns the statistics and transforms in one call. The result is a NumPy array.

```python
    # 3 numeric columns + one-hot: school (2) + sex (2) + higher (2) = 9 columns
    assert output.shape == (len(X), 9)
```

- The number of rows is unchanged.
- The number of columns is 3 scaled numbers plus 2 one-hot columns for each of the 3 text
  columns, which is 9.
- If a column were dropped, or encoded the wrong way, this count would change.

```python
    assert not np.isnan(output).any()
```

There are no `NaN` values left: the imputer filled the missing age. Models such as logistic
regression crash on `NaN`.

```python
    assert np.allclose(output[:, :3].mean(axis=0), 0)  # numeric columns are centred
```

- The numeric columns come first, because the `"numeric"` branch is listed first in the
  `ColumnTransformer`.
- After standard scaling, each one has mean 0 on the data it was fitted on.

```python
    unseen = X.iloc[[0]].copy()
    unseen["school"] = "NEW_SCHOOL"  # a category never seen during fit
    assert build_preprocessor().fit(X).transform(unseen).shape == (1, 9)
```

- This simulates a new student from a school the model never saw, like a typo in a JSON input.
- Thanks to `handle_unknown="ignore"`, the school is encoded as all zeros. The transform still
  works and returns the same 9 columns, so the model can still predict.

**Make it fail on purpose:**

- In `features.py`, remove `handle_unknown="ignore", `. The last line raises a `ValueError`.
- Or delete the line `("impute", SimpleImputer(strategy="median")),`. The scaler keeps the `NaN`,
  and the `np.isnan` assert fails.
- Or delete the line `("scale", StandardScaler()),`. The mean is no longer 0, and the centring
  assert fails. In our trial, test 4 also fails, because the `"scale"` step no longer exists.
