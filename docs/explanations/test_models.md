# `tests/test_models.py`: explained line by line

**Role:** check the `Trainer` on the small synthetic dataset from `conftest.py`:

- the probabilities are valid;
- it only learns from the rows it receives (train only);
- it is reproducible;
- save then load gives back the same model (Phase 3).

It uses the tiny grids from the test `config` (2 values per model), so the whole file runs in
about 2 seconds.

---

## Imports and helper

```python
"""Tests for the Trainer: probabilities, train-only fitting and reproducibility."""

import numpy as np
import pandas as pd

from student_risk.data import DatasetLoader
from student_risk.models import Trainer


def split_fake_data(config, raw_df):
    """Run the real data chain on the synthetic table (30 train rows, 10 test rows)."""
    loader = DatasetLoader(config)
    X, y = loader.build_target(loader.clean(raw_df))
    return loader.split(X, y)
```

- `pandas` gives `pd.testing.assert_frame_equal`, which compares two tables cell by cell and
  shows the differences if the test fails.
- `split_fake_data` is a small helper used by both tests, so the same 4 lines are not written
  twice. It runs the *real* `DatasetLoader` steps on fake data, so the Trainer receives exactly
  the kind of input it gets in production.
- Its name does not start with `test_`, so pytest does not run it as a test.
- **Note:** the "test" rows here are part of the fake 40-student table. The real test set is never
  used in any test.

---

## Test 1: `test_predict_proba_shape_range_and_sum`

**What it checks:** the chosen model returns real probabilities:

- one row per student;
- 2 columns (P(pass), P(fail));
- every value between 0 and 1;
- each row summing to 1.

The CLI's `predict` and the threshold both depend on this.

```python
    X_train, X_test, y_train, _ = split_fake_data(config, raw_df)
    trainer = Trainer(config).fit(X_train, y_train)

    proba = trainer.best_model.predict_proba(X_test)
```

- We train on the 30 fake training rows.
- Then we predict the 10 rows the Trainer never saw, as new students would be.

```python
    assert proba.shape == (len(X_test), 2)  # one column per class
    assert ((proba >= 0) & (proba <= 1)).all()
    assert np.allclose(proba.sum(axis=1), 1)
```

- Line 1 checks the shape.
- Line 2 checks the range. `&` combines the two conditions element by element, and `.all()`
  requires every value to pass.
- Line 3 checks that P(pass) + P(fail) = 1 for every student. `np.allclose` tolerates tiny
  rounding errors.

**Make it fail on purpose:**

- In the test, replace `predict_proba` with `decision_function`. This method returns raw scores:
  one column, values outside [0, 1]. The shape assert fails.
- This shows what the test protects us from: using scores as if they were probabilities. A bare
  `LinearSVC`, without `CalibratedClassifierCV`, only has `decision_function`.

---

## Test 2: `test_trainer_sees_only_train_rows_and_is_reproducible`

**What it checks:**

1. **Train only:** the final model was fitted on exactly the training rows, and the out-of-fold
   probabilities cover exactly the training rows. Together with the fact that no Trainer method
   takes test data, this shows the test set cannot leak into training.
2. **Reproducible:** two Trainers with the same seed and the same data give identical results,
   the same chosen model and the same threshold.

```python
    X_train, _, y_train, _ = split_fake_data(config, raw_df)

    first = Trainer(config).fit(X_train, y_train)
    second = Trainer(config).fit(X_train, y_train)
```

We train two completely independent Trainers on the same input.

```python
    scaler = first.best_model.named_steps["preprocess"].named_transformers_["numeric"]
    assert scaler.named_steps["scale"].n_samples_seen_ == len(X_train)
```

- We walk down into the fitted pipeline: the `preprocess` step, then its `numeric` branch, then
  the `scale` step (the `StandardScaler`).
- `n_samples_seen_` is the number of rows the scaler was fitted on. It must be exactly the 30
  training rows. If any extra row, for example a test row, had been used, this number would be
  larger.

```python
    assert len(first.oof_proba) == len(X_train)  # one out-of-fold probability per train row
```

The threshold is chosen on these probabilities, and there must be exactly one per training
student, no more.

```python
    pd.testing.assert_frame_equal(first.results, second.results)
    assert first.best_name == second.best_name
    assert np.array_equal(first.oof_proba, second.oof_proba)
    assert first.threshold == second.threshold
```

- Every output must be **exactly** equal, not just close: the results table, the model name, every
  out-of-fold probability, and the threshold.
- This works because every source of randomness uses the same seed: the CV shuffling,
  `liblinear`, and `LinearSVC`.

**Make it fail on purpose:**

- **Break "train only":** in the test, fit `first` on `pd.concat([X_train, X_test])` and
  `pd.concat([y_train, y_test])`. Unpack `X_test, y_test` from `split_fake_data`. The scaler has
  then seen 40 rows instead of 30, and the first assert fails.
- **Break reproducibility:** in `models.py`, inside `make_cv`, change
  `random_state=config.random_seed` into `random_state=None`. Each Trainer then shuffles its
  folds differently, and `assert_frame_equal` fails.

---

## Test 3: `test_save_load_roundtrip_gives_same_predictions` (Phase 3)

**What it checks:** saving with joblib and loading back gives a Trainer that predicts
**exactly** the same probabilities, with the same threshold, model name and input columns. It
also checks that `save` creates a missing folder.

```python
    X_train, X_test, y_train, _ = split_fake_data(config, raw_df)
    trainer = Trainer(config).fit(X_train, y_train)
    path = tmp_path / "new_folder" / "model.joblib"  # the folder does not exist yet

    trainer.save(path)
    loaded = Trainer.load(path, config)
```

- We train on the fake training rows.
- We save into a folder that does not exist yet, like `models/` on a fresh clone if the
  `.gitkeep` were missing.
- Then we load it into a **new** Trainer with the class method `Trainer.load`.

```python
    assert np.array_equal(
        loaded.best_model.predict_proba(X_test), trainer.best_model.predict_proba(X_test)
    )
    assert loaded.threshold == trainer.threshold
    assert loaded.best_name == trainer.best_name
    assert loaded.feature_columns == list(X_train.columns)
```

- `np.array_equal` checks **exact** equality of the probabilities on rows the model never saw.
  A saved model must be the same model, not an approximation.
- The threshold, the name, and the list of input columns (needed by `predict` to check a JSON
  case) must also survive the round trip.

**Make it fail on purpose (both checked):**

- In `models.py`, delete `path.parent.mkdir(parents=True, exist_ok=True)` in `save`. Saving into
  `new_folder/` then raises `FileNotFoundError`.
- In `Trainer.load`, replace `trainer.threshold = saved["threshold"]` with
  `trainer.threshold = 0.5`. The threshold assert fails. This is the bug of a model loaded
  without its tuned threshold.
