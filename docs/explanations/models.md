# `src/student_risk/models.py`: explained line by line (Phase 1 version)

**Role in Phase 1:** define the metrics, the shared Pipeline builder and the shared CV splitter,
and compute the **baseline** scores: `DummyClassifier`, then a plain `LogisticRegression`. The
`Trainer` class (tuning, threshold, save/load) is added here in a later phase.

Run it with:

```bash
python -m student_risk.models
```

---

## Imports

```python
"""Model pipelines and the baseline comparison (Dummy vs LogisticRegression)."""

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from student_risk.config import Config, load_config
from student_risk.data import DatasetLoader
from student_risk.features import build_preprocessor
```

- `BaseEstimator`: only used as a type hint, meaning "any scikit-learn model".
- `DummyClassifier`, `LogisticRegression`: the two baselines.
- `make_scorer`, `precision_score`: build a precision scorer with a custom option (see below).
- `StratifiedKFold`, `cross_validate`: cross-validation.
- The last three lines import our own modules: settings, data, and preprocessing.

## The metrics

```python
SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "recall": "recall",
    "precision": make_scorer(precision_score, zero_division=0),
    "f1": "f1",
}
```

This dict is used by `cross_validate` to compute all five metrics in one go.

- `roc_auc`: the probability that a random failing student gets a higher risk score than a random
  passing student. It does not depend on the threshold. 0.5 means random guessing, 1 is perfect.
- `average_precision`: the area under the precision-recall curve. **It is better than ROC-AUC for
  an imbalanced target**, because it focuses on the positive class. A random model scores about
  the fail rate (≈ 0.15), not 0.5.
- `recall`, `precision`, `f1`: computed with the default threshold of 0.5. The threshold is tuned
  in a later phase.
- **Why `make_scorer(precision_score, zero_division=0)`:** the Dummy model never predicts
  "fail", so precision = 0/0. The plain string `"precision"` would print a warning on every fold.
  With `zero_division=0` we decide on purpose that "no positive prediction" means precision 0.

## `build_pipeline`

```python
def build_pipeline(model: BaseEstimator) -> Pipeline:
    return Pipeline(steps=[("preprocess", build_preprocessor()), ("model", model)])
```

- It chains preprocessing and the model into **one object**. When you call `.fit` on it,
  preprocessing is fitted, then the model is fitted on the transformed data. When you call
  `.predict`, the *already fitted* preprocessing is applied, then the model predicts.
- **Why it matters for leakage:** `cross_validate` refits the *whole* pipeline on each training
  fold, so the scaler never sees the validation fold. If we scaled before cross-validation, each
  validation fold would have been used to compute the mean, which is a leak.
- The step names `"preprocess"` and `"model"` are used later to tune hyperparameters (for example
  `model__C`) and to read the coefficients.

## `make_cv`

```python
def make_cv(config: Config) -> StratifiedKFold:
    return StratifiedKFold(n_splits=config.cv_splits, shuffle=True, random_state=config.random_seed)
```

- `StratifiedKFold`: every fold keeps the fail rate of about 15 %. With a plain `KFold`, a fold
  could get very few fails, and recall on that fold would be meaningless.
- `shuffle=True`: rows are mixed before cutting the folds. The CSV is ordered by school, so
  without shuffling, a fold could contain only one school.
- `random_state=seed`: the same folds every run.
- **Why a function used by everyone:** the project rule is that all models are compared under
  the *same* CV. A single function that builds the splitter guarantees identical folds.

## `baseline_scores`

```python
def baseline_scores(X_train: pd.DataFrame, y_train: pd.Series, config: Config) -> pd.DataFrame:
```

The function takes **only the training set**. The test set is never passed here.

```python
    models = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=config.random_seed),
    }
```

- `DummyClassifier(strategy="most_frequent")` always predicts the majority class ("pass") and
  ignores the features.
  - **Why:** it is the score of "doing nothing". Any real model must beat it. Its recall of 0
    shows clearly why accuracy would be misleading here.
- `LogisticRegression` with the default settings (L2, C = 1) is the simplest real model.
  - `max_iter=1000`: the default of 100 iterations can stop before the solver converges, which
    gives a `ConvergenceWarning`.
  - `random_state`: our single seed. It has no effect with the default solver, but it keeps the
    rule "one seed everywhere".
  - **No `class_weight` yet, on purpose:** this is the *untuned* baseline. Handling the imbalance
    (`class_weight="balanced"`) is part of the tuning phase.

```python
    cv = make_cv(config)

    rows = {}
    for name, model in models.items():
        results = cross_validate(build_pipeline(model), X_train, y_train, cv=cv, scoring=SCORING)
        rows[name] = {metric: results[f"test_{metric}"].mean() for metric in SCORING}
    return pd.DataFrame(rows).T.round(3)
```

- For each model, we wrap it in a full pipeline and run 5-fold CV.
- `cross_validate` returns a dict with keys such as `test_recall`, each holding one score per
  fold. `.mean()` averages the 5 folds. The name says `test_` but these are the *validation*
  folds inside the training set: the real test set is not used.
- The dict comprehension builds `{metric: mean score}` for each model.
- `pd.DataFrame(rows).T` gives one row per model and one column per metric. `.round(3)` makes it
  readable.

## `main`

```python
def main() -> None:
    config = load_config()
    X_train, _X_test, y_train, _y_test = DatasetLoader(config).load_and_split()
    print(f"Training set: {len(y_train)} students, fail rate = {y_train.mean():.3f}")
    print(f"{config.cv_splits}-fold stratified CV on the training set:")
    print(baseline_scores(X_train, y_train, config).to_string())


if __name__ == "__main__":
    main()
```

- `main` loads the config, gets the split, and prints the baseline table.
- The `_` prefix in `_X_test` and `_y_test` tells the reader, and the linter, "we receive it but
  deliberately do not use it". The test set stays untouched.
- `y_train.mean()` on a 0/1 column is the fail rate.
- `if __name__ == "__main__":` runs `main()` only when the file is executed
  (`python -m student_risk.models`), not when another module imports it.
- In a later phase, this becomes a CLI command.

## Results obtained (Phase 1, train set only, 5-fold CV)

| model                | ROC-AUC | avg. precision | recall | precision | F1   |
|----------------------|---------|----------------|--------|-----------|------|
| Dummy (most frequent)| 0.500   | 0.154          | 0.000  | 0.000     | 0.00 |
| LogisticRegression   | 0.779   | 0.457          | 0.300  | 0.503     | 0.37 |

How to read it:

- Logistic regression clearly beats the Dummy on ranking: ROC-AUC 0.78 against 0.50, and average
  precision 0.46 against 0.15. So the features do carry signal.
- But with the default threshold of 0.5, it only catches **30 % of failing students**. This is far
  from our goal of recall ≥ 0.80. It motivates the next phase: class weights, tuning, and choosing
  the threshold on out-of-fold predictions.
