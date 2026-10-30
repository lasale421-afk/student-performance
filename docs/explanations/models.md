# `src/student_risk/models.py`: explained line by line

**Role:**

- build the model pipelines;
- compute the two **baselines** (Phase 1);
- compare, tune and choose among the 4 models, and choose the decision threshold, with the
  **`Trainer`** class (Phase 2);
- save and load the trained model (Phase 3).

**Everything in this file only receives the training split.** No function or method takes the
test set. Scoring on the test set happens once, in the `evaluate` command (`cli.py`).

**Phase 3 change:**

- The `main()` function and `python -m student_risk.models` were **removed**. Their work
  (baselines, comparison, threshold, report CSVs) moved to the `train` command in `cli.py`, which
  also saves the model. So there is only one way to train.
- Run `student-risk train`. It takes about 10 seconds.

How the 4 models work in theory is explained in `models_overview.md`.

---

## Imports

```python
"""Model pipelines, the baselines, and the Trainer that compares and tunes the models."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from student_risk.config import Config
from student_risk.evaluate import (
    SCORING,
    choose_threshold,
    threshold_report,
    threshold_tradeoff,
)
from student_risk.features import build_preprocessor
```

- `Path` is the type of the save and load locations. `joblib` writes and reads the saved model
  file (Phase 3).
- `numpy` is used for the type hint of the out-of-fold probabilities, and `pandas` for the
  result tables.
- `BaseEstimator` is only used as a type hint, meaning "any scikit-learn model".
- `CalibratedClassifierCV` gives `LinearSVC` a `predict_proba` method (see below).
- `DummyClassifier`, `LogisticRegression`, `KNeighborsClassifier` and `LinearSVC` are the models.
- From `sklearn.model_selection`:
  - `GridSearchCV` tries every combination of hyperparameters with cross-validation.
  - `StratifiedKFold` is the folds.
  - `cross_val_predict` gives out-of-fold predictions.
  - `cross_validate` gives the CV scores of the baselines.
- Our own modules: the settings, the metrics and threshold functions, and the preprocessing.
- `load_config` and `DatasetLoader` were only used by the removed `main()`, so they are no
  longer imported (Phase 3). ruff would flag them as unused.

## Constants

```python
# The 4 compared models, ordered from the simplest to the most complex.
# This order breaks near-ties when we pick the best model.
MODEL_NAMES = ("logreg_l2", "logreg_l1", "linear_svm", "knn")
REFIT_METRIC = "average_precision"
DEFAULT_THRESHOLD = 0.5
```

`MODEL_NAMES` is our "simplicity order". It is used by `select_best`. The reasons for the order:

1. **`logreg_l2`** is the simplest: one linear formula, probabilities come out directly, every
   coefficient is easy to read, and it is the same family as the baseline.
2. **`logreg_l1`** is still linear and readable, but its optimisation is less standard (a
   different solver, and some coefficients pushed exactly to 0).
3. **`linear_svm`** is a linear model too, but it gives no probabilities on its own. It needs an
   extra calibration model wrapped around it, so it has two layers.
4. **`knn`** has no formula at all. It keeps every training student in memory and compares each
   new student to them. Its prediction is hard to explain with coefficients.

This ordering is our (the assistant's) choice. You can argue for L1 being "simpler" than L2,
because it uses fewer features. Changing the order in this tuple changes the choice rule.

Then:

- `REFIT_METRIC`: the metric GridSearchCV uses to pick the best hyperparameters of each model
  (see `compare`).
- `DEFAULT_THRESHOLD`: scikit-learn's usual threshold, kept only to compare with ours.

## `build_pipeline` (Phase 2 change: step `"clf"`)

```python
def build_pipeline(model: BaseEstimator) -> Pipeline:
    return Pipeline(steps=[("preprocess", build_preprocessor()), ("clf", model)])
```

- This chains preprocessing and the classifier into **one object**. When you call `.fit` on it,
  preprocessing is fitted on the rows it receives, then the classifier is fitted on the
  transformed rows. When you call `.predict` or `.predict_proba`, the *already fitted*
  preprocessing is applied, then the classifier predicts.
- **Why it matters for leakage:** `GridSearchCV`, `cross_validate` and `cross_val_predict` refit
  the *whole* pipeline on each training fold, so the scaler and the encoder never see the
  validation fold.
- **Why the step is named `"clf"`:** the grid parameters in `config.yaml` are written
  `clf__C`. The double underscore means "the parameter `C` of the step called `clf`". If the
  name changed, every grid key would have to change too.

## `make_cv`

```python
def make_cv(config: Config) -> StratifiedKFold:
    return StratifiedKFold(n_splits=config.cv_splits, shuffle=True, random_state=config.random_seed)
```

- `StratifiedKFold`: every fold keeps the fail rate of about 15 %. On the real training set, each
  validation fold has 103–104 students and 16 fails (checked).
- `shuffle=True`: the rows are mixed before cutting the folds. The downloaded CSV is ordered by
  school (checked).
- `random_state=seed`: the same folds on every run.
- **Same CV for everyone:** the Trainer creates this splitter once, and gives the same object to
  all 4 GridSearchCVs and to `cross_val_predict`. Every model is therefore scored on exactly the
  same 5 folds. This is a project rule, and it makes the comparison fair.

## `baseline_scores` (Phase 1, unchanged)

```python
def baseline_scores(X_train: pd.DataFrame, y_train: pd.Series, config: Config) -> pd.DataFrame:
    models = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=config.random_seed),
    }
    cv = make_cv(config)

    rows = {}
    for name, model in models.items():
        results = cross_validate(build_pipeline(model), X_train, y_train, cv=cv, scoring=SCORING)
        rows[name] = {metric: results[f"test_{metric}"].mean() for metric in SCORING}
    return pd.DataFrame(rows).T.round(3)
```

**The two baselines:**

- `DummyClassifier(strategy="most_frequent")` always predicts "pass". It is the score of "doing
  nothing", and its recall is 0.
- An untuned `LogisticRegression` is the simplest real model.
  - `max_iter=1000` is a safety margin. **On our scaled data, the default of 100 gave no
    warning (checked in Phase 1).**
  - `random_state` has no effect with the default solver (checked: identical coefficients with
    two seeds). It is kept to follow the rule "one seed everywhere".

**The scoring loop:**

- For each model, `cross_validate` returns the scores of each fold. The keys are named `test_...`,
  but these are *validation* folds inside the training set.
- `.mean()` averages the 5 folds.

**Metrics (`SCORING`, now defined in `evaluate.py`):**

- `roc_auc`: the probability that a random failing student gets a higher score than a random
  passing one. 0.5 is chance level.
- `average_precision` (AP): the area under the precision-recall curve. It is better than ROC-AUC
  when the positive class is rare. A random model scores about the fail rate (0.154 for the Dummy).
- `recall`, `precision`, `f1`: computed at threshold 0.5.
- `zero_division=0` on precision: the Dummy never predicts "fail", so precision would be 0/0.
  The plain `"precision"` scorer prints an `UndefinedMetricWarning` on each fold (we checked:
  5 warnings).

## `summarise_search`

```python
def summarise_search(search: GridSearchCV) -> dict:
    best = search.best_index_
    results = search.cv_results_
    row = {"best_params": str(search.best_params_)}
    for metric in SCORING:
        row[f"{metric}_mean"] = results[f"mean_test_{metric}"][best]
        row[f"{metric}_std"] = results[f"std_test_{metric}"][best]
    row["train_average_precision_mean"] = results["mean_train_average_precision"][best]
    return row
```

This turns one fitted GridSearchCV into **one row** of the results table.

- `search.cv_results_` is a big dict. It has one entry per hyperparameter combination for each
  statistic, such as `mean_test_recall` or `std_test_recall`.
- `search.best_index_` is the position of the best combination, the one with the highest mean CV
  AP, because `refit="average_precision"`.
- We keep, for that combination:
  - its parameters, as a string so that they fit in a CSV cell;
  - the mean and the std over the folds of every metric;
  - the **mean train AP**.
- **The std** shows how much the score moves from fold to fold. A large std means the mean is
  uncertain.
- **The train score** is the score on the folds the model was *trained* on. If train AP is much
  higher than CV AP, the model memorises the training data (**overfitting**). This is only
  available because of `return_train_score=True` (see `compare`).
- **Why a separate function and not a method:** it does not need `self`, and it is easier to
  read on its own.

## The `Trainer` class

```python
class Trainer:
    """Compare the 4 models, pick the best one and choose its decision threshold. ..."""
```

This is the second main class of the project. The first is `DatasetLoader`.

- **Its role:** turn a training set into a chosen model and a chosen threshold, using only
  cross-validation.
- Since Phase 3, it also saves and loads the model (`save` and `load`).

### `__init__`

```python
    def __init__(self, config: Config) -> None:
        self.config = config
        self.cv = make_cv(config)
        self.searches: dict[str, GridSearchCV] = {}
        self.results: pd.DataFrame | None = None
        self.best_name: str | None = None
        self.best_model: Pipeline | None = None
        self.selection_reason: str = ""
        self.oof_proba: np.ndarray | None = None
        self.threshold: float | None = None
        self.feature_columns: list[str] = []
        self.numeric_columns: list[str] = []
```

- The splitter is created **once**, so all the methods use the same folds.
- Every result attribute starts empty (`None`, `{}` or `""`), and the methods below fill them.
  Declaring them all here, with type hints, shows the reader everything the Trainer will contain.
- `X | None` means "an X, or nothing yet". This syntax needs Python ≥ 3.10, our minimum.
- `feature_columns` and `numeric_columns` (Phase 3) remember the input fields the model was
  trained on, and which of them are numbers. `predict.py` uses them to check a new JSON case.

### `build_models`

```python
    def build_models(self) -> dict[str, Pipeline]:
        seed = self.config.random_seed
        return {
            "logreg_l2": build_pipeline(LogisticRegression(max_iter=1000, random_state=seed)),
            "logreg_l1": build_pipeline(
                LogisticRegression(
                    penalty="l1", solver="liblinear", max_iter=1000, random_state=seed
                )
            ),
            "linear_svm": build_pipeline(
                CalibratedClassifierCV(LinearSVC(max_iter=10000, random_state=seed), cv=3)
            ),
            "knn": build_pipeline(KNeighborsClassifier()),
        }
```

It creates 4 **new, unfitted** pipelines. The keys match the grid names in `config.yaml`.

**`logreg_l2`:** L2 is the default penalty, with the default solver `lbfgs`.

**`logreg_l1`:** `penalty="l1"` needs a solver that supports it.

- `lbfgs`, the default, does **not** support L1. We checked: it raises `ValueError: Solver lbfgs
  supports only 'l2' or None penalties`.
- The two choices are `liblinear` and `saga`. We chose **`liblinear`** because:
  - it is designed for small datasets like ours (519 rows);
  - it handles binary problems directly;
  - it converged here without any warning (checked).
- `saga` is meant for large datasets. It needs well-scaled data and usually many more iterations
  to converge **[from prior knowledge, to verify]**.

**`linear_svm`:**

- `LinearSVC` gives a *score* (the signed distance to the separating line), **not a
  probability**. It has no `predict_proba`, but we need probabilities for AP, ROC-AUC and the
  threshold.
- `CalibratedClassifierCV(..., cv=3)` fixes this. Inside each training fold, it splits the data
  into 3 parts. It trains `LinearSVC` on 2 of them, and fits a small sigmoid (logistic) curve on
  the third. That curve maps scores to probabilities. The default `method="sigmoid"` was
  checked; that it is called "Platt scaling" is **[from prior knowledge, to verify]**.
- **Why `cv=3`:** each CV training fold has about 415 rows and 64 fails, so each calibration
  part keeps about 21 fails (64 / 3). With 5 parts, it would keep only about 13 fails per part, which is too few to fit the
  curve reliably.
- `max_iter=10000`: LinearSVC's default of 1000 iterations often stops before converging when C
  is large **[from prior knowledge, to verify]**. With 10000, our runs show no warning (checked).

**`knn`:** the defaults, because `n_neighbors` and `weights` are set by the grid. k-NN has **no
`class_weight`** parameter and no `random_state`, because it involves no randomness.

### `compare`

```python
    def compare(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        rows = {}
        for name, pipeline in self.build_models().items():
            search = GridSearchCV(
                pipeline,
                param_grid=self.config.grids[name],
                scoring=SCORING,
                refit=REFIT_METRIC,
                cv=self.cv,
                return_train_score=True,
            )
            search.fit(X_train, y_train)
            self.searches[name] = search
            rows[name] = summarise_search(search)
        self.results = pd.DataFrame(rows).T
        return self.results
```

- For each model, `GridSearchCV` tries **every combination** of its grid. For example
  `logreg_l2` has 4 values of C × 2 class weights = 8 combinations, each scored on the 5 folds.
- `param_grid=self.config.grids[name]`: the grids come from `config.yaml`, not from the code.
- `scoring=SCORING`: multi-metric scoring, so every metric is recorded for every combination.
- **`refit="average_precision"`** has two jobs:
  1. It tells GridSearchCV which metric decides the best combination. With multi-metric scoring
     this is mandatory, because GridSearchCV cannot guess which metric matters.
  2. After the search, it refits that best combination **on the whole training set**. The result
     is `search.best_estimator_`.
- **Why AP for choosing:**
  - The positive class is rare (15.4 %). AP focuses on how well the *fails* are ranked.
  - ROC-AUC can look good even when the top of the ranking is full of false alarms, because it
    also rewards ranking the many easy "pass" students correctly.
  - AP **does not depend on a threshold**. We choose the threshold *afterwards*, so the
    hyperparameters should be chosen on a metric that does not assume threshold 0.5. Recall and
    F1 at 0.5 would be the wrong criterion: we will not use 0.5.
- `cv=self.cv`: the same folds for all 4 models, and the same as the baselines.
- `return_train_score=True`: also records the scores on the training folds, to discuss
  overfitting. It costs a little extra computation.
- `self.searches[name] = search` keeps the fitted searches, so that `select_best` can get the
  best estimator.
- `pd.DataFrame(rows).T` gives one row per model.

### `select_best`

```python
    def select_best(self) -> str:
        ap_mean = self.results["average_precision_mean"].astype(float)
        top = ap_mean.idxmax()
        margin = float(self.results.loc[top, "average_precision_std"])
        near_ties = [name for name in MODEL_NAMES if ap_mean[name] >= ap_mean[top] - margin]
```

- `.astype(float)`: the table mixes text (`best_params`) and numbers, so pandas stores the column
  as generic objects. We convert it back to numbers.
- `top`: the model with the highest mean CV AP.
- `margin`: the **standard deviation of the top model's** AP over the folds.
- `near_ties`: every model whose mean AP is at most one std below the top. The list follows
  `MODEL_NAMES`, from simplest to most complex. The top model is always in it.
- **Why this rule:** with only 103–104 students per fold, the scores move a lot between folds. A
  difference smaller than one std is not convincing evidence that one model is better, so we
  prefer the model that is easier to explain and less likely to overfit.

```python
        self.best_name = near_ties[0]
        self.best_model = self.searches[self.best_name].best_estimator_
```

- We take the simplest near-tie.
- `best_estimator_` is that model's best pipeline, **already refitted on the whole training
  set** by GridSearchCV.

```python
        if self.best_name == top:
            self.selection_reason = (...)
        else:
            self.selection_reason = (...)
        return self.best_name
```

This writes a human-readable reason, printed by `main` and kept in the Trainer, so the choice is
explained in the output and not only in our heads.

### `tune_threshold`

```python
    def tune_threshold(self, X_train: pd.DataFrame, y_train: pd.Series) -> float:
        self.oof_proba = cross_val_predict(
            self.best_model, X_train, y_train, cv=self.cv, method="predict_proba"
        )[:, 1]
        self.threshold = choose_threshold(y_train, self.oof_proba, self.config.target_recall)
        return self.threshold
```

**What "out-of-fold" means:** `cross_val_predict` uses the same 5 folds. For each fold, it trains
a **fresh copy** of the best pipeline on the other 4 folds, and predicts the held-out fold. Every
training student therefore gets a probability from a model that **never saw them**.

**Why not use `best_model.predict_proba(X_train)`:**

- `best_model` was trained on those same students, so its probabilities would be too confident.
- A threshold chosen on them would look better than reality.
- And using the test set to choose it would be the penalised mistake ("test set used to make a
  choice").

**Line by line:**

- `cross_val_predict` clones the model, so our fitted `best_model` is not modified.
- `method="predict_proba"` returns two columns, P(pass) and P(fail). `[:, 1]` keeps P(fail).
- `choose_threshold` (in `evaluate.py`) then picks the threshold with recall ≥ 0.80 (from
  `config.yaml`) and the best precision.

### `threshold_comparison`

```python
    def threshold_comparison(self, y_train: pd.Series) -> pd.DataFrame:
        rows = {
            "default_0.5": threshold_report(y_train, self.oof_proba, DEFAULT_THRESHOLD),
            "chosen": threshold_report(y_train, self.oof_proba, self.threshold),
        }
        return pd.DataFrame(rows).T
```

- It applies both thresholds to the **same out-of-fold probabilities**, so the comparison is fair.
- It returns precision, recall, F1, the share of students flagged, and the confusion matrix counts.

### `threshold_tradeoff` (Phase 2b)

```python
    def threshold_tradeoff(self, y_train: pd.Series) -> pd.DataFrame:
        return threshold_tradeoff(y_train, self.oof_proba, self.config.tradeoff_recalls)
```

- This is a thin method. It calls the function of the same name in `evaluate.py` with the
  Trainer's out-of-fold probabilities and the recall targets from `config.yaml` (0.70, 0.80,
  0.90).
- **Why a method and not just the function:** it guarantees that the table uses *the same*
  out-of-fold probabilities as the chosen threshold, and never the test set.
- The method has the same name as the imported function. Inside the method, `threshold_tradeoff`
  refers to the module-level function (the import). The method itself is only reachable as
  `self.threshold_tradeoff`. The meaning of the table is explained in `evaluate.md`.

### `fit`

```python
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "Trainer":
        self.feature_columns = list(X_train.columns)
        self.numeric_columns = list(X_train.select_dtypes(include="number").columns)
        self.compare(X_train, y_train)
        self.select_best()
        self.tune_threshold(X_train, y_train)
        return self
```

- The first two lines record the 30 input fields, in training order, and the 13 numeric ones.
  `select_dtypes(include="number")` uses the same rule as `features.py`, so "numeric" means the
  same thing everywhere.
- Then it runs the three steps in the only correct order.
- `return self` allows `Trainer(config).fit(X_train, y_train)` in one line, the same style as
  scikit-learn.
- `"Trainer"` is in quotes because the class is not fully defined yet while its own body is
  being read.

### `save` (Phase 3)

```python
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        saved = {
            "best_name": self.best_name,
            "best_model": self.best_model,
            "threshold": self.threshold,
            "feature_columns": self.feature_columns,
            "numeric_columns": self.numeric_columns,
        }
        joblib.dump(saved, path)
```

- `path.parent.mkdir(parents=True, exist_ok=True)` **creates `models/` if it does not exist**,
  for example on a fresh clone, where only `models/.gitkeep` exists, or in a test folder.
  `exist_ok=True` means "no error if it already exists".
- **What is saved:** one dict with everything `predict` and `evaluate` need, and nothing more:
  - the fitted pipeline (preprocessing and classifier together, so the scaler means and the
    one-hot categories learned on the training set are saved with it);
  - the threshold;
  - the model's name;
  - the expected input columns.
- **Why the threshold is saved with the model:** the model and its threshold belong together. A
  threshold chosen for `logreg_l2` would be meaningless for another model. Saving them in one file
  makes it impossible to mix them up.
- **Not saved:** the GridSearchCV objects and the out-of-fold probabilities. They are only needed
  during training, and the CSV reports already record them.
- `joblib.dump` serialises Python objects (here, scikit-learn objects containing NumPy arrays) to
  a file. This is the method recommended by scikit-learn and required by the subject.
- **The file is git-ignored** (`models/*.joblib`). `train` recreates it.

### `load` (Phase 3)

```python
    @classmethod
    def load(cls, path: Path, config: Config) -> "Trainer":
        if not path.exists():
            raise FileNotFoundError(f"No saved model at {path}. Run 'student-risk train' first.")
        saved = joblib.load(path)
        trainer = cls(config)
        trainer.best_name = saved["best_name"]
        trainer.best_model = saved["best_model"]
        trainer.threshold = saved["threshold"]
        trainer.feature_columns = saved["feature_columns"]
        trainer.numeric_columns = saved["numeric_columns"]
        return trainer
```

- **`@classmethod`:** the method receives the class itself (`cls`) instead of an object. We call
  it on the class, `Trainer.load(path, config)`, because before loading there is no Trainer yet.
  `cls(config)` creates a new, empty Trainer. This is the usual Python way to write an
  "alternative constructor".
- **The `FileNotFoundError` check:** if someone runs `evaluate` or `predict` before `train`, they
  get a message that says what to do, instead of a raw joblib error. The CLI catches this error
  and prints it (see `cli.md`).
- Then the five saved values are put back into the new Trainer. It is now "ready to predict":
  `best_model.predict_proba` works and `threshold` is set. The comparison attributes stay empty,
  because they are not needed to predict.
- **Security limit:** `joblib.load` can run code hidden in a malicious file. Only load model files
  you created yourself **[from prior knowledge, to verify]**.

## Results on the real training set (5-fold CV, 519 students, 80 fails)

These results come from the **widened grids** of Phase 2b: C from 0.001 to 10, and k from 5 to 51.

| model      | ROC-AUC (std) | **AP** (std)  | train AP | recall@0.5 | precision@0.5 | F1@0.5 | best parameters |
|------------|---------------|---------------|----------|------------|---------------|--------|-----------------|
| logreg_l2  | 0.803 (0.066) | 0.500 (0.143) | 0.560    | 0.050      | 0.650         | 0.091  | C = 0.01, class_weight = None |
| logreg_l1  | 0.828 (0.050) | **0.538** (0.158) | 0.564 | 0.238     | 0.647         | 0.331  | C = 0.1, class_weight = None |
| linear_svm | 0.803 (0.069) | 0.500 (0.147) | 0.552    | 0.225      | 0.632         | 0.310  | C = 0.001, class_weight = balanced |
| knn        | 0.768 (0.094) | 0.510 (0.179) | **1.000** | 0.050     | 0.567         | 0.090  | n_neighbors = 31, weights = distance |
| *Dummy*    | 0.500         | 0.154         |          | 0.000      | 0.000         | 0.000  | |
| *LogReg baseline* | 0.779  | 0.457         |          | 0.300      | 0.503         | 0.370  | C = 1 |

The full best parameters are in `reports/model_comparison.csv`.

**What the table says:**

- All 4 tuned models beat the Dummy by far, and they beat the untuned baseline on AP
  (0.50–0.54 against 0.457).
- **The choice:** `logreg_l1` has the highest mean AP (0.538). But the std is large (0.158), and
  all 4 models are within one std of it. The rule therefore picks the simplest:
  **`logreg_l2`** (AP 0.500).
- **Fold-by-fold check (paired view):**
  - Every model has its worst AP on fold 3 (0.25–0.33) and its best on folds 2 and 4
    (0.63–0.73). The *folds* change the score more than the *model* does.
  - `logreg_l1` beats `logreg_l2` on 4 of the 5 folds.

  The one-std rule ignores this pairing. It is a point to discuss: a paired comparison could
  favour L1.
- **Overfitting:**
  - k-NN has a train AP of 1.000 against 0.510 in CV. With `weights="distance"`, a training
    student is its own nearest neighbour at distance 0, so it predicts its own label perfectly.
    This is memorisation, not learning **[from prior knowledge, to verify]**.
  - The three linear models have train AP only 0.03 to 0.06 above CV AP, so they barely overfit
    (strong regularisation).
- **Grid edges (Phase 2b):** the first grids stopped at C = 0.01 and k = 31, and three models
  chose those edge values. After widening:
  - `logreg_l2` still chose C = 0.01. It is now *inside* the grid, so it is a real optimum.
  - `knn` still chose k = 31. Now 41 and 51 were also tried, so 31 is inside the grid.
  - `linear_svm` moved to C = 0.001, with `balanced` weights, which is again the **lowest** value
    of the grid. Its AP went from 0.492 to 0.500. It could perhaps improve a little with an even
    smaller C, but it would need a large gain to change the decision, because all 4 models are
    already near-ties. **We did not widen further.**
  - **The chosen model (`logreg_l2`, C = 0.01) and the threshold (0.14) did not change.**
- **Recall at 0.5 is tiny** (0.05 for `logreg_l2`). With C = 0.01, the probabilities are pulled
  strongly towards the base rate: the out-of-fold probabilities have median 0.12 and maximum
  0.63, and only 1.3 % of students are above 0.5 (checked). This is exactly why we choose the
  threshold ourselves.
- **`class_weight`:** `None` won for both logistic regressions, and `balanced` won for the SVM.
  - AP only measures the *ranking*. For a logistic regression, class weighting mostly shifts the
    probabilities up for everyone, which changes the ranking very little **[from prior
    knowledge, to verify]**.
  - So for the chosen model, "balanced" brings no AP gain, and the threshold step handles the
    imbalance instead.

## Limit: the CV score of the chosen model is slightly optimistic

- The hyperparameters, and then the model, were chosen **on the same 5 folds** that produced the
  scores. We picked the maximum of several noisy scores, and the maximum of noisy numbers tends
  to be above its true value. This is **selection bias**.
- So the 0.500 AP of the chosen model is probably a bit optimistic.
- The honest estimate is the **test score**, computed once (Phase 3, `student-risk evaluate`) on
  data never used for any choice:
  - **test AP = 0.346**, against 0.500 in CV;
  - test ROC-AUC = 0.778, against 0.803 in CV.

  The drop agrees with this optimism. But the test set is small (130 students, only 20 fails),
  so its score is itself noisy: one or two students more or less changes AP noticeably. **No
  modelling choice was changed after seeing it.**
- A nested cross-validation would remove this bias, at about 5 times the cost. We did not use it,
  for simplicity.
- This is one of the model's limits to state in the report.
