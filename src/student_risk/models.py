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

# The five metrics used everywhere in the project. Precision gets zero_division=0
# so that a model that never predicts "fail" scores 0 instead of raising a warning.
SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "recall": "recall",
    "precision": make_scorer(precision_score, zero_division=0),
    "f1": "f1",
}


def build_pipeline(model: BaseEstimator) -> Pipeline:
    """Chain the preprocessing step and a model into one Pipeline.

    Args:
        model: Any unfitted scikit-learn classifier.

    Returns:
        Pipeline with steps "preprocess" then "model".
    """
    return Pipeline(steps=[("preprocess", build_preprocessor()), ("model", model)])


def make_cv(config: Config) -> StratifiedKFold:
    """Return the cross-validation splitter shared by every model.

    Args:
        config: Project settings (number of folds and seed).

    Returns:
        A shuffled StratifiedKFold with a fixed seed.
    """
    return StratifiedKFold(n_splits=config.cv_splits, shuffle=True, random_state=config.random_seed)


def baseline_scores(X_train: pd.DataFrame, y_train: pd.Series, config: Config) -> pd.DataFrame:
    """Cross-validate the two baselines on the training set only.

    Args:
        X_train: Training features.
        y_train: Training target.
        config: Project settings.

    Returns:
        One row per model, one column per metric (mean over the folds).
    """
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


def main() -> None:
    """Load the data, split it, and print the baseline CV scores."""
    config = load_config()
    X_train, _X_test, y_train, _y_test = DatasetLoader(config).load_and_split()
    print(f"Training set: {len(y_train)} students, fail rate = {y_train.mean():.3f}")
    print(f"{config.cv_splits}-fold stratified CV on the training set:")
    print(baseline_scores(X_train, y_train, config).to_string())


if __name__ == "__main__":
    main()
