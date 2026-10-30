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

# The 4 compared models, ordered from the simplest to the most complex.
# This order breaks near-ties when we pick the best model.
MODEL_NAMES = ("logreg_l2", "logreg_l1", "linear_svm", "knn")
REFIT_METRIC = "average_precision"
DEFAULT_THRESHOLD = 0.5


def build_pipeline(model: BaseEstimator) -> Pipeline:
    """Chain the preprocessing step and a classifier into one Pipeline.

    Args:
        model: Any unfitted scikit-learn classifier.

    Returns:
        Pipeline with steps "preprocess" then "clf".
    """
    return Pipeline(steps=[("preprocess", build_preprocessor()), ("clf", model)])


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


def summarise_search(search: GridSearchCV) -> dict:
    """Extract the scores of the best hyperparameters from a fitted GridSearchCV.

    Args:
        search: A fitted GridSearchCV with multi-metric scoring and train scores.

    Returns:
        Dict with the best parameters, the mean and std of each CV metric, and
        the mean train average precision (to compare with the CV one).
    """
    best = search.best_index_
    results = search.cv_results_
    row = {"best_params": str(search.best_params_)}
    for metric in SCORING:
        row[f"{metric}_mean"] = results[f"mean_test_{metric}"][best]
        row[f"{metric}_std"] = results[f"std_test_{metric}"][best]
    row["train_average_precision_mean"] = results["mean_train_average_precision"][best]
    return row


class Trainer:
    """Compare the 4 models, pick the best one and choose its decision threshold.

    The Trainer only ever receives the training split: every score it produces
    comes from cross-validation inside the training set.

    Args:
        config: Project settings (seed, folds, grids, target recall).
    """

    def __init__(self, config: Config) -> None:
        """Store the settings and create the shared CV splitter."""
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

    def build_models(self) -> dict[str, Pipeline]:
        """Create the 4 unfitted pipelines, one per model name.

        Returns:
            Dict from model name to Pipeline (preprocessing + classifier "clf").
        """
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

    def compare(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        """Tune every model with GridSearchCV on the same folds and collect the scores.

        Args:
            X_train: Training features.
            y_train: Training target.

        Returns:
            Results table: one row per model, best parameters and CV scores.
        """
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

    def select_best(self) -> str:
        """Pick the best model by mean CV average precision, preferring simpler models.

        Every model whose mean AP is within one standard deviation of the top
        model's mean AP counts as a near-tie; among them we take the simplest,
        following the order of MODEL_NAMES.

        Returns:
            The name of the chosen model.
        """
        ap_mean = self.results["average_precision_mean"].astype(float)
        top = ap_mean.idxmax()
        margin = float(self.results.loc[top, "average_precision_std"])
        near_ties = [name for name in MODEL_NAMES if ap_mean[name] >= ap_mean[top] - margin]

        self.best_name = near_ties[0]
        self.best_model = self.searches[self.best_name].best_estimator_
        if self.best_name == top:
            self.selection_reason = (
                f"{top} has the highest mean CV average precision ({ap_mean[top]:.3f}) "
                f"and no simpler model is within one std ({margin:.3f})."
            )
        else:
            self.selection_reason = (
                f"{top} has the highest mean CV average precision ({ap_mean[top]:.3f}), "
                f"but the simpler {self.best_name} ({ap_mean[self.best_name]:.3f}) is within "
                f"one std ({margin:.3f}), so we prefer {self.best_name}."
            )
        return self.best_name

    def tune_threshold(self, X_train: pd.DataFrame, y_train: pd.Series) -> float:
        """Choose the decision threshold on out-of-fold probabilities of the best model.

        Args:
            X_train: Training features.
            y_train: Training target.

        Returns:
            The threshold reaching the target recall with the best precision.
        """
        self.oof_proba = cross_val_predict(
            self.best_model, X_train, y_train, cv=self.cv, method="predict_proba"
        )[:, 1]
        self.threshold = choose_threshold(y_train, self.oof_proba, self.config.target_recall)
        return self.threshold

    def threshold_comparison(self, y_train: pd.Series) -> pd.DataFrame:
        """Compare the default threshold 0.5 with the chosen one, on out-of-fold predictions.

        Args:
            y_train: Training target (the same rows used in tune_threshold).

        Returns:
            One row per threshold: precision, recall, F1, share flagged, confusion counts.
        """
        rows = {
            "default_0.5": threshold_report(y_train, self.oof_proba, DEFAULT_THRESHOLD),
            "chosen": threshold_report(y_train, self.oof_proba, self.threshold),
        }
        return pd.DataFrame(rows).T

    def threshold_tradeoff(self, y_train: pd.Series) -> pd.DataFrame:
        """Compare several recall targets on the out-of-fold predictions of the best model.

        Args:
            y_train: Training target (the same rows used in tune_threshold).

        Returns:
            One row per recall target from config: threshold, precision, recall, share flagged.
        """
        return threshold_tradeoff(y_train, self.oof_proba, self.config.tradeoff_recalls)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "Trainer":
        """Run the whole training procedure: compare, select, choose the threshold.

        Args:
            X_train: Training features.
            y_train: Training target.

        Returns:
            The Trainer itself, now holding the results, best model and threshold.
        """
        self.feature_columns = list(X_train.columns)
        self.numeric_columns = list(X_train.select_dtypes(include="number").columns)
        self.compare(X_train, y_train)
        self.select_best()
        self.tune_threshold(X_train, y_train)
        return self

    def save(self, path: Path) -> None:
        """Save the chosen model, its threshold and the expected input columns with joblib.

        Args:
            path: Where to write the file. Missing parent folders are created.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        saved = {
            "best_name": self.best_name,
            "best_model": self.best_model,
            "threshold": self.threshold,
            "feature_columns": self.feature_columns,
            "numeric_columns": self.numeric_columns,
        }
        joblib.dump(saved, path)

    @classmethod
    def load(cls, path: Path, config: Config) -> "Trainer":
        """Load a Trainer saved with save(), ready to predict.

        Args:
            path: The file written by save().
            config: Project settings.

        Returns:
            A Trainer holding the saved model, threshold and input columns.

        Raises:
            FileNotFoundError: If no model was saved at this path.
        """
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
