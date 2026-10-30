"""Load project settings from config.yaml into a typed Config object."""

from dataclasses import dataclass
from pathlib import Path

import yaml

# src/student_risk/config.py -> parents[2] is the project root folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class Config:
    """All project settings, read once from config.yaml.

    Attributes:
        random_seed: The single seed used everywhere (split, CV, models).
        uci_id: Id of the dataset on the UCI repository.
        cache_path: Absolute path of the local CSV copy of the dataset.
        grade_column: Name of the final grade column (G3).
        pass_mark: A student fails if the final grade is below this value.
        leakage_columns: Columns removed because they leak the answer.
        test_size: Fraction of students kept for the final test.
        cv_splits: Number of StratifiedKFold folds.
        target_recall: Minimum recall wanted when choosing the threshold.
        tradeoff_recalls: Recall targets compared in the threshold trade-off table.
        reports_dir: Absolute path of the folder for result tables.
        model_path: Absolute path of the saved model file (joblib).
        grids: One GridSearchCV parameter grid per model name.
    """

    random_seed: int
    uci_id: int
    cache_path: Path
    grade_column: str
    pass_mark: int
    leakage_columns: list[str]
    test_size: float
    cv_splits: int
    target_recall: float
    tradeoff_recalls: list[float]
    reports_dir: Path
    model_path: Path
    grids: dict[str, dict[str, list]]


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    """Read a YAML config file and return a Config object.

    Args:
        path: Path to the YAML file. Defaults to config.yaml at the project root.

    Returns:
        The settings as a frozen Config dataclass.
    """
    with open(path, encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    root = Path(path).resolve().parent
    return Config(
        random_seed=raw["random_seed"],
        uci_id=raw["data"]["uci_id"],
        cache_path=root / raw["data"]["cache_path"],
        grade_column=raw["data"]["grade_column"],
        pass_mark=raw["data"]["pass_mark"],
        leakage_columns=list(raw["data"]["leakage_columns"]),
        test_size=raw["split"]["test_size"],
        cv_splits=raw["cv"]["n_splits"],
        target_recall=raw["threshold"]["target_recall"],
        tradeoff_recalls=list(raw["threshold"]["tradeoff_recalls"]),
        reports_dir=root / raw["paths"]["reports_dir"],
        model_path=root / raw["paths"]["model_path"],
        grids=raw["grids"],
    )
