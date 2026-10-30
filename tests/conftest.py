"""Shared test fixtures: a small synthetic dataset and a test config (no network)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from student_risk.config import Config, load_config


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


@pytest.fixture
def config(config_path: Path) -> Config:
    """The test settings, read with the real loader. Every path is inside tmp_path."""
    return load_config(config_path)


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """40 fake students with the same kinds of columns as the real data."""
    rng = np.random.default_rng(0)
    n = 40
    g3 = np.array([5, 8, 9, 12, 14, 15, 10, 11] * 5)  # 3 fails out of 8 -> 15 fails
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
