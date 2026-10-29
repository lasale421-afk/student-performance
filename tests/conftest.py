"""Shared test fixtures: a small synthetic dataset and a test config (no network)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from student_risk.config import Config


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
