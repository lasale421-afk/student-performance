"""Tests for DatasetLoader: cleaning, target creation and the train/test split."""

import numpy as np
import pandas as pd

from student_risk.data import DatasetLoader


def test_clean_strips_text_drops_missing_grade_and_duplicates(config, raw_df):
    dirty = raw_df.copy()
    dirty.loc[0, "school"] = "  GP "  # extra spaces
    dirty.loc[1, "G3"] = np.nan  # no final grade
    dirty = pd.concat([dirty, dirty.iloc[[2]]])  # one duplicated row

    cleaned = DatasetLoader(config).clean(dirty)

    assert cleaned.loc[0, "school"] == "GP"
    assert cleaned["G3"].notna().all()
    assert not cleaned.duplicated().any()
    assert len(cleaned) == len(raw_df) - 1  # lost the missing-grade row only
    assert list(cleaned.index) == list(range(len(cleaned)))


def test_build_target_and_drop_grade_columns(config):
    df = pd.DataFrame(
        {"age": [16, 17, 18, 19], "G1": [1, 2, 3, 4], "G2": [1, 2, 3, 4], "G3": [0, 9, 10, 20]}
    )

    X, y = DatasetLoader(config).build_target(df)

    assert list(y) == [1, 1, 0, 0]  # 9 fails, 10 passes (boundary)
    assert y.name == "fail"
    assert list(X.columns) == ["age"]  # G1, G2 and G3 are gone


def test_split_is_disjoint_stratified_and_reproducible(config, raw_df):
    loader = DatasetLoader(config)
    X, y = loader.build_target(loader.clean(raw_df))

    X_train, X_test, y_train, y_test = loader.split(X, y)

    assert set(X_train.index).isdisjoint(X_test.index)  # no test row in train
    assert len(X_train) + len(X_test) == len(X)
    assert abs(y_train.mean() - y_test.mean()) < 0.1  # same fail rate in both
    X_train_again = loader.split(X, y)[0]
    assert list(X_train.index) == list(X_train_again.index)  # same seed, same split
