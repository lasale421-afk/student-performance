"""Tests for the preprocessing step: no leakage and correct output."""

import numpy as np

from student_risk.data import DatasetLoader
from student_risk.features import build_preprocessor


def test_scaler_is_fitted_on_train_only(config, raw_df):
    loader = DatasetLoader(config)
    X, y = loader.build_target(loader.clean(raw_df))
    X_train, X_test, _, _ = loader.split(X, y)
    numeric_columns = ["age", "studytime", "absences"]

    preprocessor = build_preprocessor().fit(X_train)
    scaler = preprocessor.named_transformers_["numeric"].named_steps["scale"]

    assert np.allclose(scaler.mean_, X_train[numeric_columns].mean())
    assert not np.allclose(scaler.mean_, X[numeric_columns].mean())  # not the full data


def test_preprocessor_output_shape_and_values(config, raw_df):
    X, _ = DatasetLoader(config).build_target(raw_df)
    X.loc[0, "age"] = np.nan  # a missing number must be imputed

    output = build_preprocessor().fit_transform(X)

    # 3 numeric columns + one-hot: school (2) + sex (2) + higher (2) = 9 columns
    assert output.shape == (len(X), 9)
    assert not np.isnan(output).any()
    assert np.allclose(output[:, :3].mean(axis=0), 0)  # numeric columns are centred

    unseen = X.iloc[[0]].copy()
    unseen["school"] = "NEW_SCHOOL"  # a category never seen during fit
    assert build_preprocessor().fit(X).transform(unseen).shape == (1, 9)
