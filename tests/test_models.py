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


def test_predict_proba_shape_range_and_sum(config, raw_df):
    X_train, X_test, y_train, _ = split_fake_data(config, raw_df)
    trainer = Trainer(config).fit(X_train, y_train)

    proba = trainer.best_model.predict_proba(X_test)

    assert proba.shape == (len(X_test), 2)  # one column per class
    assert ((proba >= 0) & (proba <= 1)).all()
    assert np.allclose(proba.sum(axis=1), 1)


def test_trainer_sees_only_train_rows_and_is_reproducible(config, raw_df):
    X_train, _, y_train, _ = split_fake_data(config, raw_df)

    first = Trainer(config).fit(X_train, y_train)
    second = Trainer(config).fit(X_train, y_train)

    # The final model was fitted on exactly the training rows, nothing more.
    scaler = first.best_model.named_steps["preprocess"].named_transformers_["numeric"]
    assert scaler.named_steps["scale"].n_samples_seen_ == len(X_train)
    assert len(first.oof_proba) == len(X_train)  # one out-of-fold probability per train row

    # Same seed, same data -> identical results, model choice and threshold.
    pd.testing.assert_frame_equal(first.results, second.results)
    assert first.best_name == second.best_name
    assert np.array_equal(first.oof_proba, second.oof_proba)
    assert first.threshold == second.threshold


def test_save_load_roundtrip_gives_same_predictions(config, raw_df, tmp_path):
    X_train, X_test, y_train, _ = split_fake_data(config, raw_df)
    trainer = Trainer(config).fit(X_train, y_train)
    path = tmp_path / "new_folder" / "model.joblib"  # the folder does not exist yet

    trainer.save(path)
    loaded = Trainer.load(path, config)

    assert np.array_equal(
        loaded.best_model.predict_proba(X_test), trainer.best_model.predict_proba(X_test)
    )
    assert loaded.threshold == trainer.threshold
    assert loaded.best_name == trainer.best_name
    assert loaded.feature_columns == list(X_train.columns)
