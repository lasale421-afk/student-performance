"""Predict the risk of failing for one new student described in a JSON file."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from student_risk.config import Config
from student_risk.models import Trainer


def read_case(
    input_path: Path, feature_columns: list[str], numeric_columns: list[str]
) -> pd.DataFrame:
    """Read one student from a JSON file and check it has exactly the expected fields.

    Args:
        input_path: JSON file with one object, e.g. {"school": "GP", "age": 17, ...}.
        feature_columns: The fields the model was trained on, in training order.
        numeric_columns: The fields that must be numbers.

    Returns:
        A one-row DataFrame with the columns in training order. null values become
        NaN so that the pipeline's imputers fill them.

    Raises:
        ValueError: If the file is not valid JSON, is not one object, a field is
            missing or unknown, or a numeric field is not a number.
    """
    with open(input_path, encoding="utf-8") as file:
        try:
            case = json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError(f"{input_path} is not valid JSON: {error}") from error
    if not isinstance(case, dict):
        raise ValueError(f"{input_path} must contain one JSON object {{...}} for one student.")

    missing = [column for column in feature_columns if column not in case]
    unknown = sorted(set(case) - set(feature_columns))
    if missing or unknown:
        problems = []
        if missing:
            problems.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            problems.append(f"unknown fields: {', '.join(unknown)}")
        raise ValueError(
            f"Invalid case in {input_path}: {'; '.join(problems)}. "
            f"Expected exactly these {len(feature_columns)} fields: {', '.join(feature_columns)}."
        )

    for column in numeric_columns:
        value = case[column]
        if value is not None and (isinstance(value, bool) or not isinstance(value, int | float)):
            raise ValueError(f"Field '{column}' must be a number (or null), got {value!r}.")

    row = {}
    for column in feature_columns:
        value = case[column]
        if value is None:
            value = np.nan
        elif isinstance(value, str):
            value = value.strip()
        row[column] = value
    df = pd.DataFrame([row], columns=feature_columns)
    df[numeric_columns] = df[numeric_columns].astype(float)
    return df


def predict_new_case(config: Config, input_path: Path) -> dict:
    """Load the saved model and threshold, and predict one new student.

    Args:
        config: Project settings (gives the path of the saved model).
        input_path: JSON file describing the student.

    Returns:
        Dict with the model name, the probability of failing, the threshold and
        the decision at_risk (True if probability >= threshold).
    """
    trainer = Trainer.load(config.model_path, config)
    case = read_case(input_path, trainer.feature_columns, trainer.numeric_columns)
    probability = float(trainer.best_model.predict_proba(case)[0, 1])
    return {
        "model": trainer.best_name,
        "probability_fail": probability,
        "threshold": trainer.threshold,
        "at_risk": probability >= trainer.threshold,
    }
