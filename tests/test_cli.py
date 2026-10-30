"""Tests for the command-line tool (train, evaluate, predict) on synthetic data, no network."""

import json
import re

import pytest

from student_risk.cli import main

# One new student with exactly the 6 feature fields of the synthetic data.
CASE = {"school": "GP", "sex": "F", "age": 17, "studytime": 2, "absences": 4, "higher": "yes"}


@pytest.fixture
def trained(config_path, config, raw_df):
    """Put the synthetic data in the cache (so nothing is downloaded) and run train."""
    raw_df.to_csv(config.cache_path, index=False)
    assert main(["--config", str(config_path), "train"]) == 0
    return config_path


def write_case(tmp_path, case):
    """Save a case as a JSON file and return its path as a string."""
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case), encoding="utf-8")
    return str(path)


def test_cli_train_then_predict_on_json_case(trained, config, tmp_path, capsys):
    assert config.model_path.exists()  # train created models/ and saved the file

    code = main(["--config", str(trained), "predict", "--input", write_case(tmp_path, CASE)])

    output = capsys.readouterr().out
    assert code == 0
    probability = float(re.search(r"Probability of failing: ([0-9.]+)", output).group(1))
    assert 0 <= probability <= 1
    assert re.search(r"Decision \(threshold [0-9.]+\): (AT RISK|not at risk)", output)


def test_cli_evaluate_prints_test_metrics(trained, capsys):
    assert main(["--config", str(trained), "evaluate"]) == 0

    output = capsys.readouterr().out
    for word in ["ROC-AUC", "Average precision", "Recall", "Confusion matrix"]:
        assert word in output


@pytest.mark.parametrize(
    ("bad_case", "expected_message"),
    [
        ({k: v for k, v in CASE.items() if k != "age"}, "missing fields: age"),
        ({**CASE, "G3": 12}, "unknown fields: G3"),
        ({**CASE, "age": "seventeen"}, "Field 'age' must be a number"),
    ],
)
def test_cli_predict_rejects_bad_json(trained, tmp_path, capsys, bad_case, expected_message):
    code = main(["--config", str(trained), "predict", "--input", write_case(tmp_path, bad_case)])

    assert code == 1  # error exit code, no crash
    assert expected_message in capsys.readouterr().err
