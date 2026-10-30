# `tests/test_cli.py`: explained line by line

**Role:** run the **real command-line tool** end to end (`train`, then `predict` or `evaluate`)
on the synthetic data, and check its output and its errors. Nothing is downloaded: the synthetic
table is written into the cache file before `train`, so `DatasetLoader.load_raw` reads it
instead of calling UCI.

The file has 5 test cases: 1 + 1 + 3 (one test is parametrized). With the other files, there are
19 tests in total.

---

## Imports and constants

```python
"""Tests for the command-line tool (train, evaluate, predict) on synthetic data, no network."""

import json
import re

import pytest

from student_risk.cli import main

# One new student with exactly the 6 feature fields of the synthetic data.
CASE = {"school": "GP", "sex": "F", "age": 17, "studytime": 2, "absences": 4, "higher": "yes"}
```

- `json` writes the case files, and `re` (regular expressions) reads numbers back out of the
  printed text.
- `main` is the CLI entry point, called with a list of arguments, exactly like the command line.
- `CASE` is a valid student for the **synthetic** data, which has 6 features. It is not the real
  30-field format of `examples/new_case.json`.

## Fixture `trained`

```python
@pytest.fixture
def trained(config_path, config, raw_df):
    """Put the synthetic data in the cache (so nothing is downloaded) and run train."""
    raw_df.to_csv(config.cache_path, index=False)
    assert main(["--config", str(config_path), "train"]) == 0
    return config_path
```

- `raw_df.to_csv(config.cache_path, ...)` writes the fake 40 students where the real download
  would be cached (inside the test's temporary folder). `load_raw` sees that the file exists and
  never imports `ucimlrepo`. **This is what keeps the tests offline.**
- It then runs `train` through the real CLI, with the temporary config. The model is saved in
  `tmp_path/models/model.joblib`, never in the real `models/`.
- `assert ... == 0` stops immediately if training itself fails, so the later asserts are not
  confusing.
- It returns the config path, so each test can run more commands with the same settings.

## Helper `write_case`

```python
def write_case(tmp_path, case):
    """Save a case as a JSON file and return its path as a string."""
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case), encoding="utf-8")
    return str(path)
```

`json.dumps` turns the dict into JSON text. The path is returned as a string because command-line
arguments are strings.

---

## Test 1: `test_cli_train_then_predict_on_json_case`

**What it checks:**

- `train` saved a model file, and created the `models/` folder to do so;
- `predict --input case.json` succeeds (exit code 0);
- it prints a probability between 0 and 1, and a decision line in the documented format.

```python
    assert config.model_path.exists()  # train created models/ and saved the file

    code = main(["--config", str(trained), "predict", "--input", write_case(tmp_path, CASE)])

    output = capsys.readouterr().out
    assert code == 0
    probability = float(re.search(r"Probability of failing: ([0-9.]+)", output).group(1))
    assert 0 <= probability <= 1
    assert re.search(r"Decision \(threshold [0-9.]+\): (AT RISK|not at risk)", output)
```

- `capsys` is a fixture built into pytest. It captures what the code prints. `.out` is
  `stdout` and `.err` is `stderr`.
- The regular expression `Probability of failing: ([0-9.]+)` finds the number after the label.
  `group(1)` is the part in parentheses. If the line is missing, `re.search` returns None and
  `.group` raises an error, so the test fails.
- The last regex checks the decision line. `\(` matches a literal parenthesis, and
  `(AT RISK|not at risk)` accepts either decision. On random synthetic data, we only check the
  *format*, not which decision is made.

**Make it fail on purpose:**

- In `cli.py`, delete the line `print(f"Decision (threshold ...")`. The last assert fails. We
  checked this.
- In `models.py`, delete `path.parent.mkdir(...)` in `save`. `train` then fails to write into the
  missing `models/` folder. The roundtrip test in `test_models.py` catches this too.

---

## Test 2: `test_cli_evaluate_prints_test_metrics`

**What it checks:** `evaluate` loads the saved model, succeeds, and prints the required metrics
and the confusion matrix.

```python
    assert main(["--config", str(trained), "evaluate"]) == 0

    output = capsys.readouterr().out
    for word in ["ROC-AUC", "Average precision", "Recall", "Confusion matrix"]:
        assert word in output
```

- The "test set" here is the 10 synthetic test rows from the temporary config, **not** the real
  test set.
- The loop checks that each required section is printed.

**Make it fail on purpose:** in `cli.py`, delete the line
`print("Confusion matrix (rows = truth, columns = decision):")`. The assert on
`"Confusion matrix"` fails. We checked this.

---

## Test 3 (parametrized ×3): `test_cli_predict_rejects_bad_json`

**What it checks:** three kinds of bad input each give **exit code 1** and a **clear message on
stderr**, not a crash.

```python
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
```

- **The three cases:**
  1. **Missing field:** the dict comprehension copies `CASE` without `age`.
  2. **Unknown field:** `{**CASE, "G3": 12}` copies `CASE` and adds `G3`. This is a realistic
     mistake: G3 is the grade we are trying to predict.
  3. **Wrong type:** `age` given as text.
- Each case runs as a separate test, with both values passed as arguments.
- We check the exit code, and that the expected message appears in **stderr** (`.err`), where
  `main` prints errors.

**Make it fail on purpose (each one checked):**

- In `predict.py`, replace `unknown = sorted(set(case) - set(feature_columns))` with
  `unknown = []`. The `G3` case then fails: G3 is silently ignored, and the prediction succeeds
  with exit code 0.
- In `predict.py`, delete the `for column in numeric_columns:` block (4 lines). The `"seventeen"`
  case fails: pandas raises a different error later ("could not convert string to float"), so the
  expected message is missing.
- In `cli.py`, change `except (FileNotFoundError, ValueError)` into `except FileNotFoundError`.
  All 3 cases fail, because the `ValueError` is no longer caught and becomes a crash.
