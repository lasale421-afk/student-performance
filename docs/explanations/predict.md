# `src/student_risk/predict.py`: explained line by line

**Role:** the prediction module. It reads **one new student** from a JSON file, checks that the
file is valid, and returns the probability of failing and the decision, using the saved model and
threshold. `cli.py` calls it for the `predict` command. It is a separate module because the
subject asks for "one module per job (…, prediction)".

---

## Imports

```python
"""Predict the risk of failing for one new student described in a JSON file."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from student_risk.config import Config
from student_risk.models import Trainer
```

- `json` is the standard library module that parses JSON files.
- `Path` is used in type hints for file paths.
- `numpy` gives `np.nan`, the "missing value" marker that the imputers recognise.
- `pandas` is used because the pipeline expects a DataFrame with named columns.
- `Config` gives the location of the saved model. `Trainer` gives `Trainer.load`.

---

## `read_case`

```python
def read_case(
    input_path: Path, feature_columns: list[str], numeric_columns: list[str]
) -> pd.DataFrame:
```

- **Inputs:**
  - the JSON path;
  - the 30 fields the model was trained on (`feature_columns`);
  - the 13 that must be numbers (`numeric_columns`).

  Both lists come from the saved model, so they always match the trained model, never a list
  typed by hand.
- **Output:** a one-row DataFrame ready for `predict_proba`.

```python
    with open(input_path, encoding="utf-8") as file:
        try:
            case = json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError(f"{input_path} is not valid JSON: {error}") from error
```

- `json.load` turns the file into Python objects: `{...}` becomes a dict, `"text"` a str, `17`
  an int, `null` None.
- If the file does not exist, `open` raises `FileNotFoundError` with the file name. The CLI prints
  it.
- If the content is not valid JSON (a missing quote, a trailing comma...), `json.load` raises
  `JSONDecodeError`. Its message gives the line and column, but **not the file name**. We
  re-raise it as a `ValueError` that also names the file (checked: before this change, the
  message had no file name).
- `from error` keeps the original error attached, which helps debugging.
- `JSONDecodeError` is itself a kind of `ValueError` (checked: `issubclass` returns True). We
  still convert it explicitly, to control the message.

```python
    if not isinstance(case, dict):
        raise ValueError(f"{input_path} must contain one JSON object {{...}} for one student.")
```

- A file can be valid JSON but contain a list (`[1, 2]`) or a number. We need exactly one object.
- `{{...}}` inside an f-string prints a literal `{...}`, because doubled braces escape the
  f-string braces.

```python
    missing = [column for column in feature_columns if column not in case]
    unknown = sorted(set(case) - set(feature_columns))
```

- `missing` lists the expected fields that are absent, in training order, so the message is easy
  to compare with the example file.
- `unknown` lists the fields given but not expected, using a set difference. For example `G3`, or
  a typo like `"abscences"`. `sorted` makes the message stable and readable.
- **Why unknown fields are an error and not ignored:** a typo would otherwise be silently dropped.
  The real field would then be "missing", and the student described would not be the one the
  user meant. G1, G2 or G3 in the input also signals a misunderstanding (these grades are not
  known early), so the user should be told.

```python
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
```

- We collect **all** problems before raising, so the user can fix everything in one go.
- The message lists the expected fields. Example (checked):
  `Invalid case in bad.json: missing fields: school; unknown fields: G2. Expected exactly these 30 fields: school, sex, age, ...`

```python
    for column in numeric_columns:
        value = case[column]
        if value is not None and (isinstance(value, bool) or not isinstance(value, int | float)):
            raise ValueError(f"Field '{column}' must be a number (or null), got {value!r}.")
```

- Every numeric field must be a JSON number, or `null`.
- **Why `bool` is checked separately:** in Python, `True` is a kind of `int`
  (`isinstance(True, int)` is True). Without this test, `"age": true` would be accepted as age 1.
- `int | float` inside `isinstance` needs Python ≥ 3.10, our minimum.
- `{value!r}` shows the value with quotes, for example `got 'seventeen'`, so the user sees it is
  text.
- **Without this check:** `"age": "17"` would reach the pipeline as text and fail later, deep
  inside scikit-learn, with an unclear message.

```python
    row = {}
    for column in feature_columns:
        value = case[column]
        if value is None:
            value = np.nan
        elif isinstance(value, str):
            value = value.strip()
        row[column] = value
```

- We build the row **in training order**.
- `null` (None) becomes `np.nan`. **Why:** the imputers in the pipeline look for NaN. We checked
  that `SimpleImputer(strategy="most_frequent")` leaves a Python `None` untouched in a text
  column.
- Text is stripped of surrounding spaces, the same cleaning as `DatasetLoader.clean`, so `" yes"`
  is read as `"yes"`.

```python
    df = pd.DataFrame([row], columns=feature_columns)
    df[numeric_columns] = df[numeric_columns].astype(float)
    return df
```

- `pd.DataFrame([row], columns=...)` builds one row with the columns in a fixed order.
- `.astype(float)` makes the numeric columns real numbers even when they contain NaN. A column
  holding only `NaN` would otherwise have the generic `object` type.
- **Checked by hand:** a case with `"Mjob": null` and `"absences": null` gives a normal prediction
  (probability 0.468 instead of 0.496).

---

## `predict_new_case`

```python
def predict_new_case(config: Config, input_path: Path) -> dict:
    trainer = Trainer.load(config.model_path, config)
    case = read_case(input_path, trainer.feature_columns, trainer.numeric_columns)
    probability = float(trainer.best_model.predict_proba(case)[0, 1])
    return {
        "model": trainer.best_name,
        "probability_fail": probability,
        "threshold": trainer.threshold,
        "at_risk": probability >= trainer.threshold,
    }
```

This is the function asked for: **it loads the saved model and threshold, and returns the
probability of failing and the decision for one new case.**

- `Trainer.load` raises a clear `FileNotFoundError` if `train` was never run.
- `predict_proba(case)` returns a 1 × 2 array, `[[P(pass), P(fail)]]`. `[0, 1]` takes row 0,
  column 1: the probability of failing.
- The saved pipeline applies the preprocessing fitted on the training set (medians, means, std,
  categories), then the classifier. Nothing is refitted on the new case.
- `at_risk` uses `>=`, **the same rule** as `choose_threshold` and `threshold_report`. Using `>`
  here would make a student exactly at the threshold be treated differently from training.
- It returns a dict, not printed text, so the function can be tested or reused (in the notebook,
  for example). Printing is the CLI's job.

**Real example (checked):** `examples/new_case.json` gives probability 0.496, which is above the
threshold 0.140, so the decision is "AT RISK".

**Limit:** a known field with an **unseen category**, for example `"Mjob": "astronaut"`, is not
an error. `OneHotEncoder(handle_unknown="ignore")` encodes it as all zeros, and the prediction
continues. This is a deliberate choice of `features.py`, but it means a typo inside a *value*,
rather than a field name, is not reported.
