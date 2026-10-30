# `src/student_risk/cli.py`: explained line by line

**Role:** the command-line tool required by the subject, with three commands:

| command | what it does | uses the test set? |
|---|---|---|
| `student-risk train` | baselines, model comparison, threshold, report CSVs, saves the model | no |
| `student-risk evaluate` | scores the saved model on the test set and prints the metrics | **yes, once** |
| `student-risk predict --input file.json` | probability of failing and decision for one student | no |

The `student-risk` command exists because `pyproject.toml` declares
`[project.scripts] student-risk = "student_risk.cli:main"` (see `pyproject_toml.md`).
`python -m student_risk.cli ...` does the same thing.

---

## Imports

```python
"""Command-line tool with three commands: train, evaluate and predict."""

import argparse
import sys
from pathlib import Path

from student_risk.config import DEFAULT_CONFIG_PATH, Config, load_config
from student_risk.data import DatasetLoader
from student_risk.evaluate import final_scores
from student_risk.models import Trainer, baseline_scores
from student_risk.predict import predict_new_case
```

- `argparse` is the standard library module for command-line arguments, as the subject requires
  ("argparse or typer").
- `sys` gives `sys.stderr`, where errors are printed, and `sys.exit`, which sets the exit code.
- `Path` is the type of the `--config` and `--input` arguments.
- `DEFAULT_CONFIG_PATH` is the project's `config.yaml`, used when `--config` is not given.
- The rest are the pieces the commands put together. The CLI contains **no ML logic of its own**.

---

## `run_train`

```python
def run_train(config: Config) -> None:
    X_train, _X_test, y_train, _y_test = DatasetLoader(config).load_and_split()
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training set: {len(y_train)} students, fail rate = {y_train.mean():.3f}")
```

- It loads (or downloads), cleans and splits the data, and **keeps only the training part**. The
  `_` prefix means "received but deliberately unused".
- It creates `reports/` if needed.

```python
    baseline = baseline_scores(X_train, y_train, config)
    baseline.to_csv(config.reports_dir / "baseline.csv")
    print(...)
```

This runs the two baselines (Dummy and untuned logistic regression) with 5-fold CV, saves them
and prints them.

```python
    trainer = Trainer(config).fit(X_train, y_train)
    trainer.results.to_csv(config.reports_dir / "model_comparison.csv")
    scores = trainer.results.drop(columns="best_params").astype(float).round(3)
    print(...)
    print(f"\nChosen model: {trainer.best_name}. {trainer.selection_reason}")
```

- `fit` compares the 4 models, selects one, and chooses the threshold on out-of-fold predictions
  (see `models.md`).
- For printing, the text column `best_params` is dropped so that the numbers can be rounded. The
  CSV keeps everything.
- The reason for the choice is printed, so the choice is never hidden.

```python
    comparison = trainer.threshold_comparison(y_train)
    comparison.to_csv(config.reports_dir / "threshold_report.csv")
    ...
    tradeoff = trainer.threshold_tradeoff(y_train)
    tradeoff.to_csv(config.reports_dir / "threshold_tradeoff.csv", index=False)
    ...
```

These are the threshold tables (default 0.5 against chosen, and recall targets 0.70/0.80/0.90),
all on **out-of-fold** predictions of the training set.

```python
    trainer.save(config.model_path)
    print(f"\nSaved the model and its threshold to {config.model_path}")
```

This saves the model and its threshold together with joblib. `save` creates `models/` if it is
missing.

**Phase 3 note:** this code was `main()` in `models.py` in Phase 2. It moved here, with the
`save` added, so there is one training command and no duplicate.

---

## `run_evaluate`

```python
def run_evaluate(config: Config) -> None:
    trainer = Trainer.load(config.model_path, config)
    _X_train, X_test, _y_train, y_test = DatasetLoader(config).load_and_split()
    proba = trainer.best_model.predict_proba(X_test)[:, 1]
    scores = final_scores(y_test, proba, trainer.threshold)
```

- It loads the **saved** model and threshold. Nothing is retrained.
- `load_and_split` recreates exactly the same split as in training, thanks to the fixed seed and
  the same cached data. It keeps only the test part.
- `[:, 1]` keeps P(fail) for each test student.
- `final_scores` (in `evaluate.py`) gives ROC-AUC, AP, and the threshold metrics with the
  confusion matrix.
- **This is the only place in the project that touches `X_test` and `y_test`.** It makes no
  choice: the model and threshold are already fixed. Nothing is saved or tuned from these
  numbers. That is the "test once" rule.

```python
    print(f"Test set: {len(y_test)} students, model {trainer.best_name}")
    print(f"ROC-AUC            {scores['roc_auc']:.3f}")
    ...
    print("Confusion matrix (rows = truth, columns = decision):")
    print("                  pred pass  pred fail")
    print(f"  true pass       {scores['tn']:9d}  {scores['fp']:9d}")
    print(f"  true fail       {scores['fn']:9d}  {scores['tp']:9d}")
```

- These lines print the metrics with 3 decimals, and the share flagged as a percentage
  (`:.1%`).
- The confusion matrix is printed as a 2 × 2 table. `:9d` pads each integer to 9 characters so
  the columns line up.
- **Reading it:**
  - `fn` (true fail, predicted pass) are the missed students, the costly error.
  - `fp` (true pass, predicted fail) are the unneeded tutoring sessions.

---

## `run_predict`

```python
def run_predict(config: Config, input_path: Path) -> None:
    result = predict_new_case(config, input_path)
    decision = "AT RISK - flag for tutoring" if result["at_risk"] else "not at risk"
    print(f"Probability of failing: {result['probability_fail']:.3f}")
    print(f"Decision (threshold {result['threshold']:.3f}): {decision}")
```

- `predict_new_case` (in `predict.py`) does the work: it loads, validates and predicts.
- The CLI only turns the result into two readable lines.
- The threshold is printed next to the decision, so the user sees *why* a probability of, say,
  0.2 already means "at risk". Our threshold is 0.14, not 0.5.

---

## `build_parser`

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="student-risk",
        description="Early-warning tool: flag students at risk of failing (final grade < 10).",
    )
```

- `prog` is the name shown in `--help` and in error messages.
- `description` is the first line of `student-risk --help`.

```python
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="path to the YAML settings file (default: config.yaml of the project)",
    )
```

- This is an **optional** global option, placed *before* the command:
  `student-risk --config other.yaml train`.
- `type=Path` converts the text into a `Path`.
- **Why it exists:** the tests run the real CLI with a temporary config (synthetic data,
  temporary model folder), so they never touch the real `data/` or `models/` and never need the
  network. Users normally never pass it.

```python
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("train", help="compare models on the training set and save the best one")
    commands.add_parser("evaluate", help="score the saved model once on the test set")
    predict = commands.add_parser("predict", help="predict one new student from a JSON file")
    predict.add_argument("--input", type=Path, required=True, help="JSON file with one student")
    return parser
```

- `add_subparsers` creates the three **sub-commands**. The chosen one is stored in
  `args.command`.
- `required=True`: running `student-risk` with no command prints a usage error, instead of doing
  nothing.
- Only `predict` has an extra argument, `--input`. It is required, so `student-risk predict`
  without it gives an argparse error with the usage line.
- argparse automatically provides `--help` for each level: `student-risk --help` and
  `student-risk predict --help`.

---

## `main`

```python
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
```

- `argv=None` makes argparse read the real command line (`sys.argv`). The tests instead pass a
  list such as `["--config", path, "predict", "--input", file]`, which lets them run the CLI
  **in-process**, without starting a new program.
- **It returns an int, the exit code,** instead of exiting itself, so the tests can check it.

```python
    try:
        if args.command == "train":
            run_train(config)
        elif args.command == "evaluate":
            run_evaluate(config)
        else:
            run_predict(config, args.input)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0
```

- It dispatches to the chosen command. `else` can only be `predict`, because argparse rejects
  any other command.
- **Expected user errors** produce one clear line on `stderr` and exit code 1, instead of a long
  Python traceback:
  - `FileNotFoundError`: no saved model yet, or no input file;
  - `ValueError`: an invalid JSON case.
- **Other errors**, which are real bugs, are not caught, so their full traceback stays visible.
- `stderr` is the standard place for error messages. A script that captures `stdout` still sees
  the errors separately.
- **Exit codes:** 0 means success and anything else means failure. Shells and CI tools rely on
  this convention.

```python
if __name__ == "__main__":
    sys.exit(main())
```

- When the module is run as `python -m student_risk.cli ...`, `main()` runs and its return value
  becomes the program's exit code.
- The installed `student-risk` command does the same thing through the small launcher that pip
  generates from `[project.scripts]` **[from prior knowledge, to verify: that the launcher
  passes the return value to `sys.exit`]**. Our check: `student-risk predict` with a bad file
  returned exit code 1 in the terminal.
