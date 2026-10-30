# student-risk: early warning for students at risk of failing

Authors: **Mohamed Reda Zarrouk** and **Georgy Salameh**. This is a PGE 2 final project for the
Machine Learning and Advanced Python courses.

## What it does

A school wants to know, **early in the year**, which students are likely to fail, so that tutors
can help them in time. This tool reads a student's profile (school, family, study habits,
absences, past failures and so on: 30 fields) and returns:

- the **probability of failing**, where failing means a final grade G3 < 10 out of 20;
- a **decision**: flag the student for tutoring, or not.

**Design choices** (all explained in `docs/explanations/`):

- **No leakage.** The period grades G1 and G2 are removed, because they are almost the final
  grade and are not known early.
- **Recall first.** Missing a student who will fail costs more than an unneeded tutoring
  session. The decision threshold is therefore chosen so that **at least 80 % of failing students
  are flagged** (recall ≥ 0.80), with the best precision possible. It is chosen on
  cross-validated (out-of-fold) predictions of the training set, never on the test set.
- **Honest protocol.**
  - One stratified train/test split, with seed 42.
  - All preprocessing happens inside a scikit-learn `Pipeline`.
  - Two baselines: a Dummy classifier and a logistic regression.
  - 4 models compared under the same 5-fold stratified CV, with GridSearchCV: logistic
    regression L2, logistic regression L1, k-NN, and a calibrated linear SVM.
  - The test set is used **once**, by `evaluate`.

## Installation

You need Python **3.10, 3.11 or 3.12**, and an internet connection for the first `train` (the
dataset is downloaded from UCI once, then cached in `data/`).

```bash
git clone <repository-url> student-risk
cd student-risk
python -m venv .venv
```

Activate the virtual environment:

- Linux / macOS:

  ```bash
  source .venv/bin/activate
  ```

- Windows (Command Prompt or PowerShell):

  ```bat
  .venv\Scripts\activate
  ```

Then install the package. This also installs the pinned versions of every library, plus `pytest`
and `ruff`:

```bash
pip install -e .
```

On some Linux systems the command is `python3` instead of `python`.

## Usage

All commands are run from the project folder, with the virtual environment activated. Settings
(seed, split, folds, grids, recall target, paths) are in `config.yaml`.

### Train

```bash
student-risk train
```

This does the following:

- downloads the data the first time and caches it in `data/`;
- splits it once;
- runs the baselines and tunes the 4 models with cross-validation **on the training set only**;
- picks the best model and chooses the threshold;
- saves the model and threshold to `models/model.joblib`;
- writes the result tables in `reports/` (`baseline.csv`, `model_comparison.csv`,
  `threshold_report.csv`, `threshold_tradeoff.csv`).

It takes about 10 seconds.

### Evaluate

```bash
student-risk evaluate
```

This loads the saved model, scores it **once** on the test set, and prints ROC-AUC, average
precision, precision, recall, F1, the share of students flagged, and the confusion matrix.

### Predict one new student

```bash
student-risk predict --input examples/new_case.json
```

This prints the probability of failing and the decision. For example:

```text
Probability of failing: 0.496
Decision (threshold 0.140): AT RISK - flag for tutoring
```

**The input file:**

- It must be **one JSON object with exactly the 30 feature fields**. See
  `examples/new_case.json`, an invented student.
- A missing field, an unknown field (for example G3), or a text value in a numeric field
  stops the command with a clear error message and exit code 1.
- `null` is accepted for an unknown value. It is filled in the same way as during training.

**Running without the `student-risk` command:** if the command is not found, use the equivalent
`python -m student_risk.cli train`, `python -m student_risk.cli evaluate` and
`python -m student_risk.cli predict --input examples/new_case.json`.

## Analysis notebook

`notebooks/analysis.ipynb` tells the full modelling story. It covers problem framing, data
exploration (including the G3 = 0 special code and leakage), the pipeline, baselines, the model
comparison, the selection rule, the threshold, the single test evaluation with a 95 % Wilson
interval, the coefficients, a G3 = 0 sensitivity check, a fairness check by sex and school, the
limits, and the risks.

It **calls the package** and reads the tables in `reports/`, and it checks that it reproduces
them exactly. It is saved with its outputs, so you can read it directly. To run it again:

```bash
pip install -e ".[notebook]"
jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb
```

The figures are written to `reports/figures/` (precision-recall curve, test confusion matrix,
coefficients). The notebook tools are an optional extra, so the main install stays light.

## Tests and code quality

```bash
pytest
ruff check .
```

- There are 19 tests. They use small synthetic data and never download anything.
- What each test checks, and how to make it fail on purpose, is explained in
  `docs/explanations/test_*.md`.

## Project layout

```text
student-risk/
├── README.md
├── AI_USAGE.md              # declaration of generative-AI use
├── pyproject.toml           # package metadata, pinned dependencies, pytest and ruff settings
├── config.yaml              # seed, split, CV folds, recall target, paths, hyperparameter grids
├── examples/new_case.json   # an invented student for the predict command
├── notebooks/analysis.ipynb # the modelling story, calling the package
├── src/student_risk/
│   ├── __init__.py
│   ├── config.py            # reads config.yaml into a typed Config object
│   ├── data.py              # DatasetLoader: download/cache, clean, build target, split
│   ├── features.py          # ColumnTransformer: impute, scale, one-hot encode
│   ├── models.py            # baselines and Trainer: compare, tune, threshold, save/load
│   ├── evaluate.py          # metrics, threshold, Wilson interval, per-group report, coefficients
│   ├── predict.py           # read and validate one JSON case, predict it
│   ├── plots.py             # figures (needs the optional "notebook" extra)
│   └── cli.py               # train, evaluate, predict
├── tests/                   # pytest tests (synthetic data, no network)
├── docs/explanations/       # line-by-line explanation of every file
├── data/                    # cached dataset (git-ignored, re-downloaded if missing)
├── models/                  # saved model (model.joblib is git-ignored, created by train)
└── reports/                 # result tables written by train; figures/ written by the notebook
```

## Credits and licence of the data

- **Dataset:** P. Cortez, *Student Performance* [Dataset], UCI Machine Learning Repository,
  2008. https://doi.org/10.24432/C5TG7T (page:
  https://archive.ics.uci.edu/dataset/320/student+performance). Licence:
  **Creative Commons Attribution 4.0 International (CC BY 4.0)**. We use it unchanged, except
  that we build a pass/fail target and drop G1, G2 and G3 from the features, as described above.
- **Original study:** P. Cortez and A. M. G. Silva, "Using Data Mining to Predict Secondary School
  Student Performance", *Proceedings of the 5th Annual Future Business Technology Conference
  (FUBUTEC 2008)*, 2008.
- **Libraries:** scikit-learn, pandas, NumPy, SciPy, joblib, PyYAML, ucimlrepo, pytest and ruff,
  plus Jupyter, nbconvert, ipykernel and matplotlib for the notebook. Their versions are pinned in
  `pyproject.toml`.
- **Methods:** the Wilson score interval for proportions (E. B. Wilson, 1927) is used for the
  confidence intervals on recall.
- **Generative AI:** an AI coding assistant was used; see `AI_USAGE.md` for what it did.
