# `notebooks/analysis.ipynb`: explained section by section

**Role:** the modelling story for the Machine Learning grade. The subject says: "the notebook
only tells the story and calls the package, and the package must reproduce the notebook's
results". So:

- **Every computation comes from `student_risk`:**
  - `DatasetLoader`, `Trainer`, `make_cv`, `SCORING`;
  - `final_scores`, `wilson_interval`, `group_report` and `coefficient_table` (from
    `evaluate.py`);
  - the figures, from `plots.py`.
- **The tables come from `reports/*.csv`**, written by `student-risk train`. The notebook reads
  them; it does not recompute them.
- **Where the notebook does recompute something** (the out-of-fold threshold), it calls the
  package and then **checks with `assert`** that it gets exactly the saved threshold and the
  saved CSVs. If anything differed, the notebook would stop with an error.

**How to run it (checked):**

```bash
pip install -e ".[notebook]"
jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb
```

It takes about 4 seconds. It ran without errors on Python 3.12 and 3.10 in fresh environments.
The executed notebook, with its outputs, is saved, so the teachers can read it without running
it.

**The rule that no number may change:**

- After the notebook runs, the four `reports/*.csv` files are **byte-for-byte identical** to
  before. We checked with `md5sum`.
- The test numbers are the ones already obtained in Phase 3.
- The only new files are the three figures in `reports/figures/`.

---

## Setup cell

- It imports the package functions and loads `config.yaml`.
- `if not config.model_path.exists(): run_train(config)`: on a fresh clone there is no saved
  model yet, so the notebook runs the package's own `train` step. It is deterministic (seed 42),
  so it recreates exactly the same model and CSVs. **We checked this in fresh environments: the
  CSVs were identical.**
- `Trainer.load` then loads the saved model and threshold. **Every later section uses this one
  saved model.**

## 1. Problem framing (text)

- **Why it is first:** the subject's ML requirement 1, and the first grading criterion.
- **What it covers:**
  - who uses the tool (a school, at the start of the year);
  - the decision (offer tutoring);
  - the target (G3 < 10);
  - the cost of each mistake: a false negative is costly, a false positive is cheap.
- **Why it matters for the rest:** it justifies recall ≥ 0.80 as the goal, which drives the
  threshold choice.

## 2. Data exploration

- **Types, missing values, duplicates:** 17 text and 16 integer columns, 0 missing, 0 duplicates.
  This justifies that cleaning and imputation are only safety nets.
- **Class balance:** 15.4 % fail. This is why accuracy is never used.
- **Special code G3 = 0:** 15 students, 8 of whom had G2 > 0. It is our documented decision to
  keep them as fails, and section 10 tests it.
- **Leakage:** G1 and G2 have correlations of 0.83 and 0.92 with G3 (computed in the cell), and
  are not known early, so they are dropped. The cell also proves that none of G1, G2, G3 is
  among the 30 kept features.
- **The split:** 519 train / 130 test, with 80 / 20 fails. The cell checks that no student is in
  both sets.

## 3. Pipeline

- The cell prints the two branches of the `ColumnTransformer`:
  - numeric: impute, then scale;
  - categorical: impute, then one-hot.
- The text explains why the whole pipeline is refitted inside each fold: no leakage from
  validation folds.
- Only the *structure* is printed, not the default object display, because that display contains
  a memory address that changes on every run.

## 4. Baseline

- It reads `reports/baseline.csv`.
- The Dummy has recall 0, and its AP equals the fail rate. The untuned logistic regression has
  ROC-AUC 0.78 but recall 0.30 at 0.5.
- **Why show it:** the subject requires a baseline before the models. It is the reference that
  every model must beat.

## 5. Model comparison

- It reads `reports/model_comparison.csv`: mean and std of CV AP, ROC-AUC, train AP, recall and
  F1, plus the best hyperparameters.
- The text reminds the reader that all 4 models used **the same folds**, and were refitted on AP.

## 6. Selection rule

- It explains the "simplest model within one std of the best" rule with the actual numbers:
  - L1 has the best AP, 0.538 ± 0.158;
  - all 4 models are within one std of it;
  - so the rule picks `logreg_l2`.
- It notes that k-NN overfits (train AP 1.000) and that the CV score is optimistic (selection
  bias).
- The code cell shows that the saved model is indeed `logreg_l2` with C = 0.01.
- **No selection logic is re-implemented here.** The choice was made by `Trainer.select_best`
  during training.

## 7. Threshold on out-of-fold predictions

- `trainer.tune_threshold(X_train, y_train)` recomputes the out-of-fold probabilities, with the
  same 5 folds and the same model.
- **Three asserts** check that:
  1. the threshold equals the saved one (0.1403…);
  2. `threshold_comparison` equals `reports/threshold_report.csv`;
  3. `threshold_tradeoff` equals `reports/threshold_tradeoff.csv`.

  This is the "package reproduces the notebook" requirement, verified in code.
- **Figure `reports/figures/precision_recall_oof.png`:** the out-of-fold precision-recall curve,
  with the chosen threshold marked, and the random-guess level (precision = fail rate) as a dashed
  line.
- The text reads the trade-off table: recall 0.70 / 0.80 / 0.90 means flagging 28.7 % / 39.5 % /
  55.9 % of students.

## 8. Single test evaluation

- It scores the saved model and threshold on the 130 test students, with `final_scores`.
- The numbers are those already obtained in Phase 3:
  - ROC-AUC 0.778, AP 0.346;
  - recall 0.750 (15/20), precision 0.300;
  - 38.5 % flagged;
  - TN 75, FP 35, FN 5, TP 15.
- **Why this does not break the "test once" rule:** nothing is chosen from these numbers. The
  model and threshold were fixed before, and the notebook just reports the same single
  evaluation.
- **Wilson interval for the test recall:** `wilson_interval(15, 20)` gives [0.531, 0.888].
  - **Why an interval:** with only 20 failing students, one more or one fewer caught moves recall
    by 0.05. The interval shows that 0.75 is compatible with the 0.80 target.
  - **Why Wilson rather than "p ± 1.96·√(p(1−p)/n)":** see `evaluate.md`.
- **Figure `reports/figures/confusion_matrix_test.png`.**

## 9. Interpretation (coefficients)

- `coefficient_table(trainer.best_model)` gives one coefficient per encoded column (56), with
  readable names, sorted by size, plus the odds ratio. The cell adds the training standard
  deviation of each numeric feature, read from the fitted scaler. For example, the std of
  `failures` is 0.63.
- **How to read the coefficients, as written in the notebook:**
  - **Numeric features are standardised**, so a coefficient is the effect of +1 std.
  - **One-hot with every category kept** gives opposite coefficients for the two values of a
    yes/no feature. The *difference* is what matters: MS rather than GP adds 0.346.
  - C = 0.01 makes every coefficient small.
  - They are associations, not causes.
- **Top features (our results):**
  - `failures` +0.340, the strongest;
  - `school` MS +0.173 / GP −0.173;
  - `studytime` −0.167;
  - `Fedu` −0.109 and `Medu` −0.071;
  - `higher` no +0.096 / yes −0.096;
  - `freetime` +0.080, `Walc` +0.073, `absences` +0.058 and `goout` +0.049.
- **Figure `reports/figures/coefficients.png`:** the top 15 coefficients. Orange means the
  feature raises the risk, blue means it lowers it.

## 10. Sensitivity check: G3 = 0 (training set only)

- **Same model and hyperparameters:** `clone(trainer.best_model)` copies the pipeline with its
  settings (C = 0.01, ...) but without its fitted values. It is evaluated with the **same**
  `make_cv` folds and `SCORING`, on the **training set** with and without the 11 training
  students with G3 = 0.
- **The result:**
  - ROC-AUC 0.803 → 0.805;
  - AP 0.500 → 0.464, but the fail rate also drops (15.4 % → 13.6 %) and AP depends on it, so the
    two are not directly comparable. Both are within one std.
  - 9 of the 11 G3 = 0 students are flagged out-of-fold.
- **The "with" row equals the `logreg_l2` row of `model_comparison.csv`** (AP 0.500 ± 0.143,
  ROC-AUC 0.803). This is another consistency check.
- **Conclusion:** keeping them does not distort the model. **Nothing was changed.**

## 11. Fairness check (training set, out-of-fold)

- `group_report` applies the chosen threshold to the out-of-fold probabilities, per group: sex
  (F/M) and school (GP/MS). It reports group size, number of fails, recall, precision and share
  flagged. A small notebook helper adds the 95 % Wilson interval of each group's recall, using
  the package's `wilson_interval`.
- **Our results:**
  - By sex: F has recall 0.75 (40 fails) and 35.0 % flagged; M has recall 0.85 (40 fails) and
    45.8 % flagged.
  - By school: GP has recall 0.68 (25 fails) and 23.3 % flagged; MS has recall 0.86 (55 fails)
    and 70.0 % flagged.
- **⚠ The small-group warning:**
  - 25 to 55 fails per group gives wide intervals: GP recall [0.48, 0.83] against MS
    [0.74, 0.92].
  - The intervals overlap, so these are signals to monitor, not proven biases.
  - The school difference follows from the school coefficient: MS has a 31 % fail rate in the
    training data, against 7 % for GP.
- **Why only out-of-fold on the training set:** to compare groups we need predictions on
  students the model did not see. Using the test set for an *analysis* would be possible, but the
  rule is to touch it only once. The out-of-fold predictions also give more students (519
  instead of 130).
- **Nothing was changed.** Per-school thresholds are mentioned only as an option for a real
  school, and would be a new modelling decision.

## 12. Two limits (text)

1. **Small data and optimistic selection:**
   - fold-to-fold AP std ≈ 0.14;
   - the four models are tied;
   - selection bias (CV AP 0.500 against test AP 0.346);
   - only 20 test fails (recall interval [0.53, 0.89]).
2. **Limited signal without grades:**
   - recall 0.80 costs about 40 % of students flagged;
   - a linear model, with no interactions;
   - two Portuguese schools, one period **[from prior knowledge, to verify]**.

## 13. Risks for real students (text)

- **Errors:** about 1 in 4 failing students is missed, and about 2 in 3 flagged students would
  pass. So a flag must lead to an *offer* of help, never to a sanction or a label.
- **Fairness:** the model leans on school, past failures and parental education, which reflect
  social background, and group rates differ (section 11). Outcomes must be monitored per group,
  with a human in the loop.
- **Privacy:** the inputs include sensitive data (family, alcohol, health, relationships). This
  requires a legal basis and consent (for example the GDPR), access control, limited retention,
  and dropping features that are not necessary **[from prior knowledge, to verify]**. Families
  should be informed and able to contest a decision.

---

## How the notebook was produced

- The assistant wrote the cells with a small generator script (not part of the repository).
- It formatted the code cells with `ruff format`. ruff also lints notebooks, and the notebook
  passes `ruff check`.
- It executed the notebook with `jupyter nbconvert --execute --inplace`.
- **The text cells quote numbers.** Each number was taken from a run of the package, and then
  checked against the executed outputs.
