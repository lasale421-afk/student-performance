# `config.yaml`: explained block by block

All the "magic numbers" and paths of the project live in this one file, not in the Python code.
If a teacher asks "what if you used 10 folds?", you change one line here and nothing in `src/`.
The file is read by `src/student_risk/config.py` (see `config.md`).

---

## Block 1: the seed

```yaml
random_seed: 42
```

- **What:** the only random seed of the project. It is used by the train/test split, the CV
  shuffling and the models.
- **Why:** with the same seed, every run gives the same split and the same scores, so our results
  can be reproduced.
- **Without it:** each run would give a slightly different split and different numbers, so we
  could not compare models fairly or reproduce a result in front of the teachers.

## Block 2: data

```yaml
data:
  uci_id: 320
  cache_path: data/student_performance.csv
  grade_column: G3
  pass_mark: 10
  leakage_columns: [G1, G2]
```

- `uci_id: 320`: the id of "Student Performance" on the UCI repository, used by `fetch_ucirepo`.
  The download has 649 students (checked on the real data).
  - **Source study:** P. Cortez and A. M. G. Silva, "Using Data Mining to Predict Secondary
    School Student Performance", Proceedings of the 5th Annual Future Business Technology
    Conference (FUBUTEC), 2008.
  - **Checked on the UCI page (Phase 3):**
    - the title, authors, venue and year above;
    - the dataset DOI 10.24432/C5TG7T;
    - the CC BY 4.0 licence;
    - "# Instances: 649";
    - that the dataset comes in two files, `student-mat.csv` (Math course) and `student-por.csv`
      (Portuguese language course).
  - **Still [from prior knowledge, to verify]:**
    - that the 649 students are the **Portuguese-language** course. The UCI page does not say
      which file id 320 returns, and the download does not say either;
    - the pages "pp. 5–12" and the city "Porto", which are not on the UCI page.
- `cache_path`: where the downloaded data is saved as a CSV. The path is relative to the project
  root, and `config.py` turns it into an absolute path. The folder `data/` is in `.gitignore`, so
  the dataset is not committed: it is downloaded again automatically on a fresh machine.
- `grade_column: G3`: the final grade, from 0 to 20.
- `pass_mark: 10`: in the Portuguese system, 10/20 is the pass mark **[from prior knowledge, to
  verify]**. In any case, 10 is the threshold set in our project specification. **Target:** `fail = 1` if
  `G3 < 10`, else `0`. A student with exactly 10 passes.
- `leakage_columns: [G1, G2]`: why we remove the first- and second-period grades:
  1. **They leak the answer.** G3 is strongly correlated with G1 and G2: on the real download,
     the correlation with G3 is 0.83 for G1 and 0.92 for G2. A student's grade barely changes
     between periods. A model using them would mostly copy G2 and look excellent,
     but it would learn nothing useful.
  2. **They are not known early.** Our use case is an *early* warning, at the start of the year,
     so tutors have time to help. G2 is only known late in the year, when it is too late to act.
  3. G3 itself is removed from the features after we build the target from it. Otherwise the
     model would see the answer directly.

## Block 3: split

```yaml
split:
  test_size: 0.2
```

- 20 % of the students (130 of 649) are kept aside as the test set. It is touched only once, at
  the very end.
- **Why 0.2:** a common compromise. With only 100 failing students in total (real data; 20 of
  them end up in the test set), a bigger test
  set would leave too few fails to train on. A smaller one would make the final test score very
  noisy.

## Block 4: cross-validation

```yaml
cv:
  n_splits: 5
```

- **What:** the number of folds of the StratifiedKFold used for every model comparison.
- **Why 5:** each validation fold has 103 or 104 students, 16 of whom fail (checked on the real
  training set). That is enough
  to compute recall and precision. With 10 folds, each fold would have only about 8 fails, so the
  scores would jump around a lot.

## Block 5: threshold goal

```yaml
threshold:
  target_recall: 0.80
  tradeoff_recalls: [0.70, 0.80, 0.90]
```

- **`tradeoff_recalls` (Phase 2b):** the other recall targets shown next to ours in
  `reports/threshold_tradeoff.csv`, so the school can see what each target would cost (see
  `evaluate.md`). Only `target_recall` is used to choose the threshold.
- **What:** when we choose the decision threshold (on out-of-fold predictions), we keep only thresholds where recall ≥ 0.80. Among those, we choose the one with
  the best precision.
- **Why we favour recall:**
  - **Recall** = "of all students who really fail, how many did we flag?"
  - **Precision** = "of all students we flagged, how many really fail?"
  - A **false negative** (a failing student that we miss) means a student who needed help gets
    none and may fail the year. A **false positive** (a student flagged who would have passed)
    only costs some tutoring time, and the extra help is rarely harmful.
  - So a missed student costs much more than an unneeded tutoring session, and we ask for high
    recall first. We still pick the best precision among those thresholds, so tutors are not
    flooded with false alarms.
- **Why 0.80 and not 0.95:** pushing recall close to 1 usually means flagging almost everyone,
  and then the tool is useless. "4 out of 5 failing students caught" is a stated, defensible goal.

## Block 6: paths (Phase 2)

```yaml
paths:
  reports_dir: reports
  model_path: models/model.joblib
```

- **`reports_dir`:** the folder where `student-risk train` writes its result tables:
  `baseline.csv`, `model_comparison.csv`, `threshold_report.csv` and `threshold_tradeoff.csv`.
- **`model_path` (Phase 3):** where `train` saves the model and its threshold, and where
  `evaluate` and `predict` load them.
  - `Trainer.save` creates `models/` if it is missing.
  - The `.joblib` file is git-ignored, and only `models/.gitkeep` is committed.
- **Why here:** this follows the "no hard-coded paths" rule. The path is relative to the
  project root, and `config.py` makes it absolute. The code creates the folder if it is missing.

## Block 7: hyperparameter grids (Phase 2, widened in Phase 2b)

```yaml
grids:
  logreg_l2:
    clf__C: [0.001, 0.01, 0.1, 1.0, 10.0]
    clf__class_weight: [null, balanced]
  logreg_l1:
    clf__C: [0.001, 0.01, 0.1, 1.0, 10.0]
    clf__class_weight: [null, balanced]
  linear_svm:
    clf__estimator__C: [0.001, 0.01, 0.1, 1.0, 10.0]
    clf__estimator__class_weight: [null, balanced]
  knn:
    clf__n_neighbors: [5, 11, 21, 31, 41, 51]
    clf__weights: [uniform, distance]
```

- **What:** one grid per model, given as is to `GridSearchCV`. GridSearchCV tries every
  combination, each on 5 folds:
  - 5 × 2 = 10 per linear model;
  - 6 × 2 = 12 for k-NN.
- **Why in the config:** the course's suggested layout puts the grids in `config.yaml`. Changing
  a grid then needs no code change.
- **The names:**
  - `clf__C` means "parameter `C` of the pipeline step named `clf`".
  - For the SVM, the step `clf` is a `CalibratedClassifierCV`, so we go one level deeper:
    `clf__estimator__C` is the `C` of the `LinearSVC` inside it.
  - A wrong name makes GridSearchCV raise an "Invalid parameter" error.
- `null` is YAML for Python `None`.
- **`C` values `[0.001, 0.01, 0.1, 1, 10]`:** a log scale, because C acts multiplicatively.
  Going from 1 to 2 matters much less than going from 0.01 to 1. Five values cover very strong
  to weak regularisation while keeping the search small. See `models_overview.md` for what C does.
- **`class_weight: [null, balanced]`:** fails are only 15 %. `balanced` makes each fail count
  more in the loss (about 5.5 passes, see `models_overview.md`). We let cross-validation decide
  whether it helps, instead of assuming it does.
- **`n_neighbors: [5, 11, 21, 31, 41, 51]`:** odd numbers avoid tied votes. The range goes
  from local (5) to smooth (51), around the rule of thumb √415 ≈ 20 **[from prior knowledge, to verify:
  the rule of thumb]**.
- **`weights: [uniform, distance]`:** all neighbours equal, or closer neighbours count more.
- **k-NN has no `class_weight`:** the scikit-learn class has no such parameter. The imbalance is
  handled by the threshold instead.
- **Why the grids were widened (Phase 2b, our results):** with the first grids (C from 0.01,
  k up to 31), three models chose an **edge** value: C = 0.01 for `logreg_l2` and `linear_svm`,
  and k = 31 for `knn`. When the best value sits at the edge, a better one may lie outside the
  grid. So we added C = 0.001 and k = 41, 51. After widening:
  - `logreg_l2` (C = 0.01) and `knn` (k = 31) kept the same values. These are now inside the
    grid, so they are real optima.
  - `linear_svm` moved to C = 0.001 (`balanced`), again the lowest value. We did not widen
    further (see `models.md`).
  - The chosen model and the threshold did not change.
