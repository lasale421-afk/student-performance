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
  - This is the Portuguese-language course, from the original study: P. Cortez and A. Silva,
    "Using Data Mining to Predict Secondary School Student Performance", Proceedings of the 5th
    Future Business Technology Conference (FUBUTEC 2008), Porto, Portugal, pp. 5–12.
    **[from prior knowledge, to verify]**: the course and the citation details. The downloaded
    file does not say which course it is.
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
```

- **What:** when we choose the decision threshold (in a later phase, on out-of-fold
  predictions), we keep only thresholds where recall ≥ 0.80. Among those, we choose the one with
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
