# How the 4 models work (overview)

This page explains the theory behind the models compared in `models.py`. It is general ML
knowledge, **not** something we measured, so all of it is **[from prior knowledge, to verify]**
against the course slides (Labs 1 and 2). The few statements that come from our own results are
labelled "(our results)".

All 4 models receive the *same* preprocessed input: scaled numbers and one-hot columns, 56
columns in total (our results).

---

## 1. Logistic regression

**How it predicts:**

- It computes a weighted sum of the features: `z = b + w1·x1 + w2·x2 + …`.
- It passes `z` through the **sigmoid** function `σ(z) = 1 / (1 + e^(−z))`, which squeezes any
  number into (0, 1). The result is read as P(fail).
- `z = 0` gives 0.5. A large positive `z` gives a value close to 1, a large negative `z` a value
  close to 0.

**How it learns:**

- It looks for the weights that minimise the **log-loss** (cross-entropy):
  `−[y·log(p) + (1−y)·log(1−p)]`, averaged over the students.
- The loss is small when the model gives a high p to students who fail and a low p to students who
  pass. It is very large when the model is confidently wrong.

**Reading a weight:**

- A positive weight means "a larger value of this feature increases the risk of failing", the
  other features being equal.
- Because the numeric features are standardised, a weight is the change in log-odds for **+1
  standard deviation** of that feature.
- `e^w` is the multiplication factor on the odds.

**Regularisation:** a penalty on large weights is added to the loss, to avoid fitting noise.

- **L2 (Ridge-style):** the penalty is `Σ w²`.
  - It shrinks all weights smoothly towards 0, but rarely to exactly 0.
  - Correlated features share the weight between them.
- **L1 (Lasso-style):** the penalty is `Σ |w|`.
  - It pushes many weights to **exactly 0**, so it also selects features. The model is sparser
    and easier to read.
  - Among correlated features, it tends to keep one and drop the others, somewhat arbitrarily.

**Hyperparameter `C`** (in scikit-learn, C is the *inverse* of the penalty strength):

- **C too small** means a very strong penalty. The weights are crushed towards 0 and the
  probabilities are pulled towards the base rate. Ranking can still be fine, but the model
  **underfits**.
  - (our results) The chosen `logreg_l2` has C = 0.01. Its out-of-fold probabilities have median
    0.12 and maximum 0.63.
  - With L1, a very small C can set *every* weight to 0, giving a constant prediction.
- **C too large** means almost no penalty. The weights can grow to fit noise in 519 students, so
  the model **overfits**: the train score goes up and the CV score goes down.

**Hyperparameter `class_weight`:**

- With `None`, every student counts the same in the loss. With 15 % fails, the model can lower
  the loss a lot just by predicting low probabilities for everyone.
- `"balanced"` multiplies the loss of each class by `n_samples / (2 · n_class)`. With our 80
  fails out of 519 (our results): 519 / (2 · 80) ≈ 3.2 for each fail and 519 / (2 · 439) ≈ 0.6
  for each pass. So one fail counts about as much as 5.5 passes. This raises the predicted
  probabilities of fails.
- It mostly *shifts* the probabilities, and changes the *ranking* less. So it matters most when
  you use the 0.5 threshold. We choose our own threshold, which also handles the imbalance.
  (our results) `None` won for both logistic regressions, and `balanced` won for the SVM.

## 2. k-nearest neighbours (k-NN)

**How it predicts:**

- To predict a new student, it finds the **k most similar students** in the training set. The
  similarity is the Euclidean distance on the scaled features.
- The probability of failing is the share of fails among those k neighbours.
- It **learns nothing** in advance. It stores the training set, and all the work happens at
  prediction time.

**Why scaling is essential:** without it, the feature with the largest units (for example
`absences`, 0 to 32) would decide the distances by itself.

**Hyperparameter `n_neighbors` (k):**

- **k too small** (for example 1): the prediction follows single, possibly noisy students. The
  model **overfits**, and the probabilities can only take a few values (0 or 1 when k = 1).
- **k too large:** the neighbourhood covers a big part of the data, every student gets close to
  the global fail rate, and the model **underfits**.
- **Odd values** avoid tied votes between the two classes.
- Our grid is 5 to 51. A common rule of thumb is k ≈ √n ≈ √415 ≈ 20. (our results) The best
  value is k = 31.

**Hyperparameter `weights`:**

- `uniform`: every neighbour counts the same.
- `distance`: closer neighbours count more (weight = 1 / distance).
  - (our results) With `distance`, train AP = 1.000. A training student is its own neighbour at
    distance 0, so it predicts itself perfectly. This is a classic sign of overfitting on the
    training data.

**No `class_weight`:** `KNeighborsClassifier` has no such parameter. With 15 % fails, most
neighbourhoods contain mostly passes, so the probabilities are low. Again, the threshold step
compensates.

**Limits:**

- It is slow on big data, because it measures the distance to every training student.
- Distances lose meaning with many dimensions.
- There are no weights to interpret.

## 3. Linear SVM (`LinearSVC`)

**How it predicts:**

- It is also a linear model: `z = b + w·x`. It predicts "fail" if `z > 0`.
- The difference with logistic regression is *how the weights are learned*.

**The margin:**

- Among all the lines (hyperplanes) that separate the classes, the SVM looks for the one with the
  **largest margin**, meaning the largest distance to the closest students of each class. Those
  closest students are the "support vectors".
- The loss is the **hinge loss**, `max(0, 1 − y·z)` with y in {−1, +1}:
  - students correctly classified *and* beyond the margin cost **nothing**;
  - only students inside the margin, or on the wrong side, push the line.
- So the SVM focuses on the hard, borderline cases. Logistic regression's log-loss, in contrast,
  uses every student a little.

**Hyperparameter `C`:**

- **C small:** a wide margin, with many violations allowed. The model is simpler and may underfit.
- **C large:** a narrow margin, which tries to classify every training student correctly. It may
  overfit, and it needs more iterations to converge.
- (our results) The best value was C = 0.001 with `balanced` weights. This is still the smallest
  value of the widened grid.

**`class_weight`:** the same idea as in logistic regression. Errors on fails cost more.

**Why calibration (`CalibratedClassifierCV`):**

- `LinearSVC` outputs a score `z`, not a probability. The hinge loss does not produce
  probabilities.
- `CalibratedClassifierCV` with `method="sigmoid"` fits a small logistic curve,
  `P(fail) = 1 / (1 + e^(A·z + B))`, that maps the SVM score to a probability. This is "Platt
  scaling".
- It uses **internal** cross-validation (`cv=3`): it trains the SVM on part of the data and fits
  the curve on the other part. So the curve is learned on scores of students the SVM did not see.
- The final model averages the 3 calibrated SVMs.
- **Without it:** no `predict_proba`. We could not compute AP from probabilities, choose a
  probability threshold, or have `predict` print "probability of failing".

---

## Summary table

| model | type | learns | key hyperparameters | too small | too large |
|---|---|---|---|---|---|
| Logistic regression L2 | linear | weights, by minimising log-loss + Σw² | C, class_weight | C: underfit, probabilities near the base rate | C: overfit |
| Logistic regression L1 | linear, sparse | weights, by minimising log-loss + Σ\|w\| | C, class_weight | C: all weights at 0 | C: overfit, less sparse |
| Linear SVM + calibration | linear | widest-margin line (hinge loss), then a sigmoid | C, class_weight | C: wide margin, underfit | C: narrow margin, overfit |
| k-NN | instance-based | nothing (stores the data) | n_neighbors, weights | k: overfit, noisy | k: underfit, everyone near the base rate |
