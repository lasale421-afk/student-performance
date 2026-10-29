# `src/student_risk/features.py`: explained line by line

**Role:** build the preprocessing step: fill missing values, scale numbers, and one-hot encode
text. It returns it **unfitted**, so it is always fitted *inside* a Pipeline and only on training
data. This is our main protection against data leakage.

---

## Imports

```python
"""Build the preprocessing step: impute, scale numeric columns, one-hot encode text columns."""

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
```

- `ColumnTransformer`: applies different processing to different columns, then sticks the
  results side by side.
- `make_column_selector`: chooses columns by their type (numeric or not).
- `SimpleImputer`: fills missing values.
- `Pipeline`: chains steps (impute, then scale).
- `OneHotEncoder`, `StandardScaler`: the encoding and scaling steps.

## The function

```python
def build_preprocessor() -> ColumnTransformer:
```

It takes no arguments and returns a fresh, unfitted object each time. **Why a function and not a
global object:** each model, and each CV fold, must get its *own* new preprocessor. Sharing one
fitted object between models would mix their statistics.

## Numeric branch

```python
    numeric_steps = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
```

- `SimpleImputer(strategy="median")` replaces a missing number with the **median of the training
  data**.
  - **Why the median:** it is robust to extreme values. `absences` has a few very large values,
    which would pull the mean up.
  - **Why keep an imputer if the dataset has no missing values:** a new student sent to `predict`
    may have an empty field. Without the imputer, scikit-learn would crash on `NaN`. It costs
    nothing when there is nothing to fill.
- `StandardScaler()` turns each column into (value − mean) / std, learned on the training data.
  - **Why:** k-NN uses distances, and SVM and logistic regression use regularisation that
    punishes large coefficients. Without scaling, `absences` (0 to 30) would dominate `studytime`
    (1 to 4) only because of its units. Scaling also makes logistic regression coefficients
    comparable with each other, which helps the interpretation.
- The order matters: we impute first, because the scaler cannot handle `NaN`.

## Categorical branch

```python
    categorical_steps = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
```

- `strategy="most_frequent"`: a missing text value gets the most common category of the training
  data. A median is impossible for text.
- `OneHotEncoder` turns, for example, `Mjob ∈ {teacher, health, ...}` into one 0/1 column per
  category. **Why:** models need numbers. Writing teacher = 1, health = 2... would invent a false
  order.
- `handle_unknown="ignore"`: a category never seen in training (for example a typo in a JSON
  case) becomes all zeros instead of crashing the prediction.
- `sparse_output=False`: the output is a normal dense array. With only about 60 columns, the
  memory gain of a sparse matrix is useless, and dense arrays are easier to inspect and test.

## Putting the branches together

```python
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_steps, make_column_selector(dtype_include="number")),
            ("categorical", categorical_steps, make_column_selector(dtype_exclude="number")),
        ]
    )
```

- Each entry is `(name, steps, which columns)`. The names `"numeric"` and `"categorical"` let us
  inspect the fitted parts later, as the no-leakage test does with
  `named_transformers_["numeric"]`.
- `make_column_selector(dtype_include="number")` selects every numeric column, and
  `dtype_exclude="number"` selects everything else (the text columns).
  - **Why by dtype and not a hand-written list:** no list to maintain, and the same code works on
    the small synthetic test tables.
  - **What it means for this dataset:**
    - 13 numeric columns go to the scaler: `age`, `absences`, `failures`, and ordinal scores such
      as `Medu`, `studytime` and `Dalc`.
    - 17 text columns go to the one-hot encoder: `school`, `sex`, `Mjob`, the yes/no columns...
  - **Limit to know:** ordinal scores such as `Medu` (0 to 4) are treated as numbers, which
    assumes equal steps between levels. This is simple and common, and it keeps the number of
    columns small.
- By default, `ColumnTransformer` **drops** any column not selected (`remainder="drop"`). Every
  column is either numeric or not, so nothing is lost here.

## Why "unfitted" is the key point

The function never calls `.fit`. The statistics (medians, means, standard deviations, the list of
categories) are learned when the full `Pipeline` is fitted:

- inside cross-validation, on the training folds only;
- at the end, on the training set only.

If we fitted the scaler on all 649 students before splitting, the test students' mean would leak
into training. The test `test_scaler_is_fitted_on_train_only` checks this.
