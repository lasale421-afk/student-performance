"""Build the preprocessing step: impute, scale numeric columns, one-hot encode text columns."""

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor() -> ColumnTransformer:
    """Create the (unfitted) preprocessing step used in every model pipeline.

    Numeric columns: fill missing values with the median, then standardise.
    Text (categorical) columns: fill missing values with the most frequent value,
    then one-hot encode. Columns are chosen by dtype when the step is fitted.

    Returns:
        An unfitted ColumnTransformer. It learns its statistics only when the
        pipeline is fitted, so inside cross-validation it only sees training folds.
    """
    numeric_steps = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_steps = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_steps, make_column_selector(dtype_include="number")),
            ("categorical", categorical_steps, make_column_selector(dtype_exclude="number")),
        ]
    )
