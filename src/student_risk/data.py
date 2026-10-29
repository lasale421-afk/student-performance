"""Load the UCI Student Performance data, clean it, build the target and split it."""

import pandas as pd
from sklearn.model_selection import train_test_split

from student_risk.config import Config

TARGET_NAME = "fail"


class DatasetLoader:
    """Download, clean, build the target and split the Student Performance data.

    Args:
        config: Project settings (dataset id, cache path, pass mark, seed...).
    """

    def __init__(self, config: Config) -> None:
        """Store the settings used by every step."""
        self.config = config

    def load_raw(self) -> pd.DataFrame:
        """Return the raw dataset, from the local CSV cache or from UCI.

        If the cache file does not exist, the data is downloaded with ucimlrepo
        and saved to the cache, so later runs work without the network.

        Returns:
            One DataFrame with the 30 features and the grades G1, G2, G3.
        """
        if self.config.cache_path.exists():
            return pd.read_csv(self.config.cache_path)

        # Imported here so that tests and cached runs never need the network library.
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=self.config.uci_id)
        raw = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

        self.config.cache_path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(self.config.cache_path, index=False)
        return raw

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a cleaned copy of the data.

        Steps: strip spaces in text columns, drop rows without a final grade,
        drop exact duplicate rows. Each step looks at one row at a time, so it
        learns nothing from the data and is safe to run before the split.

        Args:
            df: Raw data.

        Returns:
            Cleaned data with a fresh 0..n-1 index.
        """
        df = df.copy()
        text_columns = df.select_dtypes(include="object").columns
        df[text_columns] = df[text_columns].apply(lambda column: column.str.strip())

        df = df.dropna(subset=[self.config.grade_column])
        df = df.drop_duplicates()
        return df.reset_index(drop=True)

    def build_target(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Split the data into features X and the binary target y.

        The target is fail = 1 if the final grade is below the pass mark, else 0.
        The final grade and the leakage columns (G1, G2) are removed from X.

        Args:
            df: Cleaned data that still contains the grade columns.

        Returns:
            A tuple (X, y): the feature table and the 0/1 target.
        """
        y = (df[self.config.grade_column] < self.config.pass_mark).astype(int)
        y.name = TARGET_NAME

        columns_to_drop = [self.config.grade_column, *self.config.leakage_columns]
        X = df.drop(columns=columns_to_drop)
        return X, y

    def split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split once into train and test sets, stratified on the target.

        Args:
            X: Features.
            y: Target.

        Returns:
            X_train, X_test, y_train, y_test.
        """
        return train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            stratify=y,
            random_state=self.config.random_seed,
        )

    def load_and_split(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Run every step: load, clean, build target, split.

        Returns:
            X_train, X_test, y_train, y_test.
        """
        df = self.clean(self.load_raw())
        X, y = self.build_target(df)
        return self.split(X, y)
