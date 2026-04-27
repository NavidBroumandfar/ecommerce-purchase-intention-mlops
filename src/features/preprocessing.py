from __future__ import annotations

from pandas.api.types import (
    is_bool_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "Revenue"

KNOWN_CATEGORICAL_COLUMNS = {
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
}

TRUE_VALUES = {"true", "1", "yes", "y", "t"}
FALSE_VALUES = {"false", "0", "no", "n", "f"}


def _coerce_target_to_binary(y: pd.Series) -> pd.Series:
    if y.isna().any():
        raise ValueError(f"Target column '{TARGET_COLUMN}' contains missing values.")

    if is_bool_dtype(y):
        return y.astype(int)

    if is_numeric_dtype(y):
        unique_values = set(y.dropna().unique().tolist())
        if not unique_values.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(
                f"Numeric target column '{TARGET_COLUMN}' must contain only 0/1 values. "
                f"Found: {sorted(unique_values)}"
            )
        return y.astype(int)

    normalized = y.astype(str).str.strip().str.lower()
    mapped = normalized.map({value: 1 for value in TRUE_VALUES} | {value: 0 for value in FALSE_VALUES})

    if mapped.isna().any():
        invalid_values = sorted(y[mapped.isna()].astype(str).unique().tolist())
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' must be boolean-like or binary. "
            f"Could not map values: {invalid_values}"
        )

    return mapped.astype(int)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataset into feature columns and a binary target series."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset is missing the required target column '{TARGET_COLUMN}'.")

    X = df.drop(columns=[TARGET_COLUMN]).copy()
    if X.empty:
        raise ValueError("Dataset must contain at least one feature column.")

    y = _coerce_target_to_binary(df[TARGET_COLUMN])
    return X, y


def get_feature_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return categorical and numeric feature columns using UCI-aware defaults."""
    categorical_columns: list[str] = []
    numeric_columns: list[str] = []

    for column in X.columns:
        series = X[column]
        if (
            column in KNOWN_CATEGORICAL_COLUMNS
            or is_bool_dtype(series)
            or is_object_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            categorical_columns.append(column)
        elif is_numeric_dtype(series):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)

    return categorical_columns, numeric_columns


def _build_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a scikit-learn preprocessing transformer for mixed feature types."""
    categorical_columns, numeric_columns = get_feature_column_types(X)

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", StandardScaler(), numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", _build_one_hot_encoder(), categorical_columns))

    if not transformers:
        raise ValueError("No usable feature columns were found for preprocessing.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def prepare_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """Split features and target, then build a preprocessor for the features."""
    X, y = split_features_target(df)
    preprocessor = build_preprocessor(X)
    return X, y, preprocessor
