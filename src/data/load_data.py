from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "online_shoppers_intention.csv"
TARGET_COLUMN = "Revenue"


def validate_dataset(df: pd.DataFrame) -> None:
    """Validate the minimum schema required for training and evaluation."""
    if df.empty:
        raise ValueError("Dataset is empty. Add rows before training the model.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Dataset is missing the required target column '{TARGET_COLUMN}'. "
            "Use the UCI Online Shoppers Purchasing Intention CSV with the original schema."
        )


def load_dataset(path: Path | str | None = None) -> pd.DataFrame:
    """Load a CSV dataset and validate that it contains the target column."""
    dataset_path = Path(path) if path is not None else DEFAULT_DATA_PATH

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{dataset_path}'. "
            "Download the UCI Online Shoppers Purchasing Intention dataset and place it at "
            "'data/raw/online_shoppers_intention.csv'. The raw dataset is intentionally excluded from Git."
        )

    df = pd.read_csv(dataset_path)
    validate_dataset(df)
    return df
