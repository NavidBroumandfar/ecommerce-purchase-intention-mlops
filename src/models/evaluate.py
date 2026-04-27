from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from src.data.load_data import load_dataset
from src.models.train import MODEL_PATH, evaluate_model, make_train_test_split, print_model_comparison


def load_model_artifact(model_path: Path | str = MODEL_PATH) -> Any:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at '{model_path}'. "
            "Run 'python3 -m src.models.train' after placing the raw dataset in data/raw/. "
            "If you are inside a virtual environment that provides 'python', that command works too."
        )
    return joblib.load(model_path)


def evaluate_saved_model(
    data_path: Path | str | None = None,
    model_path: Path | str = MODEL_PATH,
) -> dict[str, Any]:
    """Evaluate a saved model artifact on the deterministic test split."""
    model = load_model_artifact(model_path)
    df = load_dataset(data_path)
    _, X_test, _, y_test, _ = make_train_test_split(df)
    return evaluate_model(model, X_test, y_test)


def print_evaluation(metrics: dict[str, Any]) -> None:
    print_model_comparison({"saved_model": metrics}, "saved_model")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}")


def main() -> None:
    try:
        metrics = evaluate_saved_model()
        print_evaluation(metrics)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Evaluation failed: {exc}") from None


if __name__ == "__main__":
    main()
