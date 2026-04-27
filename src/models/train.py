from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.load_data import load_dataset
from src.features.preprocessing import (
    TARGET_COLUMN,
    build_preprocessor,
    get_feature_column_types,
    split_features_target,
)

PROJECT_NAME = "ecommerce-purchase-intention-mlops"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
RANDOM_STATE = 42
TEST_SIZE = 0.2
SELECTION_METRIC = "roc_auc"


def get_model_candidates(random_state: int = RANDOM_STATE) -> dict[str, BaseEstimator]:
    """Return the baseline model candidates for this MVP."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def _can_use_stratified_split(y: pd.Series, test_size: float | int) -> bool:
    class_counts = y.value_counts()
    if len(class_counts) < 2 or class_counts.min() < 2:
        return False

    if isinstance(test_size, float):
        test_rows = math.ceil(len(y) * test_size)
    else:
        test_rows = test_size

    train_rows = len(y) - test_rows
    return test_rows >= len(class_counts) and train_rows >= len(class_counts)


def make_train_test_split(
    df: pd.DataFrame,
    test_size: float | int = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, bool]:
    """Create the deterministic train/test split used by training and evaluation."""
    X, y = split_features_target(df)
    stratify = y if _can_use_stratified_split(y, test_size) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    return X_train, X_test, y_train, y_test, stratify is not None


def build_model_pipeline(model: BaseEstimator, X_train: pd.DataFrame) -> Pipeline:
    """Create a full preprocessing-plus-model scikit-learn pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("model", model),
        ]
    )


def _positive_class_probabilities(model: Pipeline, X: pd.DataFrame) -> list[float]:
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model does not support probability estimates.")

    probabilities = model.predict_proba(X)
    classes = list(getattr(model, "classes_", []))
    if 1 in classes:
        positive_class_index = classes.index(1)
    else:
        positive_class_index = 1 if probabilities.shape[1] > 1 else 0

    return probabilities[:, positive_class_index].tolist()


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Evaluate a fitted model with classification metrics useful for an imbalanced target."""
    y_pred = model.predict(X_test)
    probabilities = _positive_class_probabilities(model, X_test)

    roc_auc = None
    if y_test.nunique() == 2:
        roc_auc = float(roc_auc_score(y_test, probabilities))

    return {
        "roc_auc": roc_auc,
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).astype(int).tolist(),
    }


def _select_best_model(metrics: dict[str, dict[str, Any]]) -> str:
    def score(model_name: str) -> float:
        value = metrics[model_name].get(SELECTION_METRIC)
        return float("-inf") if value is None else float(value)

    return max(metrics, key=score)


def train_model_candidates(
    df: pd.DataFrame,
    model_candidates: dict[str, BaseEstimator] | None = None,
    test_size: float | int = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Train model candidates and return the best fitted pipeline plus metadata."""
    X_train, X_test, y_train, y_test, used_stratified_split = make_train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
    )

    candidates = model_candidates or get_model_candidates(random_state=random_state)
    trained_models: dict[str, Pipeline] = {}
    metrics: dict[str, dict[str, Any]] = {}

    for model_name, model in candidates.items():
        pipeline = build_model_pipeline(model, X_train)
        pipeline.fit(X_train, y_train)
        trained_models[model_name] = pipeline
        metrics[model_name] = evaluate_model(pipeline, X_test, y_test)

    best_model_name = _select_best_model(metrics)
    categorical_columns, numeric_columns = get_feature_column_types(X_train)

    metadata = {
        "project": PROJECT_NAME,
        "model_version": "portfolio-mvp-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_column": TARGET_COLUMN,
        "best_model_name": best_model_name,
        "selection_metric": SELECTION_METRIC,
        "random_state": random_state,
        "test_size": test_size,
        "stratified_split": used_stratified_split,
        "training_rows": int(len(df)),
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "feature_columns": list(X_train.columns),
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
    }

    return {
        "trained_models": trained_models,
        "metrics": metrics,
        "best_model_name": best_model_name,
        "best_model": trained_models[best_model_name],
        "metadata": metadata,
    }


def save_training_artifacts(
    training_result: dict[str, Any],
    model_path: Path | str = MODEL_PATH,
    metrics_path: Path | str = METRICS_PATH,
    metadata_path: Path | str = METADATA_PATH,
) -> dict[str, Path]:
    """Persist the best model, comparison metrics, and model metadata."""
    model_path = Path(model_path)
    metrics_path = Path(metrics_path)
    metadata_path = Path(metadata_path)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(training_result["best_model"], model_path)

    metrics_payload = {
        "best_model_name": training_result["best_model_name"],
        "selection_metric": SELECTION_METRIC,
        "models": training_result["metrics"],
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(training_result["metadata"], indent=2), encoding="utf-8")

    return {
        "model_path": model_path,
        "metrics_path": metrics_path,
        "metadata_path": metadata_path,
    }


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def print_model_comparison(metrics: dict[str, dict[str, Any]], best_model_name: str) -> None:
    """Print a compact model comparison table."""
    print("\nModel comparison on deterministic test split")
    print("model                 roc_auc  precision  recall  f1")
    print("--------------------  -------  ---------  ------  -----")
    for model_name, values in metrics.items():
        print(
            f"{model_name:<20}  "
            f"{_format_metric(values['roc_auc']):>7}  "
            f"{_format_metric(values['precision']):>9}  "
            f"{_format_metric(values['recall']):>6}  "
            f"{_format_metric(values['f1']):>5}"
        )
    print(f"\nBest model: {best_model_name} selected by ROC-AUC")


def main() -> None:
    try:
        df = load_dataset()
        training_result = train_model_candidates(df)
        paths = save_training_artifacts(training_result)
        print_model_comparison(training_result["metrics"], training_result["best_model_name"])
        print("\nSaved artifacts:")
        for label, path in paths.items():
            print(f"- {label}: {path}")
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Training failed: {exc}") from None


if __name__ == "__main__":
    main()
