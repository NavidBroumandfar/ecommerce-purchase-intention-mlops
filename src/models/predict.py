from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.joblib"
METADATA_PATH = PROJECT_ROOT / "artifacts" / "model_metadata.json"


def load_model(model_path: Path | str = MODEL_PATH) -> Any:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at '{model_path}'. "
            "Train the model first with 'python3 -m src.models.train'. "
            "If you are inside a virtual environment that provides 'python', that command works too."
        )
    return joblib.load(model_path)


def load_model_metadata(metadata_path: Path | str = METADATA_PATH) -> dict[str, Any] | None:
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _prepare_input_frame(input_data: dict[str, Any], metadata: dict[str, Any] | None) -> pd.DataFrame:
    frame = pd.DataFrame([input_data])
    expected_columns = metadata.get("feature_columns") if metadata else None

    if expected_columns:
        missing_columns = [column for column in expected_columns if column not in frame.columns]
        if missing_columns:
            raise ValueError(
                "Prediction input is missing required feature columns: "
                + ", ".join(missing_columns)
            )
        frame = frame.loc[:, expected_columns]

    return frame


def predict_purchase(
    input_data: dict[str, Any],
    model_path: Path | str = MODEL_PATH,
    metadata_path: Path | str = METADATA_PATH,
) -> dict[str, Any]:
    """Predict whether a single visitor session is likely to result in a purchase."""
    model = load_model(model_path)
    metadata = load_model_metadata(metadata_path)
    input_frame = _prepare_input_frame(input_data, metadata)

    predicted_class = int(model.predict(input_frame)[0])

    if not hasattr(model, "predict_proba"):
        raise ValueError("The loaded model does not support purchase probability estimates.")

    probabilities = model.predict_proba(input_frame)
    classes = list(getattr(model, "classes_", []))
    if 1 in classes:
        positive_class_index = classes.index(1)
    else:
        positive_class_index = 1 if probabilities.shape[1] > 1 else 0

    purchase_probability = float(probabilities[0, positive_class_index])

    result: dict[str, Any] = {
        "predicted_class": predicted_class,
        "purchase_probability": purchase_probability,
    }

    if metadata:
        result["model_version"] = metadata.get("model_version")
        result["model_metadata"] = metadata

    return result
