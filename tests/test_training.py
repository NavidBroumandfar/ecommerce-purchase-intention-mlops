from pathlib import Path
import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.models.train import save_training_artifacts, train_model_candidates


def _training_dataframe(rows: int = 40) -> pd.DataFrame:
    months = ["Feb", "Mar", "May", "Nov"]
    visitor_types = ["Returning_Visitor", "New_Visitor", "Other"]

    records = []
    for index in range(rows):
        page_values = float((index % 8) * 4.5)
        product_related = 5 + (index % 30)
        revenue = bool(page_values >= 18.0 or index % 11 == 0)
        records.append(
            {
                "Administrative": index % 5,
                "Administrative_Duration": float(index * 3.2),
                "Informational": index % 3,
                "Informational_Duration": float((index % 4) * 7.0),
                "ProductRelated": product_related,
                "ProductRelated_Duration": float(product_related * 24.5),
                "BounceRates": float((index % 5) / 100),
                "ExitRates": float((index % 7) / 80),
                "PageValues": page_values,
                "SpecialDay": 0.0,
                "Month": months[index % len(months)],
                "OperatingSystems": (index % 3) + 1,
                "Browser": (index % 4) + 1,
                "Region": (index % 5) + 1,
                "TrafficType": (index % 6) + 1,
                "VisitorType": visitor_types[index % len(visitor_types)],
                "Weekend": bool(index % 2),
                "Revenue": revenue,
            }
        )

    return pd.DataFrame(records)


def _light_model_candidates() -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=500, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=10,
            class_weight="balanced",
            random_state=42,
        ),
    }


def test_training_produces_metrics_and_saves_artifacts(tmp_path: Path) -> None:
    df = _training_dataframe()

    result = train_model_candidates(
        df,
        model_candidates=_light_model_candidates(),
        test_size=0.25,
        random_state=42,
    )

    assert result["best_model_name"] in result["metrics"]
    assert "roc_auc" in result["metrics"][result["best_model_name"]]
    assert "confusion_matrix" in result["metrics"][result["best_model_name"]]
    assert result["metadata"]["stratified_split"] is True
    assert result["metadata"]["random_state"] == 42
    assert result["metadata"]["test_size"] == 0.25

    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    metadata_path = tmp_path / "model_metadata.json"

    paths = save_training_artifacts(
        result,
        model_path=model_path,
        metrics_path=metrics_path,
        metadata_path=metadata_path,
    )

    assert paths["model_path"].exists()
    assert paths["metrics_path"].exists()
    assert paths["metadata_path"].exists()

    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics_payload["best_model_name"] == result["best_model_name"]
