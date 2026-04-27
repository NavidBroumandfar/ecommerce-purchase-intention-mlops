from pathlib import Path

from fastapi.testclient import TestClient
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.api import main as api_main
from src.models.predict import predict_purchase as real_predict_purchase
from src.models.train import save_training_artifacts, train_model_candidates


def _training_dataframe(rows: int = 40) -> pd.DataFrame:
    records = []
    for index in range(rows):
        page_values = float((index % 8) * 5.0)
        revenue = bool(page_values >= 20.0 or index % 9 == 0)
        records.append(
            {
                "Administrative": index % 4,
                "Administrative_Duration": float(index * 2.0),
                "Informational": index % 2,
                "Informational_Duration": float((index % 3) * 5.0),
                "ProductRelated": 6 + (index % 25),
                "ProductRelated_Duration": float((6 + (index % 25)) * 18.0),
                "BounceRates": float((index % 4) / 100),
                "ExitRates": float((index % 6) / 70),
                "PageValues": page_values,
                "SpecialDay": 0.0,
                "Month": ["Feb", "Mar", "May", "Nov"][index % 4],
                "OperatingSystems": (index % 3) + 1,
                "Browser": (index % 4) + 1,
                "Region": (index % 5) + 1,
                "TrafficType": (index % 6) + 1,
                "VisitorType": ["Returning_Visitor", "New_Visitor"][index % 2],
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


def _payload() -> dict:
    return {
        "Administrative": 1,
        "Administrative_Duration": 18.2,
        "Informational": 0,
        "Informational_Duration": 0.0,
        "ProductRelated": 12,
        "ProductRelated_Duration": 340.1,
        "BounceRates": 0.01,
        "ExitRates": 0.03,
        "PageValues": 12.3,
        "SpecialDay": 0.0,
        "Month": "Mar",
        "OperatingSystems": 2,
        "Browser": 2,
        "Region": 3,
        "TrafficType": 2,
        "VisitorType": "Returning_Visitor",
        "Weekend": False,
    }


def test_health_endpoint() -> None:
    client = TestClient(api_main.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "project": "ecommerce-purchase-intention-mlops",
    }


def test_predict_endpoint_returns_prediction(monkeypatch, tmp_path: Path) -> None:
    df = _training_dataframe()
    result = train_model_candidates(
        df,
        model_candidates=_light_model_candidates(),
        test_size=0.25,
        random_state=42,
    )
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "model_metadata.json"
    save_training_artifacts(
        result,
        model_path=model_path,
        metrics_path=tmp_path / "metrics.json",
        metadata_path=metadata_path,
    )

    def predict_with_temp_artifact(input_data: dict) -> dict:
        return real_predict_purchase(
            input_data,
            model_path=model_path,
            metadata_path=metadata_path,
        )

    monkeypatch.setattr(api_main, "predict_purchase", predict_with_temp_artifact)
    client = TestClient(api_main.app)

    response = client.post("/predict", json=_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] in {0, 1}
    assert 0.0 <= body["purchase_probability"] <= 1.0
    assert body["model_version"] == "portfolio-mvp-v1"


def test_predict_endpoint_returns_503_when_model_is_missing(monkeypatch) -> None:
    def raise_missing_artifact(input_data: dict) -> dict:
        raise FileNotFoundError("Model artifact not found")

    monkeypatch.setattr(api_main, "predict_purchase", raise_missing_artifact)
    client = TestClient(api_main.app)

    response = client.post("/predict", json=_payload())

    assert response.status_code == 503
    assert "Model artifact not found" in response.json()["detail"]
