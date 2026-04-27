from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.models.predict import predict_purchase
from src.models.train import save_training_artifacts, train_model_candidates


def _training_dataframe(rows: int = 40) -> pd.DataFrame:
    records = []
    for index in range(rows):
        page_values = float((index % 8) * 5.0)
        revenue = bool(page_values >= 20.0 or index % 13 == 0)
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


def test_predict_purchase_returns_class_and_probability(tmp_path: Path) -> None:
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

    input_data = df.drop(columns=["Revenue"]).iloc[0].to_dict()
    prediction = predict_purchase(
        input_data,
        model_path=model_path,
        metadata_path=metadata_path,
    )

    assert prediction["predicted_class"] in {0, 1}
    assert 0.0 <= prediction["purchase_probability"] <= 1.0
    assert prediction["model_version"] == "portfolio-mvp-v1"


def test_predict_purchase_does_not_require_metadata(tmp_path: Path) -> None:
    df = _training_dataframe()
    result = train_model_candidates(
        df,
        model_candidates=_light_model_candidates(),
        test_size=0.25,
        random_state=42,
    )

    model_path = tmp_path / "model.joblib"
    save_training_artifacts(
        result,
        model_path=model_path,
        metrics_path=tmp_path / "metrics.json",
        metadata_path=tmp_path / "model_metadata.json",
    )

    input_data = df.drop(columns=["Revenue"]).iloc[0].to_dict()
    prediction = predict_purchase(
        input_data,
        model_path=model_path,
        metadata_path=tmp_path / "missing_model_metadata.json",
    )

    assert prediction["predicted_class"] in {0, 1}
    assert 0.0 <= prediction["purchase_probability"] <= 1.0
    assert "model_version" not in prediction
    assert "model_metadata" not in prediction
