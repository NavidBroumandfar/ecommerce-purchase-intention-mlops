from pathlib import Path
import json

import pandas as pd

from src.models.generate_report import generate_model_results_report


def test_generate_model_results_report_uses_artifacts_and_dataset(tmp_path: Path) -> None:
    metrics = {
        "best_model_name": "random_forest",
        "selection_metric": "roc_auc",
        "models": {
            "logistic_regression": {
                "roc_auc": 0.81,
                "precision": 0.42,
                "recall": 0.70,
                "f1": 0.52,
                "confusion_matrix": [[3, 1], [1, 1]],
            },
            "random_forest": {
                "roc_auc": 0.88,
                "precision": 0.67,
                "recall": 0.50,
                "f1": 0.57,
                "confusion_matrix": [[4, 0], [1, 1]],
            },
        },
    }
    metadata = {
        "created_at_utc": "2026-04-27T00:00:00+00:00",
        "test_size": 0.2,
        "stratified_split": True,
        "random_state": 42,
        "train_rows": 4,
        "test_rows": 2,
    }
    dataset = pd.DataFrame(
        {
            "Administrative": [0, 1, 2, 3, 4, 5],
            "Month": ["Feb", "Mar", "May", "Nov", "Dec", "Mar"],
            "Revenue": [False, False, False, False, True, True],
        }
    )

    metrics_path = tmp_path / "metrics.json"
    metadata_path = tmp_path / "model_metadata.json"
    dataset_path = tmp_path / "online_shoppers_intention.csv"
    report_path = tmp_path / "model_results.md"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    dataset.to_csv(dataset_path, index=False)

    generated_path = generate_model_results_report(
        metrics_path=metrics_path,
        metadata_path=metadata_path,
        dataset_path=dataset_path,
        report_path=report_path,
    )

    report = generated_path.read_text(encoding="utf-8")
    assert generated_path == report_path
    assert "Dataset shape: `6` rows x `3` columns" in report
    assert "| True | 2 | 33.33% |" in report
    assert "ROC-AUC: `0.880`" in report
    assert "false negatives" in report
    assert "missed buyers" in report
