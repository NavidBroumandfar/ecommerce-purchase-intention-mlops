from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.load_data import DEFAULT_DATA_PATH, TARGET_COLUMN, load_dataset
from src.models.train import METADATA_PATH, METRICS_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = PROJECT_ROOT / "reports" / "model_results.md"
UCI_DATASET_PAGE = "https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset"
UCI_DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"


def _load_json(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found at '{path}'. "
            "Run 'python3 -m src.models.train' before generating the results report."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _format_model_name(model_name: str) -> str:
    return model_name.replace("_", " ").title()


def _display_path(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _target_distribution(df: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    counts = df[TARGET_COLUMN].value_counts().to_dict()
    shares = df[TARGET_COLUMN].value_counts(normalize=True).to_dict()

    return {
        str(label): {
            "count": int(count),
            "share": float(shares[label]),
        }
        for label, count in counts.items()
    }


def _model_comparison_table(metrics: dict[str, Any], best_model_name: str) -> str:
    lines = [
        "| Model | ROC-AUC | Precision | Recall | F1-score | Confusion Matrix [[TN, FP], [FN, TP]] |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]

    for model_name, values in metrics["models"].items():
        display_name = _format_model_name(model_name)
        if model_name == best_model_name:
            display_name = f"**{display_name}**"

        lines.append(
            "| "
            + " | ".join(
                [
                    display_name,
                    _format_metric(values["roc_auc"]),
                    _format_metric(values["precision"]),
                    _format_metric(values["recall"]),
                    _format_metric(values["f1"]),
                    str(values["confusion_matrix"]),
                ]
            )
            + " |"
        )

    return "\n".join(lines)


def build_model_results_report(
    metrics: dict[str, Any],
    metadata: dict[str, Any],
    df: pd.DataFrame,
    dataset_path: Path | str = DEFAULT_DATA_PATH,
) -> str:
    """Build the Markdown model results report from saved artifacts."""
    best_model_name = metrics["best_model_name"]
    best_model_display_name = _format_model_name(best_model_name)
    best_metrics = metrics["models"][best_model_name]
    target_distribution = _target_distribution(df)

    distribution_lines = [
        "| Revenue | Sessions | Share |",
        "| --- | ---: | ---: |",
    ]
    for label, values in target_distribution.items():
        distribution_lines.append(
            f"| {label} | {values['count']:,} | {values['share']:.2%} |"
        )

    return f"""# Model Results

## Evaluation Summary

Evaluation date: `{metadata.get("created_at_utc", "unknown")}`

Dataset source: UCI Online Shoppers Purchasing Intention Dataset

Official dataset page: {UCI_DATASET_PAGE}

Official download URL used locally: {UCI_DOWNLOAD_URL}

Local dataset path: `{dataset_path}`

Target variable: `{TARGET_COLUMN}`

Dataset shape: `{df.shape[0]:,}` rows x `{df.shape[1]}` columns

## Target Distribution

{chr(10).join(distribution_lines)}

The dataset is imbalanced: most sessions do not end in purchase. This makes precision and recall more informative than accuracy alone.

## Train/Test Split

- Split approach: deterministic holdout split
- Test size: `{metadata.get("test_size")}`
- Stratified split: `{metadata.get("stratified_split")}`
- Random state: `{metadata.get("random_state")}`
- Train rows: `{metadata.get("train_rows"):,}`
- Test rows: `{metadata.get("test_rows"):,}`

## Model Candidates

- Logistic Regression with balanced class weights
- Random Forest Classifier with balanced class weights

Both candidates use the same scikit-learn preprocessing pipeline: numeric features are scaled and categorical or boolean-like features are one-hot encoded.

## Model Comparison

{_model_comparison_table(metrics, best_model_name)}

## Selected Model

Selected best model: `{best_model_name}`

Selection metric: `{metrics.get("selection_metric")}`

Best model metrics:

- ROC-AUC: `{_format_metric(best_metrics["roc_auc"])}`
- Precision: `{_format_metric(best_metrics["precision"])}`
- Recall: `{_format_metric(best_metrics["recall"])}`
- F1-score: `{_format_metric(best_metrics["f1"])}`
- Confusion matrix `[[TN, FP], [FN, TP]]`: `{best_metrics["confusion_matrix"]}`

## Business Interpretation

The {best_model_display_name} baseline achieved the strongest ROC-AUC in this run, which means it ranked purchasing sessions better than the other baseline candidate on the deterministic test split.

The precision/recall trade-off matters for this use case. Because most sessions do not convert, a model can look strong on ranking metrics while still missing many purchasing sessions or producing false positives. A business team would need to choose an operating threshold based on the cost of missed purchases versus unnecessary interventions.

## Limitations

- Results are based on the local UCI CSV at `data/raw/online_shoppers_intention.csv`.
- The raw dataset is not committed to Git.
- The evaluation uses a deterministic holdout split, not time-based validation.
- The model is a baseline portfolio model, not a production-ready decision system.
- Probability calibration and threshold tuning are not included in this MVP.
- Business impact is not validated with live experiments or intervention outcomes.
"""


def generate_model_results_report(
    metrics_path: Path | str = METRICS_PATH,
    metadata_path: Path | str = METADATA_PATH,
    dataset_path: Path | str = DEFAULT_DATA_PATH,
    report_path: Path | str = REPORT_PATH,
) -> Path:
    metrics = _load_json(metrics_path)
    metadata = _load_json(metadata_path)
    df = load_dataset(dataset_path)

    report = build_model_results_report(
        metrics=metrics,
        metadata=metadata,
        df=df,
        dataset_path=_display_path(dataset_path),
    )

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> None:
    try:
        report_path = generate_model_results_report()
        print(f"Model results report written to {report_path}")
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Report generation failed: {exc}") from None


if __name__ == "__main__":
    main()
