from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "drift_report.json"


def _missing_rate(series: pd.Series) -> float:
    return float(series.isna().mean())


def _top_values(series: pd.Series, limit: int = 5) -> dict[str, float]:
    shares = series.astype(str).value_counts(normalize=True, dropna=False).head(limit)
    return {str(index): float(value) for index, value in shares.items()}


def create_drift_report(reference_df: pd.DataFrame, new_df: pd.DataFrame) -> dict[str, Any]:
    """Compare basic feature distributions for a local monitoring simulation."""
    common_columns = sorted(set(reference_df.columns).intersection(new_df.columns))
    report: dict[str, Any] = {
        "report_type": "local_feature_drift_simulation",
        "reference_rows": int(len(reference_df)),
        "new_rows": int(len(new_df)),
        "columns_compared": common_columns,
        "features": {},
    }

    for column in common_columns:
        reference_series = reference_df[column]
        new_series = new_df[column]

        if is_numeric_dtype(reference_series) and is_numeric_dtype(new_series) and not is_bool_dtype(reference_series):
            reference_mean = float(reference_series.mean())
            new_mean = float(new_series.mean())
            report["features"][column] = {
                "type": "numeric",
                "reference": {
                    "mean": reference_mean,
                    "std": float(reference_series.std()),
                    "min": float(reference_series.min()),
                    "max": float(reference_series.max()),
                    "missing_rate": _missing_rate(reference_series),
                },
                "new": {
                    "mean": new_mean,
                    "std": float(new_series.std()),
                    "min": float(new_series.min()),
                    "max": float(new_series.max()),
                    "missing_rate": _missing_rate(new_series),
                },
                "mean_shift": new_mean - reference_mean,
                "missing_rate_shift": _missing_rate(new_series) - _missing_rate(reference_series),
            }
        else:
            report["features"][column] = {
                "type": "categorical",
                "reference_top_values": _top_values(reference_series),
                "new_top_values": _top_values(new_series),
                "missing_rate_shift": _missing_rate(new_series) - _missing_rate(reference_series),
            }

    return report


def write_drift_report(report: dict[str, Any], output_path: Path | str = DEFAULT_OUTPUT_PATH) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def generate_drift_report(
    reference_path: Path | str,
    new_path: Path | str,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
) -> Path:
    reference_df = pd.read_csv(reference_path)
    new_df = pd.read_csv(new_path)
    report = create_drift_report(reference_df, new_df)
    return write_drift_report(report, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a lightweight local drift report.")
    parser.add_argument("--reference", required=True, help="Path to the reference CSV dataset.")
    parser.add_argument("--new", required=True, help="Path to the new CSV dataset.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Path for the JSON report.")
    args = parser.parse_args()

    output_path = generate_drift_report(args.reference, args.new, args.output)
    print(f"Drift report written to {output_path}")


if __name__ == "__main__":
    main()
