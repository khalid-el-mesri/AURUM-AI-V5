
import json
import math
from pathlib import Path

import pandas as pd


# AURUM-AI — Deep Learning Model Comparison
# Validation only — Test dataset remains sealed.

ROOT = Path(__file__).resolve().parents[2]

REPORTS_DIR = (
    ROOT / "artifacts" / "experiments" / "nyse_v3"
)

MODELS = ("LSTM", "GRU", "CNN1D")
HORIZONS = (5, 15, 30, 60)


def load_model_comparison():
    """Collect MAE and RMSE from all 12 Validation reports."""

    rows = []

    for horizon in HORIZONS:
        for model in MODELS:

            report_path = (
                REPORTS_DIR
                / f"{model.lower()}_{horizon}m_seed42"
                / "report.json"
            )

            with open(
                report_path,
                encoding="utf-8",
            ) as file:
                report = json.load(file)

            if report["test_dataset_used"] is not False:
                raise ValueError(
                    "Test Dataset must remain unused"
                )

            if (
                report["model"].upper() != model
                or report["horizon_minutes"] != horizon
            ):
                raise ValueError(
                    f"Invalid report metadata: {report_path}"
                )

            metrics = report[
                "validation_metrics_selected_on_validation"
            ]

            baseline = report["zero_return_validation"]

            mae = float(metrics["MAE"])
            rmse = float(metrics["RMSE"])
            baseline_mae = float(baseline["MAE"])
            baseline_rmse = float(baseline["RMSE"])

            if not all(
                math.isfinite(value) and value >= 0
                for value in (
                    mae, rmse, baseline_mae, baseline_rmse
                )
            ):
                raise ValueError("Invalid evaluation metrics")

            if baseline_mae == 0:
                raise ValueError("Invalid baseline MAE")

            improvement = (
                100 * (baseline_mae - mae) / baseline_mae
            )

            rows.append({
                "Model": model,
                "Horizon (min)": horizon,
                "MAE": mae,
                "RMSE": rmse,
                "Zero Return MAE": baseline_mae,
                "Zero Return RMSE": baseline_rmse,
                "MAE Improvement (%)": improvement,
            })

    comparison = pd.DataFrame(rows)

    if len(comparison) != 12:
        raise ValueError("Expected exactly 12 experiments")

    return comparison
