
from pathlib import Path
import csv
import json
import math


# ============================================================
# AURUM-AI — AI MODEL ARENA
# Compare all models using VALIDATION results only
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"

OUTPUT_CSV = REPORTS_DIR / "model_arena_validation.csv"
OUTPUT_JSON = REPORTS_DIR / "model_arena_validation.json"


# Model name, report filename, metrics field, subfield
SOURCES = [
    (
        "Zero Return",
        "model_baselines_validation.json",
        "metrics",
        "Zero Return",
    ),
    (
        "Historical Mean",
        "model_baselines_validation.json",
        "metrics",
        "Historical Mean",
    ),
    (
        "Last Return",
        "model_baselines_validation.json",
        "metrics",
        "Last Return",
    ),
    (
        "Ridge",
        "ridge_validation.json",
        "ridge",
        None,
    ),
    (
        "XGBoost",
        "xgboost_validation.json",
        "validation_metrics",
        None,
    ),
    (
        "LSTM",
        "lstm_validation.json",
        "validation_metrics",
        None,
    ),
    (
        "GRU",
        "gru_validation.json",
        "validation_metrics",
        None,
    ),
    (
        "CNN1D",
        "cnn1d_validation.json",
        "validation_metrics",
        None,
    ),
    (
        "TCN",
        "tcn_validation.json",
        "validation_metrics",
        None,
    ),
    (
        "Transformer",
        "transformer_validation.json",
        "validation_metrics",
        None,
    ),
]


def load_model_metrics(name, filename, field, subfield):

    path = REPORTS_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing report for {name}: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        report = json.load(file)

    # Prevent accidental comparison with test results
    if report.get("evaluation_split") != "validation":
        raise ValueError(
            f"{name}: evaluation split must be validation."
        )

    # Verify prediction target
    if report.get("target") != "next_5min_log_return":
        raise ValueError(
            f"{name}: incorrect prediction target."
        )

    scores = report[field]

    if subfield is not None:
        scores = scores[subfield]

    mae = float(scores["MAE"])
    rmse = float(scores["RMSE"])

    if not all(
        math.isfinite(value) and value >= 0
        for value in (mae, rmse)
    ):
        raise ValueError(
            f"{name}: invalid MAE or RMSE."
        )

    return {
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
    }


def main():

    print("\n========================================")
    print("       AURUM-AI | AI MODEL ARENA")
    print("========================================")

    results = []

    # Load all validation reports
    for name, filename, field, subfield in SOURCES:

        result = load_model_metrics(
            name,
            filename,
            field,
            subfield,
        )

        results.append(result)

    # Rank by MAE, then RMSE
    results.sort(
        key=lambda row: (row["MAE"], row["RMSE"])
    )

    # Assign ranks
    for index, result in enumerate(results, start=1):
        result["rank"] = index

    print("\nVALIDATION RANKING — LOWER IS BETTER\n")

    print(
        f"{'Rank':<7}"
        f"{'Model':<22}"
        f"{'MAE':<17}"
        f"{'RMSE':<17}"
    )

    print("-" * 63)

    for result in results:

        print(
            f"{result['rank']:<7}"
            f"{result['model']:<22}"
            f"{result['MAE']:<17.8f}"
            f"{result['RMSE']:<17.8f}"
        )

    # Best model according to validation MAE
    best_mae_model = min(
        results,
        key=lambda row: row["MAE"],
    )

    # Best model according to validation RMSE
    best_rmse_model = min(
        results,
        key=lambda row: row["RMSE"],
    )

    print("\n=== BEST VALIDATION MODELS ===")

    print(
        f"Best MAE  : {best_mae_model['model']} "
        f"({best_mae_model['MAE']:.8f})"
    )

    print(
        f"Best RMSE : {best_rmse_model['model']} "
        f"({best_rmse_model['RMSE']:.8f})"
    )

    print(f"\nModels compared: {len(results)}")

    # Save results as CSV
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "rank",
                "model",
                "MAE",
                "RMSE",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    # Save results as JSON
    arena_report = {
        "project": "AURUM-AI",
        "comparison": "AI Model Arena",
        "evaluation_split": "validation",
        "target": "next_5min_log_return",
        "ranking_metric": "MAE",
        "models_compared": len(results),
        "best_mae_model": best_mae_model["model"],
        "best_rmse_model": best_rmse_model["model"],
        "results": results,
        "test_dataset_used": False,
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            arena_report,
            file,
            indent=2,
        )

    print("\n=== FILES SAVED ===")

    print(f"CSV report : {OUTPUT_CSV}")
    print(f"JSON report: {OUTPUT_JSON}")

    print("\nIMPORTANT:")
    print("Comparison uses VALIDATION results only.")
    print("Test dataset was NOT used.")
    print("Predictive accuracy does not prove profitability.")

    print("\nAI MODEL ARENA COMPLETED")


if __name__ == "__main__":
    main()
