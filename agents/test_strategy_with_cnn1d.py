
from pathlib import Path
import sys
import json
from collections import Counter

import numpy as np
import torch

from strategy_agent import StrategyAgent


# ===================================================
# AURUM-AI — CNN1D + STRATEGY AGENT
# Validation only — NO real or simulated execution
# ===================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0, str(ROOT / "deep_learning" / "cnn1d")
)

from train_cnn1d import CNN1DForecaster


DATA_PATH = (
    ROOT / "data" / "05_model_data"
    / "validation_normalized.npz"
)

MODEL_PATH = (
    ROOT / "artifacts" / "models"
    / "cnn1d_best.pt"
)

REPORT_PATH = (
    ROOT / "reports"
    / "strategy_agent_cnn1d_validation.json"
)

THRESHOLD = 0.0003
BATCH_SIZE = 512


def main():

    print("\n=== AURUM-AI | CNN1D + STRATEGY AGENT ===")

    torch.set_num_threads(4)

    # 1. Load validation only
    with np.load(DATA_PATH, allow_pickle=False) as data:
        X = data["X"].astype(np.float32)
        features = data["feature_names"].tolist()

    if not np.isfinite(X).all():
        raise ValueError("Invalid validation features")

    print(f"\nValidation windows: {len(X)}")

    # 2. Restore trained CNN1D
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["feature_names"] != features:
        raise ValueError("Feature order mismatch")

    if checkpoint["window_size"] != X.shape[1]:
        raise ValueError("Window size mismatch")

    if checkpoint["input_size"] != X.shape[2]:
        raise ValueError("Input size mismatch")

    scale = float(checkpoint["target_scale"])

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid target scale")

    model = CNN1DForecaster(
        input_size=X.shape[2]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # 3. Generate real validation forecasts
    predictions = []

    with torch.no_grad():
        for start in range(0, len(X), BATCH_SIZE):

            xb = torch.from_numpy(
                X[start:start + BATCH_SIZE]
            )

            output = model(xb)

            predictions.extend(
                (output.cpu().numpy() / scale).tolist()
            )

    predictions = np.asarray(
        predictions, dtype=np.float64
    )

    if not np.isfinite(predictions).all():
        raise ValueError("Invalid predictions")

    # 4. Convert forecasts into strategy decisions
    agent = StrategyAgent(threshold=THRESHOLD)

    decisions = [
        agent.decide(float(value))
        for value in predictions
    ]

    counts = Counter(
        decision.action for decision in decisions
    )

    # 5. Show results
    print("\n=== STRATEGY RESULTS ===")

    for action in ("BUY", "SELL", "HOLD"):
        count = counts[action]

        print(
            f"{action}: {count} "
            f"({100 * count / len(decisions):.2f}%)"
        )

    print("\n=== FIRST 5 DECISIONS ===")

    for index, decision in enumerate(
        decisions[:5], start=1
    ):
        print(
            f"{index}. "
            f"Forecast: {decision.predicted_return:+.8f} | "
            f"Decision: {decision.action} | "
            f"Reason: {decision.reason}"
        )

    # 6. Save summary, without executing trades
    report = {
        "project": "AURUM-AI",
        "forecast_model": "CNN1D",
        "agent": "StrategyAgent",
        "evaluation_split": "validation",
        "total_windows": len(predictions),
        "threshold_log_return": THRESHOLD,
        "decision_counts": {
            action: int(counts[action])
            for action in ("BUY", "SELL", "HOLD")
        },
        "example_decisions": [
            decision.to_dict()
            for decision in decisions[:5]
        ],
        "orders_executed": False,
        "test_dataset_used": False,
    }

    REPORT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    with open(
        REPORT_PATH, "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    print(f"\nReport saved: {REPORT_PATH}")

    print("No trades were executed.")
    print("Test dataset was NOT used.")
    print("\nSTRATEGY INTEGRATION COMPLETED")


if __name__ == "__main__":
    main()
