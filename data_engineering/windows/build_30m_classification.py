
from pathlib import Path
import json
import numpy as np


# ==================================================
# AURUM-AI — 30M CLASSIFICATION DATASETS
# UP / DOWN / NEUTRAL
# Threshold determined from TRAIN only
# ==================================================

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "05_model_data"
REPORT_PATH = ROOT / "reports" / "classification_30m_audit.json"

HORIZON = 30
NEUTRAL_QUANTILE = 0.30

CLASS_NAMES = ["DOWN", "NEUTRAL", "UP"]


def load_split(split):
    path = DATA_DIR / f"{split}_30m_normalized.npz"

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        result = {
            "X": data["X"].astype(np.float32),
            "y": data["y"].astype(np.float64),
            "feature_names": data["feature_names"].copy(),
            "end_timestamp_ms": data[
                "end_timestamp_ms"
            ].astype(np.int64),
            "target_timestamp_ms": data[
                "target_timestamp_ms"
            ].astype(np.int64),
        }

    X = result["X"]
    y = result["y"]
    end = result["end_timestamp_ms"]
    target = result["target_timestamp_ms"]

    if X.ndim != 3 or X.shape[1:] != (20, 6):
        raise ValueError(f"Unexpected X shape: {split}")

    if not (len(X) == len(y) == len(end) == len(target)):
        raise ValueError(f"Length mismatch: {split}")

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(f"Invalid numeric data: {split}")

    if not np.all(target - end == HORIZON * 60_000):
        raise ValueError(f"Incorrect target alignment: {split}")

    if len(np.unique(end)) != len(end):
        raise ValueError(f"Duplicate timestamps: {split}")

    return result


def classify(y, threshold):
    # Class 0: DOWN
    # Class 1: NEUTRAL
    # Class 2: UP
    return np.where(
        y < -threshold,
        0,
        np.where(y > threshold, 2, 1)
    ).astype(np.int64)


def distribution(labels):
    counts = np.bincount(labels, minlength=3)
    total = len(labels)

    return {
        name: {
            "count": int(counts[i]),
            "percentage": float(counts[i] / total * 100),
        }
        for i, name in enumerate(CLASS_NAMES)
    }


def main():
    print("\n=== AURUM-AI | 30M CLASSIFICATION ===")

    outputs = {
        split: DATA_DIR / f"{split}_30m_classification.npz"
        for split in ("train", "validation")
    }

    # Protect previously generated datasets.
    for path in [*outputs.values(), REPORT_PATH]:
        if path.exists():
            raise FileExistsError(
                f"Existing artifact preserved: {path}"
            )

    train = load_split("train")
    validation = load_split("validation")

    if not np.array_equal(
        train["feature_names"],
        validation["feature_names"]
    ):
        raise ValueError("Feature names mismatch")

    if (
        train["target_timestamp_ms"].max()
        >= validation["end_timestamp_ms"].min()
    ):
        raise ValueError("Chronological separation failed")

    # Critical: select the threshold using TRAIN only.
    threshold = float(
        np.quantile(
            np.abs(train["y"]),
            NEUTRAL_QUANTILE,
        )
    )

    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Invalid classification threshold")

    print(f"\nNeutral quantile: {NEUTRAL_QUANTILE:.0%}")
    print(f"Threshold (log return): {threshold:.10f}")
    print(f"Approx threshold: {threshold * 100:.5f}%")

    report = {
        "project": "AURUM-AI",
        "task": "30m three-class forecasting",
        "classes": CLASS_NAMES,
        "threshold_source": "train_only",
        "neutral_quantile": NEUTRAL_QUANTILE,
        "threshold_log_return": threshold,
        "splits": {},
        "test_dataset_used": False,
    }

    for split, dataset in [
        ("train", train),
        ("validation", validation),
    ]:
        labels = classify(dataset["y"], threshold)
        stats = distribution(labels)

        print(f"\n=== {split.upper()} ===")
        print(f"Windows: {len(labels)}")

        for class_name in CLASS_NAMES:
            values = stats[class_name]
            print(
                f"{class_name}: {values['count']} "
                f"({values['percentage']:.2f}%)"
            )

        np.savez_compressed(
            outputs[split],
            X=dataset["X"],
            y_class=labels,
            y_log_return=dataset["y"].astype(np.float32),
            feature_names=dataset["feature_names"],
            end_timestamp_ms=dataset["end_timestamp_ms"],
            target_timestamp_ms=dataset["target_timestamp_ms"],
            threshold_log_return=np.array(threshold),
        )

        report["splits"][split] = {
            "windows": int(len(labels)),
            "distribution": stats,
            "file": str(outputs[split]),
        }

        print(f"Saved: {outputs[split]}")

    # Reference majority class selected from TRAIN only.
    train_counts = np.bincount(
        classify(train["y"], threshold),
        minlength=3,
    )

    majority_class = int(np.argmax(train_counts))

    validation_labels = classify(
        validation["y"], threshold
    )

    baseline_accuracy = float(
        np.mean(validation_labels == majority_class) * 100
    )

    print("\n=== BASELINE ===")
    print(f"Train majority class: {CLASS_NAMES[majority_class]}")
    print(f"Validation accuracy: {baseline_accuracy:.4f}%")

    report["majority_baseline"] = {
        "selected_from": "train",
        "class": CLASS_NAMES[majority_class],
        "validation_accuracy_pct": baseline_accuracy,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"\nAudit saved: {REPORT_PATH}")
    print("Original datasets were NOT modified.")
    print("Test dataset was NOT used.")
    print("\n30M CLASSIFICATION DATASETS COMPLETED")


if __name__ == "__main__":
    main()
