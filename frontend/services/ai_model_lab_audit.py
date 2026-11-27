"""Read-only validation analytics for AURUM-AI V5 AI Model Lab.

Reuses the established Phase-5 loader (with log-return and timing checks).
No models are trained, no historical reports are modified, and no Test set is read.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from frontend.services.ai_prediction_engine import (
    HORIZONS,
    MODELS,
    PredictionArtifactError,
    evaluation_metrics,
    experiment_file,
    read_validation,
    temporal_audit,
)

EXPERIMENT_SUBDIR = Path("artifacts/experiments/nyse_v3")


def report_path(root: Path, model: str, horizon: int) -> Path:
    return experiment_file(root, model, horizon).with_name("report.json")


def inventory(root: Path) -> pd.DataFrame:
    """List the expected 12 Validation experiments, including absent artifacts."""
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        for model in MODELS:
            npz = experiment_file(root, model, horizon)
            report = report_path(root, model, horizon)
            rows.append({
                "Modèle": model,
                "Horizon (min)": horizon,
                "NPZ Validation": "Présent" if npz.is_file() else "Absent",
                "Rapport JSON": "Présent" if report.is_file() else "Absent",
                "Taille NPZ (Ko)": round(npz.stat().st_size / 1024, 1) if npz.is_file() else None,
                "Expérience": npz.parent.name,
            })
    return pd.DataFrame(rows)


def file_sha256(path: Path, *, block_size: int = 1024 * 1024) -> str:
    """Calculate an actual fingerprint, never claim it is externally anchored."""
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(block_size), b""):
            sha.update(chunk)
    return sha.hexdigest()


def read_report(path: Path) -> dict[str, Any]:
    """Read metadata only; unrecognized report schemas are not interpreted."""
    with Path(path).open("r", encoding="utf-8") as stream:
        content = json.load(stream)
    if not isinstance(content, dict):
        raise ValueError("Le rapport doit être un objet JSON")
    return content


def validation_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    """Calculate errors on the exact Validation observations used by the model.

    The zero-return baseline predicts a log-return of zero on those SAME rows.
    Directional agreement is descriptive, not trading profitability.
    """
    basic = evaluation_metrics(frame)
    true = frame["y_true"].to_numpy(dtype=np.float64)
    pred = frame["y_pred"].to_numpy(dtype=np.float64)
    delta = pred - true
    zero_rmse = float(np.sqrt(np.mean(true * true)))
    rmse_improvement = (
        100 * (zero_rmse - basic["rmse"]) / zero_rmse if zero_rmse > 0 else None
    )
    return {
        "observations": int(basic["n"]),
        "mae": float(basic["mae"]),
        "rmse": float(basic["rmse"]),
        "zero_mae": float(basic["zero_mae"]),
        "zero_rmse": zero_rmse,
        "mae_improvement_pct": basic["mae_improvement_pct"],
        "rmse_improvement_pct": rmse_improvement,
        "direction_agreement_pct": float(basic["direction_accuracy_pct"]),
        "bias": float(np.mean(delta)),
        "median_absolute_error": float(np.median(np.abs(delta))),
        "p95_absolute_error": float(np.quantile(np.abs(delta), .95)),
        "max_absolute_error": float(np.max(np.abs(delta))),
    }


def comparison_for_horizon(root: Path, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-model metrics from NPZ, recording failures instead of inventing data.

    Returns (valid_metrics, errors). A broken model is not silently replaced.
    """
    if int(horizon) not in HORIZONS:
        raise PredictionArtifactError("Horizon de validation non autorisé")
    valid, errors = [], []
    for model in MODELS:
        path = experiment_file(root, model, horizon)
        if not path.is_file():
            errors.append({"Modèle": model, "Erreur": "Validation NPZ absent"})
            continue
        try:
            frame = read_validation(path, model=model, horizon=horizon)
            stats = validation_diagnostics(frame)
            valid.append({
                "Modèle": model,
                "Horizon (min)": int(horizon),
                "Observations": stats["observations"],
                "MAE": stats["mae"],
                "RMSE": stats["rmse"],
                "Zero Return MAE": stats["zero_mae"],
                "Zero Return RMSE": stats["zero_rmse"],
                "Gain MAE (%)": stats["mae_improvement_pct"],
                "Accord directionnel (%)": stats["direction_agreement_pct"],
            })
        except (OSError, ValueError, KeyError, TypeError, EOFError) as exc:
            errors.append({"Modèle": model, "Erreur": str(exc)})
    return pd.DataFrame(valid), pd.DataFrame(errors)


def temporal_check(frame: pd.DataFrame) -> pd.DataFrame:
    """Only structural checks: a zero is NOT proof of no leakage."""
    return pd.DataFrame([
        {"Contrôle": label.replace("_", " "), "Anomalies": int(value)}
        for label, value in temporal_audit(frame).items()
    ])


def top_residuals(frame: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    if limit < 1:
        raise ValueError("limit doit être positif")
    cols = [
        "end_timestamp_utc", "target_timestamp_utc", "y_true", "y_pred",
        "absolute_error_log_return",
    ]
    return (
        frame.nlargest(limit, "absolute_error_log_return")[cols]
        .rename(columns={
            "end_timestamp_utc": "Fin fenêtre UTC",
            "target_timestamp_utc": "Cible UTC",
            "y_true": "Log-return observé",
            "y_pred": "Log-return prédit",
            "absolute_error_log_return": "Erreur absolue",
        })
        .reset_index(drop=True)
    )


def summary_export(model: str, horizon: int, metrics: dict[str, Any],
                   checks: pd.DataFrame, path: Path) -> bytes:
    """Build a downloadable JSON in memory, without writing a project file."""
    safe_metrics = {
        k: (None if v is None or (isinstance(v, float) and not np.isfinite(v))
            else v) for k, v in metrics.items()
    }
    obj = {
        "scope": "historical_validation_only",
        "no_test_metrics": True,
        "model": str(model),
        "horizon_minutes": int(horizon),
        "source_file": path.as_posix(),
        "source_sha256": file_sha256(path),
        "metrics_log_return": safe_metrics,
        "temporal_metadata_checks": {
            str(row["Contrôle"]): int(row["Anomalies"])
            for _, row in checks.iterrows()
        },
        "limitations": [
            "No inference, no Test-set claim, no real trading.",
            "Temporal metadata checks cannot prove absence of data leakage.",
            "Baseline uses the same Validation observations.",
        ],
    }
    return json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
