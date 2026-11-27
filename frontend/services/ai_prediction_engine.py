"""AURUM-AI V5: strictly read-only historical VALIDATION prediction audit.

The NYSE V3 experiment labels are LOG RETURNS (natural log of future/current
closing prices), not USD prices and not simple percentage returns.

No weights are loaded. No inference or live market data are performed.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

MODELS = ("LSTM", "GRU", "CNN1D")
HORIZONS = (5, 15, 30, 60)
REQUIRED_KEYS = (
    "y_true", "y_pred", "end_timestamp_ms", "target_timestamp_ms",
    "signal_available_at_ms", "target_available_at_ms",
    "current_close", "next_close",
)


class PredictionArtifactError(ValueError):
    """Prediction artifact is missing or inconsistent."""


def experiment_file(root: Path, model: str, horizon: int) -> Path:
    model = str(model).upper()
    horizon = int(horizon)
    if model not in MODELS or horizon not in HORIZONS:
        raise PredictionArtifactError("Modèle ou horizon non autorisé")
    return Path(root) / f"{model.lower()}_{horizon}m_seed42" / "validation_predictions.npz"


def available_experiments(root: Path) -> list[tuple[str, int]]:
    return [(m, h) for h in HORIZONS for m in MODELS
            if experiment_file(root, m, h).is_file()]


def _as_numeric(array: np.ndarray, key: str, dtype: str) -> np.ndarray:
    if array.ndim != 1 or array.dtype.kind not in "iuf":
        raise PredictionArtifactError(f"{key}: tableau numérique unidimensionnel requis")
    if dtype == "int64" and array.dtype.kind not in "iu":
        raise PredictionArtifactError(f"{key}: horodatages entiers requis")
    output = array.astype(dtype, copy=False)
    if not np.isfinite(output).all():
        raise PredictionArtifactError(f"{key}: valeurs non finies")
    return output


def read_validation(path: Path, *, model: str, horizon: int) -> pd.DataFrame:
    """Validate source NPZ, preserving row-aligned predictions and timestamps."""
    path = Path(path)
    if path.name != "validation_predictions.npz":
        raise PredictionArtifactError("Seuls les artefacts validation_predictions.npz sont autorisés")
    if str(model).upper() not in MODELS or int(horizon) not in HORIZONS:
        raise PredictionArtifactError("Modèle ou horizon inconnu")
    if not path.is_file():
        raise FileNotFoundError(f"Fichier de validation introuvable : {path}")

    with np.load(path, allow_pickle=False) as archive:
        missing = set(REQUIRED_KEYS).difference(archive.files)
        if missing:
            raise PredictionArtifactError(f"Champs absents : {', '.join(sorted(missing))}")
        arrays = {
            k: _as_numeric(archive[k], k, "int64" if k.endswith("_ms") else "float64")
            for k in REQUIRED_KEYS
        }

    n = len(arrays["y_true"])
    if n == 0 or any(len(value) != n for value in arrays.values()):
        raise PredictionArtifactError("Colonnes vides ou longueurs incohérentes")
    if (arrays["current_close"] <= 0).any() or (arrays["next_close"] <= 0).any():
        raise PredictionArtifactError("Les cours de clôture doivent être strictement positifs")

    # Confirm that y_true denotes log returns. Do NOT silently interpret them
    # as prices or simple returns. Small floating point differences are OK.
    reconstructed_y = np.log(arrays["next_close"] / arrays["current_close"])
    if not np.allclose(reconstructed_y, arrays["y_true"], rtol=0, atol=5e-6):
        raise PredictionArtifactError(
            "y_true ne correspond pas à ln(next_close/current_close). "
            "Vérifier la définition de la cible avant toute visualisation."
        )

    df = pd.DataFrame(arrays)
    for key in ("end_timestamp_ms", "target_timestamp_ms", "signal_available_at_ms", "target_available_at_ms"):
        dates = pd.to_datetime(df[key], unit="ms", utc=True, errors="coerce")
        if dates.isna().any():
            raise PredictionArtifactError(f"Dates invalides : {key}")
        df[key.removesuffix("_ms") + "_utc"] = dates

    df["model"] = str(model).upper()
    df["horizon_minutes"] = int(horizon)
    df["predicted_close"] = df["current_close"] * np.exp(df["y_pred"])
    if not np.isfinite(df["predicted_close"].to_numpy()).all():
        raise PredictionArtifactError("Prix reconstruits non finis")
    df["observed_log_return_pct"] = 100 * df["y_true"]
    df["predicted_log_return_pct"] = 100 * df["y_pred"]
    df["absolute_error_log_return"] = (df["y_true"] - df["y_pred"]).abs()
    return df


def evaluation_metrics(df: pd.DataFrame) -> dict:
    """Metrics calculated on identical validation labels for model and zero baseline."""
    true = df["y_true"].to_numpy(dtype=np.float64)
    pred = df["y_pred"].to_numpy(dtype=np.float64)
    if len(true) == 0:
        raise PredictionArtifactError("Aucune prédiction de validation")
    err = true - pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    baseline_mae = float(np.mean(np.abs(true)))
    baseline_rmse = float(np.sqrt(np.mean(true ** 2)))
    return {
        "n": len(true),
        "mae": mae,
        "rmse": rmse,
        "zero_mae": baseline_mae,
        "zero_rmse": baseline_rmse,
        "mae_improvement_pct": (100 * (baseline_mae - mae) / baseline_mae)
        if baseline_mae > 0 else None,
        "direction_accuracy_pct": float(100 * np.mean(np.sign(true) == np.sign(pred))),
    }


def temporal_audit(df: pd.DataFrame) -> dict:
    """Check structural timing; NOT a comprehensive absence-of-leakage proof."""
    end = df["end_timestamp_ms"].to_numpy(dtype=np.int64)
    target = df["target_timestamp_ms"].to_numpy(dtype=np.int64)
    sig = df["signal_available_at_ms"].to_numpy(dtype=np.int64)
    target_av = df["target_available_at_ms"].to_numpy(dtype=np.int64)
    return {
        "non_monotonic_end_timestamps": int(np.count_nonzero(np.diff(end) < 0)),
        "duplicate_end_timestamps": int(pd.Series(end).duplicated().sum()),
        "target_not_after_end": int(np.count_nonzero(target <= end)),
        "target_not_available_after_signal": int(np.count_nonzero(target_av <= sig)),
        "signal_before_end": int(np.count_nonzero(sig < end)),
        "target_available_before_timestamp": int(np.count_nonzero(target_av < target)),
        "unexpected_target_horizon": int(np.count_nonzero(target - end != int(df["horizon_minutes"].iloc[0]) * 60_000)),
    }


def metric_table(root: Path, horizon: int) -> pd.DataFrame:
    rows: list[dict] = []
    for model in MODELS:
        file = experiment_file(root, model, horizon)
        if not file.is_file():
            continue
        try:
            data = read_validation(file, model=model, horizon=horizon)
            metrics = evaluation_metrics(data)
            rows.append({"Modèle": model, "Horizon (min)": horizon,
                         "Observations": metrics["n"], "MAE": metrics["mae"],
                         "RMSE": metrics["rmse"], "MAE baseline zéro": metrics["zero_mae"],
                         "Amélioration MAE (%)": metrics["mae_improvement_pct"],
                         "Directional Accuracy (%)": metrics["direction_accuracy_pct"]})
        except (PredictionArtifactError, ValueError, OSError) as exc:
            rows.append({"Modèle": model, "Horizon (min)": horizon, "Erreur": str(exc)})
    return pd.DataFrame(rows)
