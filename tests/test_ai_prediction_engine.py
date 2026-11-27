from pathlib import Path
import numpy as np
import pytest

from frontend.services.ai_prediction_engine import (
    PredictionArtifactError, experiment_file, available_experiments,
    read_validation, temporal_audit, evaluation_metrics, metric_table,
)


def fixture_npz(tmp_path: Path, model="LSTM", horizon=5, **overrides):
    path = experiment_file(tmp_path, model, horizon)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 4
    end = 1_700_000_000_000 + np.arange(n) * 300_000
    price = np.array([100., 101., 102., 99.])
    future = np.array([100.5, 100.8, 101.9, 100.2])
    arrays = {
        "y_true": np.log(future / price).astype("float32"),
        "y_pred": np.array([0.001, -0.0004, 0.0003, 0.0011], dtype="float32"),
        "end_timestamp_ms": end,
        "target_timestamp_ms": end + horizon * 60000,
        "signal_available_at_ms": end + 300_000,
        "target_available_at_ms": end + horizon * 60000 + 300_000,
        "current_close": price,
        "next_close": future,
    }
    arrays.update(overrides)
    np.savez_compressed(path, **arrays)
    return path


def test_whitelist(tmp_path):
    with pytest.raises(PredictionArtifactError):
        experiment_file(tmp_path, "HACK", 5)
    with pytest.raises(PredictionArtifactError):
        experiment_file(tmp_path, "LSTM", 500)


def test_inventory(tmp_path):
    fixture_npz(tmp_path)
    assert available_experiments(tmp_path) == [("LSTM", 5)]


def test_valid_data_and_prices(tmp_path):
    path = fixture_npz(tmp_path)
    df = read_validation(path, model="LSTM", horizon=5)
    assert len(df) == 4
    assert df["predicted_close"].iloc[0] == pytest.approx(100 * np.exp(0.001))
    assert df["y_true"].iloc[0] == pytest.approx(np.log(100.5 / 100), abs=1e-7)


def test_metrics_and_zero_baseline(tmp_path):
    df = read_validation(fixture_npz(tmp_path), model="LSTM", horizon=5)
    met = evaluation_metrics(df)
    assert met["n"] == 4
    assert met["zero_mae"] == pytest.approx(float(np.abs(df["y_true"]).mean()))
    assert met["rmse"] >= met["mae"]


def test_time_checks_pass(tmp_path):
    df = read_validation(fixture_npz(tmp_path), model="LSTM", horizon=5)
    assert all(v == 0 for v in temporal_audit(df).values())


def test_time_violation_reports_count(tmp_path):
    path = fixture_npz(tmp_path)
    df = read_validation(path, model="LSTM", horizon=5)
    df.loc[1, "target_available_at_ms"] = df.loc[1, "signal_available_at_ms"]
    assert temporal_audit(df)["target_not_available_after_signal"] == 1


def test_missing_key_rejected(tmp_path):
    path = fixture_npz(tmp_path)
    np.savez_compressed(path, y_true=np.array([0.1]))
    with pytest.raises(PredictionArtifactError, match="Champs absents"):
        read_validation(path, model="LSTM", horizon=5)


def test_nan_rejected(tmp_path):
    path = fixture_npz(tmp_path, y_pred=np.array([np.nan, 0., 0., 0.]))
    with pytest.raises(PredictionArtifactError, match="non finies"):
        read_validation(path, model="LSTM", horizon=5)


def test_lengths_rejected(tmp_path):
    path = fixture_npz(tmp_path, y_pred=np.array([0., 0.]))
    with pytest.raises(PredictionArtifactError, match="longueurs incohérentes"):
        read_validation(path, model="LSTM", horizon=5)


def test_wrong_target_semantics_rejected(tmp_path):
    path = fixture_npz(tmp_path, y_true=np.ones(4, dtype=np.float32))
    with pytest.raises(PredictionArtifactError, match="y_true ne correspond pas"):
        read_validation(path, model="LSTM", horizon=5)


def test_no_test_predictions_allowed(tmp_path):
    path = tmp_path / "test_predictions.npz"
    np.savez_compressed(path, **{"y_true": np.array([0.])})
    with pytest.raises(PredictionArtifactError, match="Seuls"):
        read_validation(path, model="LSTM", horizon=5)


def test_comparison(tmp_path):
    fixture_npz(tmp_path, model="LSTM", horizon=15)
    fixture_npz(tmp_path, model="GRU", horizon=15)
    table = metric_table(tmp_path, 15)
    assert set(table["Modèle"].tolist()) == {"LSTM", "GRU"}
    assert len(table) == 2


def test_nonpositive_closes_rejected(tmp_path):
    path = fixture_npz(tmp_path, current_close=np.array([0., 101., 102., 99.]))
    with pytest.raises(PredictionArtifactError, match="strictement positifs"):
        read_validation(path, model="LSTM", horizon=5)
