"""Unit tests for Phase 15 — synthetic Validation artifacts only."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from frontend.services.ai_model_lab_audit import (
    comparison_for_horizon, file_sha256, inventory, read_report, report_path,
    summary_export, temporal_check, top_residuals, validation_diagnostics,
)
from frontend.services.ai_prediction_engine import read_validation


def make_fixture(root, model="LSTM", horizon=60, *, corrupt=False):
    folder = root / f"{model.lower()}_{horizon}m_seed42"
    folder.mkdir(parents=True, exist_ok=True)
    n = 12
    end = np.arange(n, dtype=np.int64) * 300_000 + 1_669_827_300_000
    future = end + horizon * 60_000
    cur = np.linspace(100, 110, n)
    nxt = cur * np.exp(np.linspace(-.003, .004, n))
    true = np.log(nxt / cur)
    pred = np.zeros(n)
    if corrupt:
        pred[0] = np.inf
    np.savez(folder / "validation_predictions.npz",
             y_true=true.astype(np.float32), y_pred=pred.astype(np.float32),
             end_timestamp_ms=end, target_timestamp_ms=future,
             signal_available_at_ms=end + 300_000,
             target_available_at_ms=future + 300_000,
             current_close=cur, next_close=nxt)
    (folder / "report.json").write_text(json.dumps({"example": True}), encoding="utf-8")
    return folder


def test_inventory_always_12(tmp_path):
    assert len(inventory(tmp_path)) == 12
    assert (inventory(tmp_path)["NPZ Validation"] == "Absent").all()


def test_inventory_reports_presence(tmp_path):
    make_fixture(tmp_path, model="CNN1D", horizon=15)
    rows = inventory(tmp_path)
    row = rows[(rows["Modèle"] == "CNN1D") & (rows["Horizon (min)"] == 15)].iloc[0]
    assert row["NPZ Validation"] == "Présent"
    assert row["Rapport JSON"] == "Présent"


def test_report_path(tmp_path):
    assert report_path(tmp_path, "GRU", 60).name == "report.json"


def test_metrics_zero_baseline_same_rows(tmp_path):
    make_fixture(tmp_path)
    frame = read_validation(tmp_path / "lstm_60m_seed42" / "validation_predictions.npz",
                            model="LSTM", horizon=60)
    metrics = validation_diagnostics(frame)
    assert metrics["observations"] == 12
    assert metrics["mae"] == pytest.approx(metrics["zero_mae"])
    assert metrics["rmse"] == pytest.approx(metrics["zero_rmse"])
    assert metrics["mae_improvement_pct"] == pytest.approx(0)
    assert metrics["rmse_improvement_pct"] == pytest.approx(0)


def test_comparison_available_models_only(tmp_path):
    make_fixture(tmp_path, "GRU", 30)
    metrics, errors = comparison_for_horizon(tmp_path, 30)
    assert metrics["Modèle"].tolist() == ["GRU"]
    assert set(errors["Modèle"]) == {"LSTM", "CNN1D"}


def test_comparison_rejects_corrupt_npz(tmp_path):
    make_fixture(tmp_path, "LSTM", 5, corrupt=True)
    metrics, errors = comparison_for_horizon(tmp_path, 5)
    assert metrics.empty
    assert "LSTM" in errors["Modèle"].tolist()


def test_comparison_bad_horizon(tmp_path):
    with pytest.raises(ValueError):
        comparison_for_horizon(tmp_path, 999)


def test_temporal_metadata_checks(tmp_path):
    make_fixture(tmp_path)
    path = tmp_path / "lstm_60m_seed42" / "validation_predictions.npz"
    frame = read_validation(path, model="LSTM", horizon=60)
    checks = temporal_check(frame)
    assert not checks.empty
    assert checks["Anomalies"].eq(0).all()


def test_largest_residuals_sorted(tmp_path):
    make_fixture(tmp_path)
    path = tmp_path / "lstm_60m_seed42" / "validation_predictions.npz"
    frame = read_validation(path, model="LSTM", horizon=60)
    first_three = top_residuals(frame, 3)
    assert len(first_three) == 3
    assert first_three["Erreur absolue"].is_monotonic_decreasing


def test_top_residuals_invalid_limit(tmp_path):
    make_fixture(tmp_path)
    path = tmp_path / "lstm_60m_seed42" / "validation_predictions.npz"
    with pytest.raises(ValueError):
        top_residuals(read_validation(path, model="LSTM", horizon=60), 0)


def test_file_sha256(tmp_path):
    p = tmp_path / "sample.txt"
    p.write_bytes(b"aurum")
    assert file_sha256(p) == hashlib.sha256(b"aurum").hexdigest()


def test_read_report(tmp_path):
    folder = make_fixture(tmp_path)
    assert read_report(folder / "report.json")["example"] is True


def test_reject_non_object_json(tmp_path):
    p = tmp_path / "r.json"
    p.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        read_report(p)


def test_json_export_does_not_write(tmp_path):
    make_fixture(tmp_path)
    p = tmp_path / "lstm_60m_seed42" / "validation_predictions.npz"
    frame = read_validation(p, model="LSTM", horizon=60)
    before = {x: x.stat().st_mtime_ns for x in tmp_path.rglob("*") if x.is_file()}
    content = summary_export("LSTM", 60, validation_diagnostics(frame),
                             temporal_check(frame), p)
    data = json.loads(content)
    assert data["no_test_metrics"] is True
    assert data["source_sha256"] == file_sha256(p)
    assert {x: x.stat().st_mtime_ns for x in before} == before


def test_missing_report_is_preserved(tmp_path):
    folder = make_fixture(tmp_path)
    (folder / "report.json").unlink()
    row = inventory(tmp_path).query('`Modèle` == "LSTM" and `Horizon (min)` == 60').iloc[0]
    assert row["Rapport JSON"] == "Absent"
    assert row["NPZ Validation"] == "Présent"
