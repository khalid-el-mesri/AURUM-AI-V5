"""Phase 17: 15 isolated tests with synthetic local artifacts, no Streamlit needed."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pytest

from frontend.services.research_experiments import (
    MODELS,HORIZONS,EXPERIMENTS_DIR,artifact_path,compare_report_metadata,
    list_experiments,manifest,manifest_json_bytes,read_report,redact,
    registry_csv_bytes,scalar_metadata,sha256,
)


def fixture_run(root: Path, model: str = "LSTM", horizon: int = 60,
                *, report: bool = True, npz: bool = True) -> Path:
    folder = root / EXPERIMENTS_DIR / f"{model.lower()}_{horizon}m_seed42"
    folder.mkdir(parents=True, exist_ok=True)
    if report:
        (folder / "report.json").write_text(json.dumps({"model":model,"seed":42,"metrics":{"validation_mae":0.01},"private_key":"DO-NOT-EXPOSE"}), encoding="utf-8")
    if npz:
        (folder / "validation_predictions.npz").write_bytes(b"synthetic-only")
    return folder


def test_expected_12_missing(tmp_path):
    records = list_experiments(tmp_path)
    assert len(records) == len(MODELS)*len(HORIZONS) == 12
    assert all(r.status == "Absent" for r in records)


def test_complete_run(tmp_path):
    fixture_run(tmp_path)
    r = next(r for r in list_experiments(tmp_path) if r.run_id == 'lstm_60m_seed42')
    assert r.status == "Complet"
    assert r.validation_present and r.report_present
    assert r.validation_size_bytes == len(b"synthetic-only")


def test_report_absent(tmp_path):
    fixture_run(tmp_path, report=False)
    r = next(r for r in list_experiments(tmp_path) if r.run_id == 'lstm_60m_seed42')
    assert r.status == "Rapport absent"


def test_predictions_absent(tmp_path):
    fixture_run(tmp_path, npz=False)
    r = next(r for r in list_experiments(tmp_path) if r.run_id == 'lstm_60m_seed42')
    assert r.status == "Prédictions absentes"


def test_fixed_known_paths(tmp_path):
    fixture_run(tmp_path, model="GRU", horizon=15)
    assert artifact_path(tmp_path, "GRU", 15, "report").name == "report.json"
    with pytest.raises(ValueError):
        artifact_path(tmp_path, "../../etc", 15, "report")
    with pytest.raises(ValueError):
        artifact_path(tmp_path, "GRU", 999, "report")
    with pytest.raises(ValueError):
        artifact_path(tmp_path, "GRU", 15, "arbitrary")


def test_report_redaction(tmp_path):
    fixture_run(tmp_path)
    data = read_report(tmp_path, "LSTM", 60)
    assert data["private_key"] == "[REDACTED]"
    assert data["metrics"]["validation_mae"] == .01


def test_no_non_object_report(tmp_path):
    folder = fixture_run(tmp_path)
    (folder / "report.json").write_text("[1,2,3]")
    with pytest.raises(ValueError):
        read_report(tmp_path, "LSTM", 60)


def test_reject_nan_and_large_report(tmp_path):
    folder = fixture_run(tmp_path)
    (folder / "report.json").write_text('{"bad":NaN}')
    with pytest.raises(ValueError):
        read_report(tmp_path, "LSTM", 60)
    (folder / "report.json").write_bytes(b' ' * (2 * 1024 * 1024 + 1))
    with pytest.raises(ValueError):
        read_report(tmp_path, "LSTM", 60)


def test_compare_same_and_different_values():
    rows = compare_report_metadata({"a":1,"nested":{"x":3}}, {"a":2,"nested":{"x":3}})
    assert next(r for r in rows if r['Champ JSON']=='a')['Comparaison'] == 'différent'
    assert next(r for r in rows if r['Champ JSON']=='nested.x')['Comparaison'] == 'identique'


def test_scalar_metadata_and_secret_masking():
    meta = scalar_metadata({"x": {"a": 2}, "access_token": "secret-value", "l": [1,2,3]})
    assert meta["x.a"] == '2'
    assert 'secret-value' not in repr(meta)
    assert "l" not in meta


def test_sha256_actual_bytes(tmp_path):
    folder = fixture_run(tmp_path)
    assert sha256(folder / 'validation_predictions.npz') == hashlib.sha256(b'synthetic-only').hexdigest()


def test_manifest_is_read_only_and_json(tmp_path):
    fixture_run(tmp_path)
    before = {x: (x.read_bytes(),x.stat().st_mtime_ns) for x in tmp_path.rglob('*') if x.is_file()}
    rows = list_experiments(tmp_path)
    data = json.loads(manifest_json_bytes(tmp_path, rows))
    assert data['scope'] == 'historical_validation_only'
    assert data['no_test_results'] is True
    assert len(data['records']) == 12
    assert 'sha256' not in data['records'][0]
    after = {x: (x.read_bytes(),x.stat().st_mtime_ns) for x in tmp_path.rglob('*') if x.is_file()}
    assert before == after


def test_manifest_with_hashes_only_explicit(tmp_path):
    fixture_run(tmp_path)
    rows = [r for r in list_experiments(tmp_path) if r.status == "Complet"]
    m = manifest(tmp_path, rows, with_hashes=True)
    assert m['records'][0]['sha256']['report'] == sha256(artifact_path(tmp_path,'LSTM',60,'report'))
    assert m['records'][0]['sha256']['validation'] == sha256(artifact_path(tmp_path,'LSTM',60,'validation'))


def test_export_csv_no_data_contents(tmp_path):
    fixture_run(tmp_path)
    content = registry_csv_bytes(list_experiments(tmp_path)).decode('utf-8-sig')
    assert 'lstm_60m_seed42' in content
    assert 'synthetic-only' not in content
    assert 'DO-NOT-EXPOSE' not in content


def test_refuse_symlink_report(tmp_path):
    folder = fixture_run(tmp_path)
    outside = tmp_path / 'outside.json'
    outside.write_text('{"danger":true}')
    (folder / 'report.json').unlink()
    try:
        (folder / 'report.json').symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip('Windows permissions do not permit creation of symlinks')
    rows = list_experiments(tmp_path)
    run = next(r for r in rows if r.run_id == 'lstm_60m_seed42')
    assert not run.report_present
    with pytest.raises(FileNotFoundError):
        artifact_path(tmp_path, 'LSTM', 60, 'report')
