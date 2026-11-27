"""Phase 18: real file path controls, missing sources, exports, no writes."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Import via module path so this file can run when installed into AURUM-AI.
sys.path.insert(0, str(PROJECT_ROOT))
from frontend.services.research_reports import (
    HORIZONS, MODELS, FILES, allowed_sources, csv_profile, flatten_scalars,
    html_report_bytes, inventory_csv_bytes, json_report_bytes,
    pdf_report_bytes, read_json, redact, report_snapshot, run_path,
    safe_source, source_inventory,
)


@pytest.fixture()
def fake_project(tmp_path: Path):
    for h in HORIZONS:
        for m in MODELS:
            dest = tmp_path / run_path(m, h)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps({"model": m, "horizon_minutes": h,
                                        "validation": {"mae": 0.02},
                                        "access_token": "NEVER-EXPORT"}), encoding="utf-8")
    for key, relative in FILES.items():
        dest = tmp_path / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if relative.suffix == ".json":
            if key == "ledger":
                dest.write_text('{"blocks":[]}', encoding="utf-8")
            else:
                dest.write_text(json.dumps({"capital": {"initial": 10000, "final": 9900},
                                            "password": "NEVER-EXPORT"}), encoding="utf-8")
        else:
            with dest.open("w", encoding="utf-8", newline="") as handle:
                w = csv.writer(handle)
                w.writerow(["action", "quantity", "private_key"])
                w.writerow(["BUY", "2", "NEVER-EXPORT"])
                w.writerow(["HOLD", "0", "NEVER-EXPORT"])
    return tmp_path


def all_file_bytes(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_whitelist_size():
    assert len(allowed_sources()) == 17
    assert len(set(allowed_sources().values())) == 17


def test_all_paths_relative_and_within_scopes():
    for path in allowed_sources().values():
        assert not path.is_absolute()
        assert ".." not in path.parts


def test_run_paths_validate():
    with pytest.raises(ValueError):
        run_path("TRANSFORMER", 60)
    with pytest.raises(ValueError):
        run_path("LSTM", -1)
    with pytest.raises(ValueError):
        run_path("LSTM", True)


def test_no_user_selected_path(fake_project):
    with pytest.raises(ValueError):
        safe_source(fake_project, "../../auth.py")


def test_all_expected_sources_resolve(fake_project):
    for k in allowed_sources():
        assert safe_source(fake_project, k).is_file()


def test_missing_source_kept_as_missing(fake_project):
    (fake_project / FILES["historical_summary"]).unlink()
    row = next(x for x in source_inventory(fake_project) if x["key"] == "historical_summary")
    assert row["status"] == "absent"
    report = report_snapshot(fake_project)
    assert report["historical_summary"] is None
    assert report["historical_summary_error"] is not None


def test_reject_file_symlink(fake_project, tmp_path):
    dest = fake_project / run_path("LSTM", 5)
    dest.unlink()
    external = tmp_path.parent / (tmp_path.name + "_outside")
    external.write_text('{"unsafe": true}', encoding="utf-8")
    try:
        dest.symlink_to(external)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink not supported without admin privileges")
    with pytest.raises(ValueError):
        safe_source(fake_project, "validation_lstm_5m")


def test_reject_symlink_ancestor(fake_project, tmp_path):
    source_dir = fake_project / "artifacts" / "experiments"
    new_dir = tmp_path.parent / (tmp_path.name + "_experiments")
    source_dir.rename(new_dir)
    try:
        source_dir.symlink_to(new_dir, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Directory symlinks not supported")
    with pytest.raises(ValueError):
        safe_source(fake_project, "validation_gru_15m")


def test_redact_nested_credentials():
    obj = redact({"meta": {"password": "NEVER-EXPORT", "api_key": "NEVER-EXPORT"}})
    assert "NEVER-EXPORT" not in json.dumps(obj)


def test_read_model_report_redacts(fake_project):
    meta = read_json(fake_project, "validation_lstm_5m")
    assert meta["access_token"] == "[REDACTED]"
    assert meta["model"] == "LSTM"


def test_reject_nan(fake_project):
    (fake_project / FILES["historical_summary"]).write_text('{"bad": NaN}', encoding="utf-8")
    with pytest.raises(ValueError):
        read_json(fake_project, "historical_summary")


def test_reject_non_object_json(fake_project):
    (fake_project / FILES["historical_summary"]).write_text('[1,2,3]', encoding="utf-8")
    with pytest.raises(ValueError):
        read_json(fake_project, "historical_summary")


def test_reject_oversized_json(fake_project):
    with (fake_project / FILES["historical_summary"]).open("w") as handle:
        handle.write(" " * (4 * 1024 * 1024 + 1))
    with pytest.raises(ValueError):
        read_json(fake_project, "historical_summary")


def test_csv_profiles_count_and_hide_secret(fake_project):
    profile = csv_profile(fake_project, "transactions_60m")
    assert profile["rows"] == 2
    assert "private_key" not in profile["columns"]
    assert "NEVER-EXPORT" not in json.dumps(profile)


def test_disallow_unknown_csv(fake_project):
    with pytest.raises(ValueError):
        csv_profile(fake_project, "validation_lstm_5m")


def test_snapshot_valid_counts(fake_project):
    report = report_snapshot(fake_project)
    assert report["model_reports_valid"] == 12
    assert report["historical_transactions"]["rows"] == 2
    assert report["historical_decisions"]["rows"] == 2
    assert len(report["model_runs"]) == 12
    assert report["test_generalization_not_established"] is True


def test_snapshot_does_not_modify_sources(fake_project):
    before = all_file_bytes(fake_project)
    report = report_snapshot(fake_project, with_hashes=True)
    assert len(report["sources"]) == 17
    assert all_file_bytes(fake_project) == before


def test_export_json_valid_utf8_and_sanitized(fake_project):
    raw = json_report_bytes(report_snapshot(fake_project))
    obj = json.loads(raw)
    assert obj["model_reports_valid"] == 12
    assert b"NEVER-EXPORT" not in raw


def test_export_csv(fake_project):
    raw = inventory_csv_bytes(report_snapshot(fake_project)).decode("utf-8-sig")
    assert len(list(csv.DictReader(raw.splitlines()))) == 17


def test_html_injection_is_escaped(fake_project):
    (fake_project / FILES["historical_summary"]).write_text(
        json.dumps({"label": '<script>alert("BAD")</script>'}), encoding="utf-8")
    doc = html_report_bytes(report_snapshot(fake_project)).decode("utf-8")
    assert "&lt;script&gt;" in doc
    assert '<script>alert("BAD")</script>' not in doc
    assert "NEVER-EXPORT" not in doc
    assert "Imprimer" not in doc or "source" in doc


def test_empty_project_without_invented_numbers(tmp_path):
    snap = report_snapshot(tmp_path)
    assert snap["model_reports_valid"] == 0
    assert snap["historical_decisions"] is None
    assert snap["historical_transactions"] is None
    assert len(snap["sources"]) == 17


def test_pdf_signature_and_sources_untouched(fake_project):
    pytest.importorskip("reportlab")
    before = all_file_bytes(fake_project)
    raw = pdf_report_bytes(report_snapshot(fake_project))
    assert raw.startswith(b"%PDF-")
    assert len(raw) > 2000
    assert all_file_bytes(fake_project) == before
