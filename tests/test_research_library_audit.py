"""Phase 16 pure-service tests: no Streamlit or external data required."""
from __future__ import annotations
import hashlib
from pathlib import Path
import pytest
from frontend.services.research_library import (
    LibraryItem, catalog_summary, filter_library, format_size,
    list_library, resolve_library_item, sha256_file,
)


def _fixture(tmp_path: Path) -> Path:
    files = {
        "README.md": "AURUM-AI\n",
        "reports/backtesting/report.json": '{"mae":0.01}',
        "artifacts/experiments/nyse_v3/lstm_60m_seed42/report.json": '{"model":"LSTM"}',
        "artifacts/experiments/nyse_v3/lstm_60m_seed42/validation_predictions.npz": "binary-stub",
        "docs/protocol.md": "audit",
        "documentation/notes.txt": "notes",
        "data/03_features/xom_features_nyse_v3.parquet": "binary-stub",
        "data/04_windows/nyse_v3/train_60m_windows.npz": "binary-stub",
        "data/05_model_data/index.json": "{}",
        "data/01_RAW/original.csv": "must-not-be-indexed",
        "frontend/auth/email_service.py": "password=SECRET",
        "docs/service_account.json": "do-not-leak",
        "docs/my_api_key.txt": "do-not-leak",
        "docs/.env": "SECRET",
        "reports/cache/__pycache__/hidden.csv": "skip",
        "reports/readme.py": "exclude source files",
    }
    for rel, content in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return tmp_path


def test_includes_only_research_scopes(tmp_path):
    rows = list_library(_fixture(tmp_path))
    paths = {r.relative_path for r in rows}
    assert "README.md" in paths
    assert "reports/backtesting/report.json" in paths
    assert "artifacts/experiments/nyse_v3/lstm_60m_seed42/report.json" in paths
    assert "docs/protocol.md" in paths
    assert "data/04_windows/nyse_v3/train_60m_windows.npz" in paths
    assert "data/01_RAW/original.csv" not in paths
    assert "frontend/auth/email_service.py" not in paths


def test_excludes_secrets_and_source(tmp_path):
    rows = list_library(_fixture(tmp_path))
    paths = {r.relative_path for r in rows}
    assert not any("service_account" in p for p in paths)
    assert not any("api_key" in p for p in paths)
    assert not any(".env" in p for p in paths)
    assert not any("__pycache__" in p for p in paths)
    assert not any(p.endswith(".py") for p in paths)


def test_catalog_is_deterministic_and_read_only(tmp_path):
    root = _fixture(tmp_path)
    before = {p.relative_to(root).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
              for p in root.rglob("*") if p.is_file()}
    one, two = list_library(root), list_library(root)
    after = {p.relative_to(root).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
             for p in root.rglob("*") if p.is_file()}
    assert one == two
    assert before == after


def test_search_and_category_filters(tmp_path):
    rows = list_library(_fixture(tmp_path))
    assert len(filter_library(rows, "LSTM")) == 2
    assert len(filter_library(rows, category="Features")) == 1
    assert len(filter_library(rows, extension=".parquet")) == 1
    assert filter_library(rows, search="unknown") == []


def test_summary(tmp_path):
    rows = list_library(_fixture(tmp_path))
    summary = catalog_summary(rows)
    assert summary["Expériences IA"] == 2
    assert summary["Rapports"] == 1


def test_resolve_known_item(tmp_path):
    root = _fixture(tmp_path)
    row = next(r for r in list_library(root) if r.name == "report.json")
    assert resolve_library_item(root, row).is_file()


def test_reject_traversal(tmp_path):
    root = _fixture(tmp_path)
    fake = LibraryItem("Rapports", "../outside.json", "outside.json", ".json", 1, "")
    with pytest.raises(ValueError):
        resolve_library_item(root, fake)


def test_reject_scope_forgery(tmp_path):
    root = _fixture(tmp_path)
    row = next(r for r in list_library(root) if r.category == "Rapports")
    forged = LibraryItem("Expériences IA", row.relative_path, row.name, row.extension,
                         row.size_bytes, row.modified_at_utc)
    with pytest.raises(ValueError):
        resolve_library_item(root, forged)


def test_reject_removed_file(tmp_path):
    root = _fixture(tmp_path)
    row = next(r for r in list_library(root) if r.category == "Rapports")
    (root / row.relative_path).unlink()
    with pytest.raises(ValueError):
        resolve_library_item(root, row)


def test_no_symlink_escape(tmp_path):
    root = _fixture(tmp_path / "project")
    outside = tmp_path / "secret.json"
    outside.write_text('{"secret":true}')
    link = root / "reports" / "linked.json"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unavailable")
    assert "reports/linked.json" not in {r.relative_path for r in list_library(root)}


def test_sha256(tmp_path):
    root = _fixture(tmp_path)
    path = root / "README.md"
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_limit_and_missing_directory(tmp_path):
    root = _fixture(tmp_path)
    assert len(list_library(root, max_files=2)) == 2
    assert list_library(tmp_path / "absent") == []
    assert list_library(root, max_files=0) == []


def test_format_size():
    assert format_size(12) == "12 o"
    assert format_size(2048) == "2.0 Ko"
