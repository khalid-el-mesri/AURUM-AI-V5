from pathlib import Path
import importlib.util
import sqlite3

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'tools/windows_desktop/diagnose_auth_storage.py'
spec = importlib.util.spec_from_file_location('aurum_auth_diagnose', SOURCE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_module_root(tmp_path):
    script = tmp_path / 'tools/windows_desktop/diagnose_auth_storage.py'
    script.parent.mkdir(parents=True)
    script.touch()
    assert mod.locate_root(script) == tmp_path


def test_detects_sqlite_and_json(tmp_path):
    d = tmp_path / 'frontend/auth'
    d.mkdir(parents=True)
    db = d / 'aurum.sqlite'
    with sqlite3.connect(db) as c:
        c.execute('create table users(email text, password_hash text, email_verified integer)')
        c.execute('insert into users values (?, ?, ?)', ('test@example.invalid','DO_NOT_DISPLAY',1))
    (d / 'accounts.json').write_text('{}')
    records = mod.scan_storage(tmp_path, {})
    assert {p.name for _,_,p in records} == {'aurum.sqlite','accounts.json'}
    assert mod.is_sqlite(db)
    assert mod.query_email_presence(db, 'test@example.invalid') == [('users','email_verified=1')]
    assert mod.query_email_presence(db, 'other@example.invalid') == []


def test_report_does_not_show_credentials(tmp_path):
    d=tmp_path / 'frontend/auth'
    d.mkdir(parents=True)
    (d / 'database.py').write_text('import sqlite3\nDB_PATH = "private.db"\n')
    db=d/'private.db'
    with sqlite3.connect(db) as c:
        c.execute('create table accounts(email text, password_hash text, verified integer)')
        c.execute('insert into accounts values (?, ?, ?)', ('private@example.invalid', 'SECRET-HASH', 0))
    out='\n'.join(mod.produce_report(tmp_path, {}, True, 'private@example.invalid'))
    assert 'Compte: présent' in out
    assert 'verified=0' in out
    assert 'private@example.invalid' not in out
    assert 'SECRET-HASH' not in out


def test_source_tags_without_source_content(tmp_path):
    d=tmp_path/'frontend/auth'; d.mkdir(parents=True)
    (d/'database.py').write_text('from pathlib import Path\nDB_PATH=Path(__file__).parent/"aurum.sqlite"\n')
    r=mod.scan_source(tmp_path)
    assert 'file-relative' in r[0][1]
    assert 'db-path' in r[0][1]
    assert 'aurum.sqlite' not in str(r)


def test_no_creation_of_missing_directories(tmp_path):
    out=mod.produce_report(tmp_path, {})
    assert any('Aucun fichier' in line for line in out)
    assert list(tmp_path.iterdir()) == []


def test_no_password_hash_query():
    source=SOURCE.read_text()
    assert 'SELECT "password_hash"' not in source
    assert "SELECT *" not in source
    assert 'update accounts' not in source.lower()


def test_tagged_bundled_file(tmp_path):
    loc=tmp_path/'dist/AURUM-AI V5/_internal/frontend/auth'
    loc.mkdir(parents=True)
    with sqlite3.connect(loc/'custom.sqlite') as c:
        c.execute('create table users(email text)')
    report='\n'.join(mod.produce_report(tmp_path, {}))
    assert '[bundled-auth]' in report
    assert 'ATTENTION' in report
