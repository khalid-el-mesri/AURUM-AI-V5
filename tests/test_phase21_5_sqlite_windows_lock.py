"""Regression: close SQLite file handles explicitly before unlink on Windows.

SQLite's ``with sqlite3.connect(...)`` does NOT close the connection.
The original implementation therefore left a temporary database open while
trying to delete it, triggering WinError 32 on Windows.
"""
from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'tools/windows_desktop/migrate_auth_storage.py'


def load_script():
    spec = importlib.util.spec_from_file_location('phase21_5_migration', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fixture_db(tmp_path):
    source = tmp_path / 'original' / 'users.sqlite3'
    source.parent.mkdir(parents=True)
    with sqlite3.connect(source) as db:
        db.executescript('''
            CREATE TABLE users(id INTEGER PRIMARY KEY, email TEXT UNIQUE,
                               password_hash TEXT NOT NULL, email_verified INTEGER);
            CREATE TABLE email_verification(user_id INTEGER, code_hash TEXT);
            INSERT INTO users VALUES(1, 'not-shared@example.invalid', 'protected-hash', 1);
            INSERT INTO email_verification VALUES(1, 'protected-code');
        ''')
    target = tmp_path / 'persistent' / 'users.sqlite3'
    return source, target


def test_inspection_explicitly_closes_sqlite_handle(fixture_db, monkeypatch):
    migration = load_script()
    real_connect = sqlite3.connect
    opened = []

    class TrackedConnection(sqlite3.Connection):
        was_explicitly_closed = False

        def close(self):
            self.was_explicitly_closed = True
            super().close()

    def tracked_connect(*args, **kwargs):
        kwargs['factory'] = TrackedConnection
        connection = real_connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(migration.sqlite3, 'connect', tracked_connect)
    details = migration.inspect_db(fixture_db[0])
    assert details['users'] == 1
    assert len(opened) == 1
    assert all(c.was_explicitly_closed for c in opened)


def test_migration_closes_all_handles_before_temp_cleanup(fixture_db, monkeypatch):
    migration = load_script()
    real_connect = sqlite3.connect
    opened = []

    class TrackedConnection(sqlite3.Connection):
        was_explicitly_closed = False

        def close(self):
            self.was_explicitly_closed = True
            super().close()

    def tracked_connect(*args, **kwargs):
        kwargs['factory'] = TrackedConnection
        connection = real_connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(migration.sqlite3, 'connect', tracked_connect)
    source, target = fixture_db
    migration.migrate(source, target)

    assert target.exists() and source.exists()
    assert len(opened) >= 4  # initial inspection, source, backup target, restored inspection
    assert all(c.was_explicitly_closed for c in opened)
    assert not list(target.parent.glob('aurum_auth_migration_*.sqlite3'))
    with real_connect(target) as db:
        assert db.execute('SELECT password_hash FROM users').fetchone()[0] == 'protected-hash'
        assert db.execute('SELECT code_hash FROM email_verification').fetchone()[0] == 'protected-code'


def test_migration_refuses_overwrite_and_keeps_old_base(fixture_db):
    migration = load_script()
    source, target = fixture_db
    migration.migrate(source, target)
    with pytest.raises(FileExistsError):
        migration.migrate(source, target)
    assert source.is_file() and target.is_file()
    assert not list(target.parent.glob('aurum_auth_migration_*.sqlite3'))
