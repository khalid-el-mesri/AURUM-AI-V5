"""Non-destructive tests for stable account storage and SQLite migration."""
import importlib.util
import os
from pathlib import Path
import sqlite3
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(rel):
    spec = importlib.util.spec_from_file_location('test_module_' + Path(rel).stem, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def modules():
    return load('frontend/auth/database.py'), load('tools/windows_desktop/migrate_auth_storage.py')


@pytest.fixture
def source_db(tmp_path):
    src = tmp_path / 'project' / 'runtime' / 'auth' / 'users.sqlite3'
    src.parent.mkdir(parents=True)
    with sqlite3.connect(src) as db:
        db.executescript('''
        CREATE TABLE users(id INTEGER PRIMARY KEY, email TEXT UNIQUE, password_hash TEXT, email_verified INTEGER NOT NULL);
        CREATE TABLE email_verification(user_id INTEGER,code_hash TEXT,expires_at INTEGER);
        INSERT INTO users VALUES(1,'private@example.com','argon2-test-hash',1);
        INSERT INTO email_verification VALUES(1,'secret-code',123);
        ''')
    return src


def test_resolve_localappdata(modules, tmp_path):
    database, _ = modules
    assert database.resolve_db_path({'LOCALAPPDATA': str(tmp_path)}) == tmp_path / 'AURUM-AI/auth/users.sqlite3'


def test_resolve_xdg(modules, tmp_path):
    database, _ = modules
    assert database.resolve_db_path({'XDG_DATA_HOME': str(tmp_path)}) == tmp_path / 'AURUM-AI/auth/users.sqlite3'


def test_resolve_home(modules, tmp_path):
    database, _ = modules
    assert database.resolve_db_path({}, home=tmp_path) == tmp_path / '.local/share/AURUM-AI/auth/users.sqlite3'


def test_target_not_inside_internal(modules, tmp_path):
    database, _ = modules
    assert '_internal' not in str(database.resolve_db_path({'LOCALAPPDATA': str(tmp_path)}))


def test_paths_exact(modules, tmp_path):
    _, migration = modules
    where = migration.paths(tmp_path / 'project', {'LOCALAPPDATA': str(tmp_path / 'local')})
    assert where['project'] == tmp_path/'project/runtime/auth/users.sqlite3'
    assert where['packaged'] == tmp_path/'project/dist/AURUM-AI V5/_internal/runtime/auth/users.sqlite3'
    assert where['persistent'] == tmp_path/'local/AURUM-AI/auth/users.sqlite3'


def test_missing_base_has_no_side_effect(modules, tmp_path):
    _, migration = modules
    missing = tmp_path/'absent.sqlite3'
    assert migration.inspect_db(missing)['exists'] is False
    assert not missing.exists()


def test_inspect_no_secrets(modules, source_db):
    _, migration = modules
    details = migration.inspect_db(source_db)
    assert details['users'] == 1 and details['verified'] == 1
    assert 'private@example.com' not in repr(details)
    assert 'argon2-test-hash' not in repr(details)


def test_inspect_invalid_file(modules, tmp_path):
    _, migration = modules
    p = tmp_path/'bad.sqlite3'
    p.write_text('not sqlite')
    assert migration.inspect_db(p) == {'exists': True, 'valid_sqlite': False}


def test_migrate_preserves_all_tables(modules, source_db, tmp_path):
    _, migration = modules
    dst = tmp_path/'local/AURUM-AI/auth/users.sqlite3'
    result = migration.migrate(source_db, dst)
    assert '1 compte(s)' in result
    with sqlite3.connect(dst) as db:
        assert db.execute('SELECT password_hash FROM users').fetchone()[0] == 'argon2-test-hash'
        assert db.execute('SELECT code_hash FROM email_verification').fetchone()[0] == 'secret-code'
    assert source_db.exists()


def test_never_overwrites_existing_db(modules, source_db, tmp_path):
    _, migration = modules
    dst = tmp_path/'existing.sqlite3'
    dst.write_bytes(b'existing destination')
    with pytest.raises(FileExistsError):
        migration.migrate(source_db, dst)
    assert dst.read_bytes() == b'existing destination'


def test_rejects_missing_source(modules, tmp_path):
    _, migration = modules
    with pytest.raises(RuntimeError):
        migration.migrate(tmp_path/'not-here.sqlite3', tmp_path/'target.sqlite3')


def test_rejects_source_without_users_table(modules, tmp_path):
    _, migration = modules
    source = tmp_path/'empty.sqlite3'
    with sqlite3.connect(source) as db:
        db.execute('CREATE TABLE notes(x TEXT)')
    with pytest.raises(RuntimeError):
        migration.migrate(source, tmp_path/'target.sqlite3')


def test_no_bundled_users_db(modules):
    database, _ = modules
    assert 'runtime' not in str(database.DB_PATH) or 'LOCALAPPDATA' not in os.environ
