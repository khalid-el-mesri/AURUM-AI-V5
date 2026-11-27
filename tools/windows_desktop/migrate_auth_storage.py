"""Inspect and optionally COPY legacy AURUM-AI account database to stable storage.

Default --check is strictly read-only: does not create directories/databases.
--apply NEVER overwrites an existing destination. Source accounts and hashes
are preserved via SQLite's online backup API; no secrets are printed.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import os
from pathlib import Path
import sqlite3
import tempfile
from urllib.parse import quote


def paths(project: Path, env: dict[str, str]) -> dict[str, Path]:
    local = env.get('LOCALAPPDATA')
    if not local:
        raise RuntimeError('LOCALAPPDATA introuvable : exécutez sur Windows.')
    return {
        'project': project / 'runtime' / 'auth' / 'users.sqlite3',
        'packaged': project / 'dist' / 'AURUM-AI V5' / '_internal' / 'runtime' / 'auth' / 'users.sqlite3',
        'persistent': Path(local) / 'AURUM-AI' / 'auth' / 'users.sqlite3',
    }


def read_only_connection(source: Path) -> sqlite3.Connection:
    uri = 'file:' + quote(str(source.resolve()).replace('\\', '/'), safe='/:') + '?mode=ro'
    return sqlite3.connect(uri, uri=True, timeout=5)


def inspect_db(path: Path) -> dict[str, int | str | bool]:
    if not path.is_file():
        return {'exists': False}
    with path.open('rb') as f:
        if f.read(16) != b'SQLite format 3\x00':
            return {'exists': True, 'valid_sqlite': False}
    with closing(read_only_connection(path)) as conn:
        if conn.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            return {'exists': True, 'valid_sqlite': False}
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'users' not in tables:
            return {'exists': True, 'valid_sqlite': True, 'users_table': False}
        users = conn.execute('SELECT COUNT(*) FROM users').fetchone()[0]
        verified = conn.execute('SELECT COUNT(*) FROM users WHERE email_verified = 1').fetchone()[0]
        return {'exists': True, 'valid_sqlite': True, 'users_table': True,
                'users': users, 'verified': verified}


def migrate(source: Path, target: Path) -> str:
    """Copy complete consistent SQLite database; never overwrite target."""
    info = inspect_db(source)
    if not info.get('valid_sqlite') or not info.get('users_table'):
        raise RuntimeError('Source absente, incorrecte ou sans table users.')
    if target.exists():
        raise FileExistsError('Base persistante déjà présente : aucune modification effectuée. Ne pas écraser ni fusionner automatiquement.')
    target.parent.mkdir(parents=True, exist_ok=True)
    # Do not copy raw SQLite bytes (would miss uncheckpointed WAL transactions).
    descriptor, temp = tempfile.mkstemp(prefix='aurum_auth_migration_', suffix='.sqlite3', dir=target.parent)
    os.close(descriptor)
    temp_path = Path(temp)
    try:
        # sqlite3.Connection's context manager commits/rolls back, but DOES NOT
        # close the connection. On Windows, an open handle prevents unlinking
        # the temporary database (WinError 32). closing() releases both handles.
        with closing(read_only_connection(source)) as src, closing(sqlite3.connect(str(temp_path))) as dst:
            src.backup(dst)
            dst.commit()
        restored = inspect_db(temp_path)
        if not restored.get('valid_sqlite') or not restored.get('users_table'):
            raise RuntimeError('La sauvegarde SQLite ne passe pas les contrôles.')
        if restored['users'] != info['users'] or restored['verified'] != info['verified']:
            raise RuntimeError('Le nombre de comptes/vérifications a changé pendant la copie. Fermez les deux applications et réessayez.')
        if target.exists():
            raise FileExistsError('Une base persistante est apparue : copie abandonnée.')
        # Windows: no replace of an existing file if it appears concurrently.
        # Create through hardlink with CREATE_NEW-like semantics, then unlink temp.
        os.link(temp_path, target)
        return f'Copie sécurisée terminée : {restored["users"]} compte(s), {restored["verified"]} confirmé(s). Source originale conservée.'
    finally:
        temp_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Diagnostic/migration du stockage auth AURUM-AI (sans affichage de secrets).')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--check', action='store_true', help='Audit en lecture seule (défaut).')
    group.add_argument('--apply', action='store_true', help='Copie sans écrasement vers LOCALAPPDATA.')
    parser.add_argument('--source', choices=('project', 'packaged'), default='project', help='Source pour --apply : project par défaut.')
    args = parser.parse_args(argv)
    project = Path(__file__).resolve().parents[2]
    loc = paths(project, dict(os.environ))
    print('AURUM-AI | Phase 21.5 : stockage SQLite persistant')
    print('Ne montre aucun email, hash, mot de passe ni token.')
    for name, location in loc.items():
        print(f'[{name}] {location}')
        try:
            print('  ', inspect_db(location))
        except (sqlite3.DatabaseError, OSError) as exc:
            print(f'  Lecture impossible : {type(exc).__name__}')
    if args.apply:
        try:
            print(migrate(loc[args.source], loc['persistent']))
        except (FileExistsError, RuntimeError, sqlite3.Error, OSError) as exc:
            print(f'COPIE NON EFFECTUÉE : {exc}')
            return 2
    else:
        print('Diagnostic terminé : aucune base n’a été créée ou modifiée.')
        print('Si project contient les comptes et persistent est absent : --apply --source project')
        print('Si seule la base packaged possède le compte, utilisez --source packaged uniquement après vérification.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
