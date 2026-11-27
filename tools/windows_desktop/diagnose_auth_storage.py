"""AURUM-AI V5: safe local authentication-store location diagnostic.

Read-only by default. Never reads password hashes or displays emails.
Do not distribute private local database files or authentication logs.
"""
from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
import re
import sqlite3
from urllib.parse import quote

DB_EXTENSIONS = {'.db', '.sqlite', '.sqlite3'}
JSON_HINTS = {'users', 'accounts', 'auth', 'account', 'user'}
SKIP = {'.venv', '.git', '__pycache__', '.pytest_cache', 'node_modules', 'build', 'tests'}
SOURCE_FILES = ('database.py', 'accounts.py', 'login.py', 'registration.py', 'security.py')
SOURCE_TAGS = {
    'sqlite3.connect': re.compile(r'\bsqlite3\s*\.\s*connect\s*\('),
    'localappdata': re.compile(r'LOCALAPPDATA', re.I),
    'appdata': re.compile(r'\bAPPDATA\b', re.I),
    'file-relative': re.compile(r'\b__file__\b'),
    'home-relative': re.compile(r'Path\s*\.\s*home\s*\('),
    'auth-db-name': re.compile(r'\.sqlite3?\b|\.db\b', re.I),
    'db-path': re.compile(r'\b(DB_PATH|DATABASE_PATH|DATABASE_URL|db_file|db_path)\b', re.I),
    'sqlite-import': re.compile(r'\bimport\s+sqlite3\b'),
}


def locate_root(script: Path) -> Path:
    # tools/windows_desktop/diagnose_auth_storage.py
    return script.resolve().parents[2]


def roots_to_scan(project: Path, env: dict[str, str]):
    """Bounded location search; do not crawl the entire Windows user profile."""
    bases = [
        ('project-auth', project / 'frontend' / 'auth'),
        ('project-data', project / 'data'),
        ('project-backend', project / 'backend'),
        ('project-storage', project / 'storage'),
        ('project-instance', project / 'instance'),
        ('project-databases', project / 'databases'),
        ('bundled-auth', project / 'dist' / 'AURUM-AI V5' / '_internal' / 'frontend' / 'auth'),
        ('bundled-data', project / 'dist' / 'AURUM-AI V5' / '_internal' / 'data'),
        ('bundled-backend', project / 'dist' / 'AURUM-AI V5' / '_internal' / 'backend'),
        ('bundled-storage', project / 'dist' / 'AURUM-AI V5' / '_internal' / 'storage'),
    ]
    for label in ('LOCALAPPDATA', 'APPDATA'):
        base = env.get(label)
        if base:
            for app in ('AURUM-AI', 'AURUM_AI', 'AURUMAI'):
                bases.append((f'{label.lower()}-{app}', Path(base) / app))
    return [(k, p) for k, p in bases if p.is_dir()]


def file_is_candidate(p: Path) -> bool:
    if p.suffix.lower() in DB_EXTENSIONS:
        return True
    if p.suffix.lower() == '.json':
        return any(hint in p.stem.lower() for hint in JSON_HINTS)
    return False


def scan_storage(project: Path, env: dict[str, str]):
    records: list[tuple[str, Path, Path]] = []
    seen: set[Path] = set()
    for label, base in roots_to_scan(project, env):
        for current, dirs, files in os.walk(base, topdown=True, followlinks=False):
            dirs[:] = [x for x in dirs if x.lower() not in SKIP and not (Path(current) / x).is_symlink()]
            for filename in files:
                p = Path(current) / filename
                if not file_is_candidate(p) or p.is_symlink():
                    continue
                resolved = p.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                records.append((label, base, p))
    # Account databases can also live directly in the repository root.
    for p in project.iterdir():
        if p.is_file() and file_is_candidate(p) and not p.is_symlink() and p.resolve() not in seen:
            records.append(('project-root', project, p))
    return sorted(records, key=lambda r: (r[0], str(r[2]).lower()))


def scan_source(project: Path):
    result: list[tuple[str, list[str]]] = []
    for name in SOURCE_FILES:
        path = project / 'frontend' / 'auth' / name
        if not path.is_file():
            continue
        data = path.read_text(encoding='utf-8-sig', errors='replace')
        tags = sorted(k for k, pattern in SOURCE_TAGS.items() if pattern.search(data))
        result.append((f'frontend/auth/{name}', tags))
    return result


def is_sqlite(path: Path) -> bool:
    with path.open('rb') as stream:
        return stream.read(16) == b'SQLite format 3\x00'


def query_email_presence(path: Path, email: str):
    """Query ONLY email equality and optional verification fields, never passwords."""
    # URI mode=ro does not allow writes to the database; never create DB files.
    uri = 'file:' + quote(str(path.resolve()).replace('\\', '/'), safe='/:') + '?mode=ro'
    conn = sqlite3.connect(uri, uri=True, timeout=1)
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )]
        rows = []
        for table in tables:
            safe_table = table.replace('"', '""')
            cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{safe_table}")')]
            emails = [c for c in cols if c.casefold() in ('email', 'email_address', 'user_email')]
            if not emails:
                continue
            col = emails[0].replace('"', '""')
            verification_names = ('email_verified', 'is_verified', 'verified', 'is_active', 'status')
            verify = next((c for c in cols if c.casefold() in verification_names), None)
            if verify is None:
                query = f'SELECT 1 FROM "{safe_table}" WHERE lower("{col}")=? LIMIT 1'
            else:
                safe_verify = verify.replace('"', '""')
                query = f'SELECT "{safe_verify}" FROM "{safe_table}" WHERE lower("{col}")=? LIMIT 1'
            hit = conn.execute(query, (email.casefold(),)).fetchone()
            if hit is not None:
                status = 'unknown (no verification field)' if verify is None else f'{verify}={hit[0]!r}'
                rows.append((table, status))
        return rows
    finally:
        conn.close()


def produce_report(project: Path, env: dict[str, str], check_email: bool = False, email: str = '') -> list[str]:
    lines = [
        'AURUM-AI | Diagnostic stockage authentification (lecture seule)',
        'Aucun mot de passe, hash, token ou email ne sera affiché.',
        '--- Indicateurs statiques du code auth (sans contenu sensible) ---',
    ]
    code = scan_source(project)
    if not code:
        lines.append('Code frontend/auth introuvable.')
    for name, tags in code:
        lines.append(f'{name}: {", ".join(tags) if tags else "aucune règle détectée"}')
    lines.append('--- Fichiers de stockage candidats ---')
    storage = scan_storage(project, env)
    if not storage:
        lines.append('Aucun fichier SQLite/JSON lié aux comptes dans les emplacements courants.')
    for label, base, path in storage:
        try:
            size = path.stat().st_size
        except OSError:
            lines.append(f'[{label}] {path.relative_to(base)} | non accessible')
            continue
        kind = 'SQLite' if path.suffix.lower() in DB_EXTENSIONS and is_sqlite(path) else 'autre/JSON'
        lines.append(f'[{label}] {path.relative_to(base)} | {size} octets | {kind}')
        if label.startswith('bundled'):
            lines.append('  ATTENTION : stockage dans _internal, non adapté à une base de comptes persistante.')
        if check_email and email and kind == 'SQLite':
            try:
                matches = query_email_presence(path, email)
                if matches:
                    for table, status in matches:
                        lines.append(f'  Compte: présent, table={table}, vérification={status}')
                else:
                    lines.append('  Compte: non trouvé dans les tables avec colonne email.')
            except (sqlite3.Error, OSError) as exc:
                lines.append(f'  Compte: lecture impossible ({type(exc).__name__})')
    lines.extend([
        '--- Interprétation ---',
        'Si la base du projet existe mais pas celle de la version packagée, vérifier le chemin dans frontend/auth/database.py.',
        'Si le compte est non confirmé, vérifier le flux de validation email sans contourner la vérification.',
        'Ne pas copier, supprimer ou inclure une base de comptes dans un ZIP public.',
        'Ce diagnostic ne lit pas et ne teste pas les mots de passe.',
    ])
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Diagnostic en lecture seule des bases de comptes AURUM-AI')
    parser.add_argument('--check-account', action='store_true', help='Demande localement un email pour vérifier sa présence sans afficher sa valeur')
    args = parser.parse_args(argv)
    root = locate_root(Path(__file__))
    email = getpass.getpass('Email du compte à vérifier (saisie masquée, ne PAS saisir de mot de passe): ').strip() if args.check_account else ''
    for line in produce_report(root, dict(os.environ), args.check_account, email):
        print(line)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
