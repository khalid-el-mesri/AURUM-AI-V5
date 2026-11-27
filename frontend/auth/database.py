"""AURUM-AI local account database, stored outside the installed application.

An installed PyInstaller app may be read-only and its _internal folder is
replaceable during upgrades.  Account records therefore live in user-local
application data, and are never placed in the downloadable executable.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Mapping


def resolve_db_path(env: Mapping[str, str] | None = None, home: Path | None = None) -> Path:
    """Choose a stable per-user SQLite path independent of __file__/cwd."""
    values = os.environ if env is None else env
    local = values.get('LOCALAPPDATA', '').strip()
    if local:
        return Path(local).expanduser() / 'AURUM-AI' / 'auth' / 'users.sqlite3'
    xdg = values.get('XDG_DATA_HOME', '').strip()
    base = Path(xdg).expanduser() if xdg else (home if home is not None else Path.home()) / '.local' / 'share'
    return base / 'AURUM-AI' / 'auth' / 'users.sqlite3'


DB_PATH = resolve_db_path()


def get_connection():
    """Open SQLite database; same persistent path in Python and packaged EXE."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON')
    return conn


def init_db():
    """Create missing schema without replacing existing accounts."""
    conn = get_connection()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                display_name TEXT NOT NULL,
                email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_hash TEXT,
                google_sub TEXT UNIQUE,
                email_verified INTEGER NOT NULL DEFAULT 0
                    CHECK (email_verified IN (0, 1)),
                created_at INTEGER NOT NULL
                    DEFAULT (CAST(strftime('%s', 'now') AS INTEGER)),
                CHECK (password_hash IS NOT NULL OR google_sub IS NOT NULL)
            );
            CREATE TABLE IF NOT EXISTS email_verification (
                user_id INTEGER PRIMARY KEY,
                code_hash TEXT NOT NULL,
                expires_at INTEGER NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0
                    CHECK (attempts BETWEEN 0 AND 5),
                created_at INTEGER NOT NULL
                    DEFAULT (CAST(strftime('%s', 'now') AS INTEGER)),
                FOREIGN KEY (user_id)
                    REFERENCES users(id)
                    ON DELETE CASCADE
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


if __name__ == '__main__':
    init_db()
    print('AURUM-AI authentication database initialized.')
    print(f'Database: {DB_PATH}')
