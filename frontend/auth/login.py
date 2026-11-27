
"""AURUM-AI — Secure local account login.

Features:
- Email and password authentication
- Argon2id password verification
- Email verification required
- Five failed attempts maximum
- Temporary lockout for 15 minutes
- Generic authentication errors
- No password hashes returned to the interface
"""

import time
from contextlib import closing
from functools import lru_cache

from frontend.auth.accounts import normalize_email
from frontend.auth.database import get_connection, init_db
from frontend.auth.security import hash_password, verify_password


MAX_FAILED_ATTEMPTS = 5
ATTEMPT_WINDOW_SECONDS = 900
LOCK_DURATION_SECONDS = 900


@lru_cache(maxsize=1)
def _dummy_password_hash():
    """Avoid skipping expensive verification for unknown accounts."""
    return hash_password("AurumDummyPassword2026!Secure")


def _create_login_attempts_table(conn):
    """Create the persistent login-attempt tracking table."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS login_attempts (
            email TEXT PRIMARY KEY COLLATE NOCASE,
            failed_attempts INTEGER NOT NULL DEFAULT 0,
            window_started_at INTEGER NOT NULL,
            locked_until INTEGER NOT NULL DEFAULT 0
        )
        """
    )


def authenticate_user(email: str, password: str) -> dict | None:
    """Authenticate a verified local account.

    Returns a public user dictionary when authentication succeeds.
    Returns None when credentials are invalid, the email is not
    verified, or the account is temporarily locked.

    Never returns the password hash.
    """

    try:
        clean_email = normalize_email(email)
    except ValueError:
        return None

    if not isinstance(password, str) or not password:
        return None

    init_db()
    now = int(time.time())

    with closing(get_connection()) as conn:
        with conn:
            # Serialize local login attempt updates.
            conn.execute("BEGIN IMMEDIATE")
            _create_login_attempts_table(conn)

            attempt = conn.execute(
                """
                SELECT failed_attempts, window_started_at, locked_until
                FROM login_attempts
                WHERE email = ?
                """,
                (clean_email,),
            ).fetchone()

            if attempt and now < attempt["locked_until"]:
                return None

            user = conn.execute(
                """
                SELECT
                    id,
                    display_name,
                    email,
                    password_hash,
                    email_verified,
                    created_at
                FROM users
                WHERE email = ?
                """,
                (clean_email,),
            ).fetchone()

            # Verify even for an unknown or Google-only account
            # to reduce differences in response time.
            stored_hash = (
                user["password_hash"]
                if user is not None and user["password_hash"]
                else _dummy_password_hash()
            )

            password_valid = verify_password(
                password,
                stored_hash,
            )

            if password_valid and user is not None:
                if user["email_verified"] != 1:
                    # Correct password, but email confirmation
                    # is still required.
                    return None

                # Successful login clears previous failures.
                conn.execute(
                    "DELETE FROM login_attempts WHERE email = ?",
                    (clean_email,),
                )

                return {
                    "id": int(user["id"]),
                    "display_name": user["display_name"],
                    "email": user["email"],
                    "email_verified": True,
                    "created_at": user["created_at"],
                }

            # Record unsuccessful password attempts.
            if (
                attempt is None
                or now - attempt["window_started_at"]
                >= ATTEMPT_WINDOW_SECONDS
            ):
                failures = 1
                window_start = now
            else:
                failures = attempt["failed_attempts"] + 1
                window_start = attempt["window_started_at"]

            locked_until = (
                now + LOCK_DURATION_SECONDS
                if failures >= MAX_FAILED_ATTEMPTS
                else 0
            )

            conn.execute(
                """
                INSERT INTO login_attempts (
                    email,
                    failed_attempts,
                    window_started_at,
                    locked_until
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(email) DO UPDATE SET
                    failed_attempts = excluded.failed_attempts,
                    window_started_at = excluded.window_started_at,
                    locked_until = excluded.locked_until
                """,
                (
                    clean_email,
                    failures,
                    window_start,
                    locked_until,
                ),
            )

            return None
