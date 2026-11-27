
"""AURUM-AI — User account management.

Local registration and account lookup.
Email verification will be implemented separately.
"""

import re
import sqlite3
from contextlib import closing

from frontend.auth.database import get_connection, init_db
from frontend.auth.security import hash_password


class AccountAlreadyExistsError(ValueError):
    """Raised when an email is already registered."""


def normalize_email(email: str) -> str:
    """Validate and normalize an email address."""

    if not isinstance(email, str):
        raise ValueError("Email must be a string.")

    email = email.strip().lower()

    if len(email) > 254:
        raise ValueError("Email address is too long.")

    pattern = r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,63}"

    if not re.fullmatch(pattern, email):
        raise ValueError("Please enter a valid email address.")

    return email


def validate_display_name(display_name: str) -> str:
    """Validate the user's display name."""

    if not isinstance(display_name, str):
        raise ValueError("Display name must be a string.")

    display_name = " ".join(display_name.split())

    if not 2 <= len(display_name) <= 80:
        raise ValueError(
            "Display name must contain between 2 and 80 characters."
        )

    return display_name


def create_account(
    display_name: str,
    email: str,
    password: str,
) -> int:
    """Register a new local account.

    The account remains unverified until the email
    confirmation process is completed.

    Returns:
        ID of the newly created user.
    """

    clean_name = validate_display_name(display_name)
    clean_email = normalize_email(email)

    # Argon2id: never store the original password.
    password_hash = hash_password(password)

    init_db()

    with closing(get_connection()) as conn:
        try:
            with conn:
                cursor = conn.execute(
                    """
                    INSERT INTO users (
                        display_name,
                        email,
                        password_hash,
                        email_verified
                    )
                    VALUES (?, ?, ?, 0)
                    """,
                    (
                        clean_name,
                        clean_email,
                        password_hash,
                    ),
                )

            return int(cursor.lastrowid)

        except sqlite3.IntegrityError as exc:
            if "UNIQUE constraint failed: users.email" in str(exc):
                raise AccountAlreadyExistsError(
                    "This email is already registered."
                ) from exc

            raise


def get_user_by_email(email: str) -> dict | None:
    """Find an account without exposing its password hash."""

    clean_email = normalize_email(email)

    init_db()

    with closing(get_connection()) as conn:
        user = conn.execute(
            """
            SELECT
                id,
                display_name,
                email,
                email_verified,
                google_sub,
                created_at
            FROM users
            WHERE email = ?
            """,
            (clean_email,),
        ).fetchone()

    return dict(user) if user is not None else None


def get_user_by_id(user_id: int) -> dict | None:
    """Find an account by its database ID."""

    init_db()

    with closing(get_connection()) as conn:
        user = conn.execute(
            """
            SELECT
                id,
                display_name,
                email,
                email_verified,
                google_sub,
                created_at
            FROM users
            WHERE id = ?
            """,
            (user_id,),
        ).fetchone()

    return dict(user) if user is not None else None
