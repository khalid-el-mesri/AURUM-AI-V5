
"""AURUM-AI — Secure email verification with OTP codes.

Features:
- Cryptographically secure 6-digit codes
- 10-minute expiration
- Maximum 5 incorrect attempts per code
- 60-second resend cooldown
- Argon2id hashing
- SQLite transaction protection

Important:
This module does NOT send emails.
The Gmail sending service will be implemented separately.
"""

import secrets
import time
from contextlib import closing

from argon2.exceptions import (
    InvalidHashError,
    VerificationError,
)

from frontend.auth.database import get_connection, init_db
from frontend.auth.security import password_hasher


OTP_EXPIRATION_SECONDS = 600
MAX_ATTEMPTS = 5
RESEND_COOLDOWN_SECONDS = 60


class OTPRequestError(ValueError):
    """Error when requesting a verification code."""


def create_verification_code(user_id: int) -> str:
    """Create an OTP for an existing, unverified user.

    Returns the plaintext code ONLY so that the email
    service can send it to the account owner.

    Never log or store the plaintext code.
    """

    if type(user_id) is not int or user_id <= 0:
        raise ValueError("Invalid user ID.")

    init_db()

    now = int(time.time())

    with closing(get_connection()) as conn:
        with conn:
            # Prevent concurrent requests from bypassing limits.
            conn.execute("BEGIN IMMEDIATE")

            user = conn.execute(
                """
                SELECT id, email_verified
                FROM users
                WHERE id = ?
                """,
                (user_id,),
            ).fetchone()

            if user is None:
                raise OTPRequestError("User does not exist.")

            if user["email_verified"]:
                raise OTPRequestError(
                    "This email address is already verified."
                )

            existing = conn.execute(
                """
                SELECT expires_at, attempts, created_at
                FROM email_verification
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()

            if existing is not None:

                if (
                    existing["attempts"] >= MAX_ATTEMPTS
                    and now < existing["expires_at"]
                ):
                    raise OTPRequestError(
                        "Too many attempts. Wait until the code expires."
                    )

                elapsed = now - existing["created_at"]

                if elapsed < RESEND_COOLDOWN_SECONDS:
                    remaining = RESEND_COOLDOWN_SECONDS - elapsed

                    raise OTPRequestError(
                        f"Please wait {remaining} seconds "
                        "before requesting another code."
                    )

            # Generate a uniformly random six-digit code.
            code = f"{secrets.randbelow(1_000_000):06d}"

            # Hash the code with Argon2id before storage.
            code_hash = password_hasher.hash(
                f"aurum-email-otp:{user_id}:{code}"
            )

            expiration = now + OTP_EXPIRATION_SECONDS

            conn.execute(
                """
                INSERT INTO email_verification (
                    user_id,
                    code_hash,
                    expires_at,
                    attempts,
                    created_at
                )
                VALUES (?, ?, ?, 0, ?)

                ON CONFLICT(user_id)
                DO UPDATE SET
                    code_hash = excluded.code_hash,
                    expires_at = excluded.expires_at,
                    attempts = 0,
                    created_at = excluded.created_at
                """,
                (
                    user_id,
                    code_hash,
                    expiration,
                    now,
                ),
            )

    return code


def verify_verification_code(
    user_id: int,
    submitted_code: str,
) -> bool:
    """Validate an OTP and activate the user's email.

    Returns:
        True if verified successfully.
        False if invalid, expired, or blocked.
    """

    if type(user_id) is not int or user_id <= 0:
        return False

    init_db()

    now = int(time.time())

    with closing(get_connection()) as conn:
        with conn:
            conn.execute("BEGIN IMMEDIATE")

            user = conn.execute(
                """
                SELECT id, email_verified
                FROM users
                WHERE id = ?
                """,
                (user_id,),
            ).fetchone()

            if user is None or user["email_verified"]:
                return False

            record = conn.execute(
                """
                SELECT code_hash, expires_at, attempts
                FROM email_verification
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()

            if record is None:
                return False

            # Reject and remove expired verification codes.
            if now >= record["expires_at"]:
                conn.execute(
                    """
                    DELETE FROM email_verification
                    WHERE user_id = ?
                    """,
                    (user_id,),
                )
                return False

            # Block further attempts on this code.
            if record["attempts"] >= MAX_ATTEMPTS:
                return False

            valid = False

            if (
                isinstance(submitted_code, str)
                and len(submitted_code) == 6
                and submitted_code.isascii()
                and submitted_code.isdecimal()
            ):
                try:
                    valid = password_hasher.verify(
                        record["code_hash"],
                        f"aurum-email-otp:{user_id}:{submitted_code}",
                    )
                except (VerificationError, InvalidHashError):
                    valid = False

            if not valid:
                conn.execute(
                    """
                    UPDATE email_verification
                    SET attempts = attempts + 1
                    WHERE user_id = ?
                    """,
                    (user_id,),
                )
                return False

            # Activate the account and consume the OTP atomically.
            conn.execute(
                """
                UPDATE users
                SET email_verified = 1
                WHERE id = ?
                """,
                (user_id,),
            )

            conn.execute(
                """
                DELETE FROM email_verification
                WHERE user_id = ?
                """,
                (user_id,),
            )

            return True
