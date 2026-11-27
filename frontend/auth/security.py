
"""AURUM-AI — Password security using Argon2id."""

from argon2 import PasswordHasher
from argon2.exceptions import (
    InvalidHashError,
    VerificationError,
    VerifyMismatchError,
)
from argon2.low_level import Type


# Secure password hashing configuration
password_hasher = PasswordHasher(
    time_cost=3,
    memory_cost=65536,
    parallelism=2,
    hash_len=32,
    salt_len=16,
    type=Type.ID,
)


def hash_password(password: str) -> str:
    """Hash a password securely before database storage."""

    if not isinstance(password, str):
        raise ValueError("Password must be a string.")

    if len(password) < 12:
        raise ValueError("Password must contain at least 12 characters.")

    if len(password.encode("utf-8")) > 1024:
        raise ValueError("Password is too long.")

    return password_hasher.hash(password)


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against its stored Argon2id hash."""

    if not isinstance(password, str) or not isinstance(stored_hash, str):
        return False

    try:
        return password_hasher.verify(stored_hash, password)

    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def password_needs_rehash(stored_hash: str) -> bool:
    """Check whether a stored password hash needs updating."""

    return password_hasher.check_needs_rehash(stored_hash)
