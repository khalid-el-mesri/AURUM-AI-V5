
"""AURUM-AI — Registration and email verification workflow.

Connects:
- accounts.py: account creation and lookup
- security.py: password verification
- verification.py: secure OTP generation and validation
- email_service.py: Gmail SMTP delivery

No access to the Dashboard is granted by this module.
"""

from contextlib import closing

from frontend.auth.accounts import (
    create_account,
    get_user_by_email,
    normalize_email,
)
from frontend.auth.database import get_connection
from frontend.auth.email_service import (
    EmailConfigurationError,
    EmailDeliveryError,
    send_verification_email,
)
from frontend.auth.security import verify_password
from frontend.auth.verification import (
    create_verification_code,
    verify_verification_code,
)


class RegistrationDeliveryError(RuntimeError):
    """Account exists, but its verification email was not sent."""


class AccountVerificationError(ValueError):
    """Email verification or OTP resend failed."""


def register_and_send_otp(
    display_name: str,
    email: str,
    password: str,
) -> int:
    """Create an account and send its email verification code.

    Workflow:
        1. Validate and normalize the email.
        2. Register the user with an Argon2id password hash.
        3. Generate a secure six-digit OTP.
        4. Send the OTP through Gmail SMTP.

    Returns:
        The new user's database ID.

    Important:
        The account remains unverified until the user
        successfully submits the OTP.
    """

    clean_email = normalize_email(email)

    # AccountAlreadyExistsError is raised automatically
    # if the email is already registered.
    user_id = create_account(
        display_name=display_name,
        email=clean_email,
        password=password,
    )

    # OTP is stored as a hash in SQLite.
    code = create_verification_code(user_id)

    try:
        send_verification_email(
            recipient_email=clean_email,
            verification_code=code,
        )

    except (EmailConfigurationError, EmailDeliveryError):
        # The account already exists in SQLite.
        # Do not delete it: the user can retry after
        # the resend cooldown has expired.
        raise RegistrationDeliveryError(
            "Le compte a été créé, mais l'email "
            "de vérification n'a pas pu être envoyé. "
            "Veuillez patienter au moins 60 secondes "
            "avant de demander un nouveau code."
        ) from None

    return user_id


def resend_verification_otp(
    email: str,
    password: str,
) -> bool:
    """Resend an OTP for an existing, unverified local account.

    The account password is required before sending.
    The verification module enforces its resend cooldown.

    Returns:
        True if the SMTP server accepted the message.

    Raises:
        AccountVerificationError for invalid credentials
        or an already verified account.
    """

    clean_email = normalize_email(email)

    # Read the password hash only inside the server.
    # Never return it to the user interface.
    with closing(get_connection()) as conn:
        user = conn.execute(
            """
            SELECT id, password_hash, email_verified
            FROM users
            WHERE email = ?
            """,
            (clean_email,),
        ).fetchone()

    if (
        user is None
        or user["email_verified"] == 1
        or user["password_hash"] is None
        or not verify_password(
            password,
            user["password_hash"],
        )
    ):
        raise AccountVerificationError(
            "Impossible de renvoyer le code "
            "avec les informations fournies."
        )

    code = create_verification_code(int(user["id"]))

    try:
        send_verification_email(
            recipient_email=clean_email,
            verification_code=code,
        )

    except (EmailConfigurationError, EmailDeliveryError):
        raise RegistrationDeliveryError(
            "L'email n'a pas pu être envoyé. "
            "Veuillez réessayer ultérieurement."
        ) from None

    return True


def confirm_email_registration(
    email: str,
    submitted_code: str,
) -> bool:
    """Verify an OTP and activate an email account.

    Returns:
        True if the verification succeeded.
        False if the account or OTP is invalid,
        expired, already verified, or blocked.

    The underlying verification module handles:
        - 10-minute OTP expiration
        - Maximum five incorrect attempts
        - One-time consumption
        - Atomic activation of the account
    """

    try:
        clean_email = normalize_email(email)
    except ValueError:
        return False

    user = get_user_by_email(clean_email)

    if user is None:
        return False

    if user["email_verified"] == 1:
        return False

    return verify_verification_code(
        user_id=int(user["id"]),
        submitted_code=submitted_code,
    )


def is_email_verified(email: str) -> bool:
    """Return whether an existing user's email is verified."""

    try:
        user = get_user_by_email(email)
    except ValueError:
        return False

    return bool(
        user is not None
        and user["email_verified"] == 1
    )
