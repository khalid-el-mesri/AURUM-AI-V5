
"""AURUM-AI — Gmail OTP email delivery service.

Sends verification codes using Gmail SMTP over TLS.

Credentials are loaded from:
1. Environment variables, or
2. .streamlit/secrets.toml

Never store credentials directly in source code.
"""

import os
import re
import ssl
import smtplib
import tomllib

from pathlib import Path
from email.message import EmailMessage

from frontend.auth.accounts import normalize_email


ROOT = Path(__file__).resolve().parents[2]

SECRETS_PATH = ROOT / ".streamlit" / "secrets.toml"

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465


class EmailConfigurationError(RuntimeError):
    """Raised when Gmail credentials are missing."""


class EmailDeliveryError(RuntimeError):
    """Raised when an email cannot be delivered to the SMTP server."""


def get_gmail_credentials():
    """Load Gmail credentials without exposing secrets."""

    username = os.getenv("AURUM_GMAIL_ADDRESS", "").strip()
    password = os.getenv("AURUM_GMAIL_APP_PASSWORD", "").strip()

    if SECRETS_PATH.is_file():
        with SECRETS_PATH.open("rb") as file:
            config = tomllib.load(file)

        smtp_config = config.get("smtp", {})

        if not username:
            username = smtp_config.get("username", "")

        if not password:
            password = smtp_config.get("app_password", "")

    if not username or not password:
        raise EmailConfigurationError(
            "Gmail SMTP credentials are not configured. "
            "Configure .streamlit/secrets.toml."
        )

    username = normalize_email(username)

    # Google may display application passwords with spaces.
    password = password.replace(" ", "").strip()

    if not password:
        raise EmailConfigurationError(
            "Gmail application password is empty."
        )

    return username, password


def send_verification_email(
    recipient_email: str,
    verification_code: str,
) -> None:
    """Send a six-digit OTP code to the recipient.

    The code must already have been generated
    by the verification module.

    Do not log the code or the Gmail password.
    """

    recipient_email = normalize_email(recipient_email)

    if not isinstance(verification_code, str):
        raise ValueError("Invalid verification code.")

    if not re.fullmatch(r"[0-9]{6}", verification_code):
        raise ValueError(
            "Verification code must contain exactly 6 digits."
        )

    sender_email, app_password = get_gmail_credentials()

    message = EmailMessage()

    message["From"] = f"AURUM-AI <{sender_email}>"
    message["To"] = recipient_email
    message["Subject"] = "AURUM-AI | Verification de votre compte"

    message.set_content(
        f"""Bonjour,

Bienvenue sur AURUM-AI !

Pour confirmer votre adresse email,
veuillez saisir le code suivant dans l'application :

{verification_code}

Ce code est valable pendant 10 minutes.

Si vous n'avez pas demandé la création
d'un compte AURUM-AI, ignorez cet email.

Cordialement,
AURUM-AI Security Team

Ce message a été généré automatiquement.
"""
    )

    context = ssl.create_default_context()

    try:
        with smtplib.SMTP_SSL(
            SMTP_HOST,
            SMTP_PORT,
            context=context,
            timeout=20,
        ) as smtp:

            smtp.login(sender_email, app_password)

            smtp.send_message(message)

    except (smtplib.SMTPException, OSError) as exc:
        raise EmailDeliveryError(
            "Impossible d'envoyer l'email de verification. "
            "Verifiez la configuration Gmail et la connexion."
        ) from None
