"""Windows-only optional remembered email. No password storage."""
import os
import sys

SERVICE = "AURUM-AI-V5-Desktop-RememberMe"
EMAIL_KEY = "__remembered_account__"


def enabled():
    if os.name != "nt" or not getattr(sys, "frozen", False):
        return False
    try:
        import keyring
        from keyring.backends.Windows import WinVaultKeyring
        return isinstance(keyring.get_keyring(), WinVaultKeyring)
    except Exception:
        return False


def _vault():
    if not enabled():
        raise RuntimeError("Windows credential storage unavailable")
    import keyring
    return keyring


def remembered_email():
    if not enabled():
        return None
    return _vault().get_password(SERVICE, EMAIL_KEY)


def save_email(email):
    email = str(email).strip().lower()
    if not email or "@" not in email:
        raise ValueError("Invalid email")
    _vault().set_password(SERVICE, EMAIL_KEY, email)


def forget_credentials():
    vault = _vault()
    from keyring.errors import PasswordDeleteError
    email = remembered_email()

    # Also clean any leftover credential from the previous implementation.
    for account in (email, EMAIL_KEY):
        if account:
            try:
                vault.delete_password(SERVICE, account)
            except PasswordDeleteError:
                pass
