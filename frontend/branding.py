"""Branding AURUM-AI V5: dedicated app logo and favicon (no research writes)."""
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
_BRAND = _ROOT / "assets" / "branding"


def aurum_favicon() -> str:
    """The PNG path is accepted by Streamlit's page_icon API."""
    return str(_BRAND / "aurum_ai_icon.png")


def apply_aurum_branding() -> None:
    """Place the logo in the Streamlit sidebar without altering session state."""
    wordmark = _BRAND / "aurum_ai_wordmark.png"
    icon = _BRAND / "aurum_ai_icon.png"
    if wordmark.is_file() and icon.is_file():
        if hasattr(st, "logo"):
            st.logo(str(wordmark), icon_image=str(icon), size="large")
