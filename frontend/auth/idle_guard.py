"""AURUM-AI local Streamlit session inactivity guard.

The Streamlit backend is the authority on session expiration.
A small same-origin browser component notifies the backend of real user
activity; passive fragment reruns are never treated as user activity.
"""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

IDLE_TIMEOUT_SECONDS = 300  # Five minutes. Use 20 only for a quick local test.
WATCHDOG_INTERVAL_SECONDS = 3

_COMPONENT_DIRECTORY = Path(__file__).resolve().parent / "idle_component"
_browser_idle = components.declare_component(
    "aurum_idle_activity_v1", path=str(_COMPONENT_DIRECTORY)
)


def start_local_session(user: dict) -> None:
    """Start or unlock a session after successful credential verification."""
    st.session_state["aurum_local_user"] = user
    st.session_state["aurum_last_activity_at"] = time.monotonic()
    st.session_state["aurum_last_browser_event_id"] = None
    # A fresh browser observer is essential after password-only unlock:
    # otherwise the old widget's cached "idle" event could replay.
    st.session_state["aurum_idle_monitor_generation"] = (
        st.session_state.get("aurum_idle_monitor_generation", 0) + 1
    )
    st.session_state["aurum_session_locked"] = False
    st.session_state["aurum_session_lock_reason"] = None


def lock_local_session() -> None:
    """Block access on inactivity while retaining only the account identity.

    No password, password hash, or authorization is retained in this state.
    The account must authenticate again before protected content is shown.
    """
    if st.session_state.get("aurum_local_user") is None:
        return

    st.session_state["aurum_session_locked"] = True
    st.session_state.pop("aurum_last_activity_at", None)
    st.session_state.pop("aurum_last_browser_event_id", None)
    timeout_label = (
        "5 minutes" if IDLE_TIMEOUT_SECONDS == 300
        else f"{IDLE_TIMEOUT_SECONDS} secondes"
    )
    st.session_state["aurum_session_lock_reason"] = (
        f"Session verrouillée après {timeout_label} d’inactivité. "
        "Saisissez votre mot de passe pour continuer."
    )


def end_local_session(*, idle: bool = False) -> None:
    """Log out completely; idle=True keeps the account but locks access."""
    if idle:
        lock_local_session()
        return

    st.session_state["aurum_local_user"] = None
    st.session_state["aurum_session_locked"] = False
    st.session_state["aurum_session_lock_reason"] = None
    st.session_state.pop("aurum_last_activity_at", None)
    st.session_state.pop("aurum_last_browser_event_id", None)
    st.session_state["aurum_auth_mode"] = "Connexion"


def unlock_local_session(password: str) -> bool:
    """Unlock ONLY after the saved account's current password is verified."""
    user = st.session_state.get("aurum_local_user")
    if not st.session_state.get("aurum_session_locked") or not isinstance(user, dict):
        return False

    saved_email = user.get("email")
    if not isinstance(saved_email, str) or not saved_email.strip() or not password:
        return False

    # Imported here to keep the session guard independent of the login UI
    # and reuse the existing Argon2id + verified-user login flow.
    from frontend.auth.login import authenticate_user

    authenticated = authenticate_user(email=saved_email, password=password)
    if not isinstance(authenticated, dict):
        return False
    returned_email = authenticated.get("email", "")
    if not isinstance(returned_email, str) or (
        returned_email.strip().casefold() != saved_email.strip().casefold()
    ):
        return False

    start_local_session(authenticated)
    return True


def session_expired(now: float | None = None) -> bool:
    """Determine expiry for an unlocked, authenticated local session."""
    if st.session_state.get("aurum_local_user") is None:
        return False
    if st.session_state.get("aurum_session_locked", False):
        return False

    last_activity_at = st.session_state.get("aurum_last_activity_at")
    if not isinstance(last_activity_at, (int, float)):
        return True

    current = time.monotonic() if now is None else now
    return current - last_activity_at >= IDLE_TIMEOUT_SECONDS


def check_local_session() -> bool:
    """Never allow a locked session to access protected pages."""
    if session_expired():
        lock_local_session()
    return (
        st.session_state.get("aurum_local_user") is not None
        and not st.session_state.get("aurum_session_locked", False)
    )


@st.fragment(run_every=WATCHDOG_INTERVAL_SECONDS)
def render_idle_watchdog() -> None:
    """Check timeouts even with no widget clicks, and accept user events."""
    if not check_local_session():
        st.rerun()  # Refresh the full page to show the password-only lock screen.

    event = _browser_idle(
        timeout_ms=IDLE_TIMEOUT_SECONDS * 1000,
        key=(
            "aurum_browser_idle_activity_"
            f"{st.session_state.get('aurum_idle_monitor_generation', 0)}"
        ),
        default=None,
    )

    # The component only reports activity or that the client timer expired.
    # Do not count the automatic fragment reruns as an activity signal.
    if isinstance(event, dict):
        event_id = event.get("id")
        if (
            isinstance(event_id, str)
            and event_id
            and event_id != st.session_state.get("aurum_last_browser_event_id")
        ):
            st.session_state["aurum_last_browser_event_id"] = event_id
            if event.get("type") == "activity":
                # Safety: a browser event cannot revive an already-expired session.
                if not check_local_session():
                    st.rerun()
                st.session_state["aurum_last_activity_at"] = time.monotonic()
            elif event.get("type") == "idle":
                lock_local_session()
                st.rerun()
            elif event.get("type") == "observer_error":
                st.warning(
                    "Détection d’activité navigateur indisponible. "
                    "La session expirera 5 minutes après la dernière "
                    "activité enregistrée."
                )


def require_local_session() -> dict:
    """Use at the top of each protected page after st.set_page_config()."""
    if not check_local_session():
        st.error("Accès protégé. Connectez-vous via AURUM-AI Secure Access.")
        st.stop()
    render_idle_watchdog()
    return st.session_state["aurum_local_user"]
