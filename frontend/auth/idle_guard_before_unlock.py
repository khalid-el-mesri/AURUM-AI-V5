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

IDLE_TIMEOUT_SECONDS = 20 # Five minutes. Use 20 only for a quick local test.
WATCHDOG_INTERVAL_SECONDS = 3

_COMPONENT_DIRECTORY = Path(__file__).resolve().parent / "idle_component"
_browser_idle = components.declare_component(
    "aurum_idle_activity_v1", path=str(_COMPONENT_DIRECTORY)
)


def start_local_session(user: dict) -> None:
    """Initialize the authenticated session. Invoke only after valid login."""
    st.session_state["aurum_local_user"] = user
    st.session_state["aurum_last_activity_at"] = time.monotonic()
    st.session_state["aurum_last_browser_event_id"] = None
    st.session_state["aurum_session_lock_reason"] = None


def end_local_session(*, idle: bool = False) -> None:
    """Revoke local access on logout or inactivity."""
    st.session_state["aurum_local_user"] = None
    st.session_state.pop("aurum_last_activity_at", None)
    st.session_state.pop("aurum_last_browser_event_id", None)
    timeout_label = (
        "5 minutes"
        if IDLE_TIMEOUT_SECONDS == 300
        else f"{IDLE_TIMEOUT_SECONDS} secondes"
    )
    st.session_state["aurum_session_lock_reason"] = (
        f"🔒 Session verrouillée après {timeout_label} d’inactivité. "
        "Reconnectez-vous pour continuer."
        if idle
        else None
    )
    st.session_state["aurum_auth_mode"] = "Connexion"


def session_expired(now: float | None = None) -> bool:
    """Fail closed if an authenticated session has no valid activity time."""
    if st.session_state.get("aurum_local_user") is None:
        return False

    last_activity_at = st.session_state.get("aurum_last_activity_at")
    if not isinstance(last_activity_at, (int, float)):
        return True

    current = time.monotonic() if now is None else now
    return current - last_activity_at >= IDLE_TIMEOUT_SECONDS


def check_local_session() -> bool:
    """Enforce expiry at every authenticated Python entrypoint."""
    if session_expired():
        end_local_session(idle=True)
        return False
    return st.session_state.get("aurum_local_user") is not None


@st.fragment(run_every=WATCHDOG_INTERVAL_SECONDS)
def render_idle_watchdog() -> None:
    """Check timeouts even with no widget clicks, and accept user events."""
    if not check_local_session():
        st.rerun()  # Refresh the full page to show the locked/login screen.

    event = _browser_idle(
        timeout_ms=IDLE_TIMEOUT_SECONDS * 1000,
        key="aurum_browser_idle_activity",
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
                end_local_session(idle=True)
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
