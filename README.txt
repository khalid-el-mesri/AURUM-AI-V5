AURUM-AI — Password-only session unlock
======================================

Your feedback: after 5 minutes idle, keep the signed-in account and show a
password-only Session Locked screen (no e-mail entry, login/signup/OTP tabs).

This ZIP replaces ONLY TWO files in your existing AURUM-AI project:

  frontend/auth/login_ui.py
  frontend/auth/idle_guard.py

KEEP the already installed frontend/auth/idle_component/index.html.
The account SQLite database, registration and OTP modules stay untouched.

INSTALL (PowerShell, from AURUM-AI project root)
  1. Stop Streamlit: Ctrl+C.
  2. Back up your latest files:
       Copy-Item .\frontend\auth\login_ui.py .\frontend\auth\login_ui_before_unlock.py
       Copy-Item .\frontend\auth\idle_guard.py .\frontend\auth\idle_guard_before_unlock.py
  3. Download this ZIP to Downloads and extract it into the project root:
       Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\AURUM_AI_Password_Unlock.zip" -DestinationPath . -Force
  4. For a quick local check, temporarily change in idle_guard.py:
       IDLE_TIMEOUT_SECONDS = 300
     to:
       IDLE_TIMEOUT_SECONDS = 20
  5. Start the app:
       python -m streamlit run frontend/auth/login_ui.py
  6. Log in, stay idle for 20-25 seconds. The lock screen should show
     ONLY the remembered account name/e-mail (read-only), password field,
     and "Déverrouiller ma session" button.
  7. Wrong password should keep the session locked; correct password should
     restore the existing session. Restore IDLE_TIMEOUT_SECONDS = 300
     after confirming the test.

SECURITY
  - Expired accounts are not authorized to view protected pages.
  - Unlock validates the current password for the same remembered account
    using the existing login verifier. It does not cache the password.
  - Locked session identity exists only in the current Streamlit session.
    Browser refresh, server restart or expired Streamlit WebSocket may
    require a normal login with e-mail again; the password-only screen is
    not a persistent cross-browser sign-in feature.
  - When adding the Dashboard, each page must call require_local_session()
    before displaying or processing protected data.
  - Session lock is not a substitute for a full authentication security
    review. Gmail OTP was not proven to work by the earlier screenshot.
