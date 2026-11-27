AURUM-AI V5 - Phase 21.3: Argon2 authentication inclusion in Windows EXE

Issue confirmed by Windows screenshot:
  ModuleNotFoundError: No module named 'argon2'
  frontend/auth/security.py imports from argon2 import PasswordHasher

Cause:
  Streamlit dynamically runs the app's authentication pages from source
  files bundled as DATA. PyInstaller's Analysis cannot follow those imports.

Contents:
  tools/windows_desktop/build_windows_exe.py
  tests/test_phase21_argon2_packaging.py

Update is LIMITED to the optimized build generator (no frontend/auth source
files changed; frozen_entry.py from Phase 21.2 remains untouched).

From AURUM-AI root in PowerShell, with .venv active:

  Copy-Item .\tools\windows_desktop\build_windows_exe.py .\tools\windows_desktop\build_windows_exe_before_phase21_3.py.bak
  Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\AURUM_AI_V5_Phase21_3_Argon2_Fix.zip" -DestinationPath . -Force

  python -c "from argon2 import PasswordHasher; from _argon2_cffi_bindings import ffi; print('Argon2 backend OK')"
  # If missing in .venv: python -m pip install argon2-cffi

  python -m pytest tests/test_phase21_argon2_packaging.py tests/test_windows_packaging_phase21.py tests/test_phase21_streamlit_devmode_fix.py -q
  python tools/windows_desktop/build_windows_exe.py --check
  python tools/windows_desktop/build_windows_exe.py --build --console

When it finishes:
  & '.\dist\AURUM-AI V5\AURUM-AI V5.exe'

If another error appears in the Streamlit panel or a launch dialog, capture its
traceback (and packaged_desktop.log) before changing the app again.

Scientific scope: research/historical simulation, validation-only presentation.
Deep learning training frameworks remain excluded in this slim Windows preview.
NEVER distribute user data/credentials automatically. The built EXE is not
an installer and still requires a proper end-to-end security review.
