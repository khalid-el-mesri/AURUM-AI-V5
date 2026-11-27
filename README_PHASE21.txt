AURUM-AI V5 - Phase 21: Windows PyInstaller EXPERIMENTAL BUILD
==============================================================

THIS IS A PRE-RELEASE BUILD TOOL, NOT A FULL INSTALLER.
Build on Windows ONLY and validate the resulting bundle on the user's PC.
It does NOT overwrite frontend/app.py or desktop_launcher.py.

Files:
  tools/windows_desktop/frozen_entry.py
  tools/windows_desktop/build_windows_exe.py
  tests/test_windows_packaging_phase21.py

1. In AURUM-AI project root with .venv active:
   python -m pytest tests/test_windows_packaging_phase21.py -q
   python tools/windows_desktop/build_windows_exe.py --check

2. To compile the debug/console build first:
   python tools/windows_desktop/build_windows_exe.py --build --console

3. Launch from the PROJECT ROOT:
   & '.\dist\AURUM-AI V5\AURUM-AI V5.exe'

   The executable runs a separate copy of itself in '--streamlit-server'
   mode to start a local-only Streamlit server; Python is not invoked via
   sys.executable -m streamlit (which would recurse into the frozen EXE).

4. If it launches and all critical pages, exports and login work, rebuild
   WITHOUT a console:
   python tools/windows_desktop/build_windows_exe.py --build

5. Do not move the EXE out of dist/AURUM-AI V5. The companion _internal
   directory and its data are required. To test portability, copy the
   ENTIRE 'AURUM-AI V5' folder to a different Windows machine.

6. Taskbar icon comes from PyInstaller embedding:
      assets/branding/aurum_ai.ico
   Windows taskbar icon cache may need a restart/unpin-repin if old icon
   persists. See pywebview Windows icon guidance.

Diagnostics:
   %LOCALAPPDATA%\AURUM-AI\logs\packaged_desktop.log

Limitations:
 - Windows-only build must be executed locally; a successful PyInstaller
   run is not sufficient to claim every page works.
 - Manifest copies whitelisted files from source, assets, artifacts,
   reports and data; it excludes common passwords/private certificates,
   accounts databases, .env, test folders and virtual env. Audit files
   before sharing; data could still be proprietary/sensitive.
 - Login configuration and any user database need a proper first-run
   writable data path under LOCALAPPDATA before making an installer.
 - Pages whose data or dynamic imports are missing may need another
   packaging iteration based on the real Windows logs.
 - Very large XOM historical datasets may make the distribution huge.
 - This package does not run any trades, training, data downloads or
   write to the RAW dataset. It bundles historical research output.
 - A Windows installation package and code signing are later steps.
