"""Login/Signup/Password-only Unlock screen; only loaded when access denied."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[1] / "auth" / "login_ui.py"), run_name="__main__")
