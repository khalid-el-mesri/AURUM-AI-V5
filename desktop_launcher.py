"""AURUM-AI V5: Windows desktop preview (not a packaged EXE).

Launches the existing Streamlit application on a loopback-only ephemeral port
inside a native desktop window provided by pywebview. This is intentionally
additive and does not modify the existing app, training data, or artifacts.

Usage (from the AURUM-AI project root):
    python -m pip install pywebview
    python desktop_launcher.py
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


HOST = "127.0.0.1"
WINDOW_TITLE = "AURUM-AI V5 | Investment Intelligence"


def project_root() -> Path:
    return Path(__file__).resolve().parent


def choose_local_port() -> int:
    """Request a currently free port on loopback (a race remains possible)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return int(sock.getsockname()[1])


def streamlit_command(app_file: Path, port: int) -> list[str]:
    """Use the active .venv interpreter; never expose Streamlit on the LAN."""
    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("Port local invalide")
    return [
        sys.executable,
        "-m", "streamlit", "run", str(app_file),
        f"--server.address={HOST}",
        f"--server.port={port}",
        "--server.headless=true",
        "--server.fileWatcherType=none",
        "--server.enableCORS=true",
        "--server.enableXsrfProtection=true",
        f"--browser.serverAddress={HOST}",
        f"--browser.serverPort={port}",
        "--browser.gatherUsageStats=false",
    ]


def server_ready(port: int) -> bool:
    """Use the Streamlit health endpoint, not the application's login page."""
    try:
        with urlopen(f"http://{HOST}:{port}/_stcore/health", timeout=1) as resp:
            return resp.status == 200
    except (HTTPError, URLError, OSError, TimeoutError):
        return False


def wait_for_streamlit(proc: subprocess.Popen, port: int, timeout_seconds: float = 40) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Le serveur s'est arrêté (code {proc.returncode}).")
        if server_ready(port):
            return
        time.sleep(0.3)
    raise RuntimeError("Le serveur Streamlit local n'est pas devenu disponible.")


def log_location() -> Path:
    """Keep diagnostic output out of the immutable research data folders."""
    base = os.environ.get("LOCALAPPDATA")
    logs_dir = (Path(base) if base else Path.home()) / "AURUM-AI" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / "desktop_streamlit.log"


def stop_child(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def main() -> int:
    if sys.platform != "win32":
        print("Cette préversion de l'interface desktop est destinée à Windows.", file=sys.stderr)
        return 2
    app_file = project_root() / "frontend" / "app.py"
    if not app_file.is_file():
        print(f"Entrée Streamlit absente : {app_file}", file=sys.stderr)
        return 2
    try:
        import webview  # Imported late: unit tests don't require GUI dependencies.
    except ImportError:
        print("Installez pywebview dans .venv : python -m pip install pywebview", file=sys.stderr)
        return 2

    port = choose_local_port()
    url = f"http://{HOST}:{port}"
    logfile = log_location()
    proc: subprocess.Popen | None = None
    try:
        with logfile.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                streamlit_command(app_file, port),
                cwd=str(project_root()),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            wait_for_streamlit(proc, port)
            # The login gate is still enforced by frontend/app.py.
            webview.settings["ALLOW_DOWNLOADS"] = True
            webview.create_window(
                WINDOW_TITLE,
                url=url,
                width=1440,
                height=900,
                min_size=(1000, 650),
                background_color="#081525",
            )
            # AURUM-AI branded desktop icon (supported by recent pywebview).
            import inspect
            icon_path = project_root() / "assets" / "branding" / "aurum_ai.ico"
            kwargs = {"debug": False}
            if icon_path.is_file() and "icon" in inspect.signature(webview.start).parameters:
                kwargs["icon"] = str(icon_path)
            webview.start(**kwargs)
    except Exception as exc:
        print(f"Échec du lancement desktop : {exc}", file=sys.stderr)
        print(f"Journal local : {logfile}", file=sys.stderr)
        return 1
    finally:
        stop_child(proc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
