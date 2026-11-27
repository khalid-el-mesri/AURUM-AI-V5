"""Experimental AURUM-AI V5 Windows bundle entry point.

A single frozen executable runs in two modes:
  no args                -> starts the local server and pywebview window
  --streamlit-server PORT -> serves the existing authenticated Streamlit app

Never change immutable source market data or model artifacts here.
"""
from __future__ import annotations

import multiprocessing
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

HOST = '127.0.0.1'
TITLE = 'AURUM-AI V5 | Investment Intelligence'


def root_dir() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS).resolve()
    return Path(__file__).resolve().parents[2]


def executable_command(port: int) -> list[str]:
    if not isinstance(port, int) or not 0 < port < 65536:
        raise ValueError('Invalid local port')
    if getattr(sys, 'frozen', False):
        return [sys.executable, '--streamlit-server', str(port)]
    return [sys.executable, str(Path(__file__).resolve()), '--streamlit-server', str(port)]


def streamlit_arguments(app: Path, port: int) -> list[str]:
    return [
        'streamlit', 'run', str(app),
        '--global.developmentMode=false',
        f'--server.address={HOST}', f'--server.port={port}',
        '--server.headless=true', '--server.fileWatcherType=none',
        '--server.enableCORS=true', '--server.enableXsrfProtection=true',
        f'--browser.serverAddress={HOST}', f'--browser.serverPort={port}',
        '--browser.gatherUsageStats=false',
    ]


def serve(port: int) -> int:
    root = root_dir()
    app = root / 'frontend' / 'app.py'
    if not app.is_file():
        raise FileNotFoundError(f'Application Streamlit manquante: {app}')
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # A PyInstaller bundle is not stored under site-packages, so Streamlit
    # may incorrectly default to developmentMode=True. That conflicts with
    # specifying the localhost server.port used by the desktop launcher.
    # Configure only this child server process, not the user's system-wide
    # Streamlit installation. The explicit CLI option takes precedence.
    os.environ['STREAMLIT_GLOBAL_DEVELOPMENT_MODE'] = 'false'
    # Do NOT invoke a bundled executable with '-m streamlit': sys.executable
    # points to the PyInstaller bootloader when frozen.
    from streamlit.web import cli as streamlit_cli
    sys.argv = streamlit_arguments(app, port)
    streamlit_cli.main()
    return 0


def find_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return int(sock.getsockname()[1])


def healthy(port: int) -> bool:
    try:
        with urlopen(f'http://{HOST}:{port}/_stcore/health', timeout=1) as response:
            return response.status == 200
    except (URLError, HTTPError, OSError, TimeoutError):
        return False


def await_server(proc: subprocess.Popen, port: int, timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f'Streamlit terminated: exit code {proc.returncode}')
        if healthy(port):
            return
        time.sleep(0.3)
    raise TimeoutError('Local Streamlit server did not respond')


def log_file() -> Path:
    base = Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) / 'AURUM-AI' / 'logs'
    base.mkdir(parents=True, exist_ok=True)
    return base / 'packaged_desktop.log'


def shutdown(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=8)


def windows_app_id() -> None:
    if sys.platform == 'win32':
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('AURUMAI.V5.Desktop')
        except (AttributeError, OSError):
            pass


def launch() -> int:
    if sys.platform != 'win32':
        raise RuntimeError('The desktop GUI build must be tested on Windows')
    root = root_dir()
    if not (root / 'frontend' / 'app.py').is_file():
        raise FileNotFoundError(f'frontend/app.py missing under {root}')
    import webview
    windows_app_id()
    port = find_port()
    logfile = log_file()
    child = None
    try:
        with logfile.open('w', encoding='utf-8') as log:
            child = subprocess.Popen(
                executable_command(port), cwd=str(root),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
            )
            await_server(child, port)
            webview.settings["ALLOW_DOWNLOADS"] = True
            webview.create_window(
                TITLE, url=f'http://{HOST}:{port}',
                width=1440, height=900, min_size=(1000, 650),
                background_color='#081525',
            )
            # On Windows the taskbar icon comes from the embedded EXE .ico.
            webview.start(debug=False)
    finally:
        shutdown(child)
    return 0


def main(argv: list[str] | None = None) -> int:
    multiprocessing.freeze_support()
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 2 and argv[0] == '--streamlit-server':
        return serve(int(argv[1]))
    if argv:
        raise ValueError('Unexpected arguments')
    try:
        return launch()
    except Exception as e:
        error_log = log_file()
        with error_log.open('a', encoding='utf-8') as f:
            f.write(f'\nLAUNCH ERROR: {type(e).__name__}: {e}\n')
        # A windowed EXE has no reliable console; show the error to the user.
        if sys.platform == 'win32':
            try:
                import ctypes
                ctypes.windll.user32.MessageBoxW(
                    None, f'{e}\n\nJournal: {error_log}',
                    'AURUM-AI V5 — démarrage impossible', 0x10)
            except Exception:
                pass
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
