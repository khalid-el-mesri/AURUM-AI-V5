"""Lightweight tests of the additive desktop preview launcher."""
from pathlib import Path

import desktop_launcher as launcher


def test_command_uses_active_python_and_loopback_only():
    args = launcher.streamlit_command(Path("frontend/app.py"), 9523)
    assert args[0] == launcher.sys.executable
    assert "--server.address=127.0.0.1" in args
    assert "--server.port=9523" in args
    assert "--server.headless=true" in args
    assert "--server.enableXsrfProtection=true" in args
    assert "--server.enableCORS=true" in args
    assert not any("0.0.0.0" in arg for arg in args)


def test_command_uses_existing_app_entrypoint():
    args = launcher.streamlit_command(Path("frontend/app.py"), 9523)
    assert args[1:4] == ["-m", "streamlit", "run"]
    assert args[4].endswith("frontend/app.py") or args[4].endswith("frontend\\app.py")


def test_bad_ports_are_rejected():
    import pytest
    for value in (0, -1, 65536, "9523"):
        with pytest.raises(ValueError):
            launcher.streamlit_command(Path("frontend/app.py"), value)


def test_ephemeral_port_is_valid():
    port = launcher.choose_local_port()
    assert 1 <= port <= 65535


def test_preview_is_not_a_frozen_executable_launcher():
    assert launcher.project_root().name
    assert launcher.HOST == "127.0.0.1"
