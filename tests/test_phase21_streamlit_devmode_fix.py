"""Regression checks for Streamlit's PyInstaller developmentMode detection.

All tests are offline and do not start Streamlit or require Windows.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest

ENTRY = Path(__file__).resolve().parents[1] / 'tools' / 'windows_desktop' / 'frozen_entry.py'


def load_entry():
    spec = importlib.util.spec_from_file_location('aurum_devmode_test_entry', ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_server_cli_disables_development_mode():
    mod = load_entry()
    opts = mod.streamlit_arguments(Path('frontend/app.py'), 8509)
    assert opts.count('--global.developmentMode=false') == 1
    assert '--server.port=8509' in opts


def test_localhost_only_and_security_not_relaxed():
    mod = load_entry()
    opts = mod.streamlit_arguments(Path('frontend/app.py'), 8509)
    assert '--server.address=127.0.0.1' in opts
    assert '--server.enableCORS=true' in opts
    assert '--server.enableXsrfProtection=true' in opts
    assert '--browser.gatherUsageStats=false' in opts


def test_no_interference_with_desktop_gui_flag():
    mod = load_entry()
    assert mod.executable_command(8509)[-2:] == ['--streamlit-server', '8509']


@pytest.mark.parametrize('bad_port', [-2, 0, 65536])
def test_invalid_port_rejected(bad_port):
    mod = load_entry()
    with pytest.raises(ValueError):
        mod.executable_command(bad_port)


def test_server_forces_false_for_streamlit_in_frozen_entry(tmp_path, monkeypatch):
    mod = load_entry()
    (tmp_path / 'frontend').mkdir()
    (tmp_path / 'frontend' / 'app.py').write_text('import streamlit as st\n', encoding='utf-8')
    monkeypatch.setattr(mod, 'root_dir', lambda: tmp_path)
    monkeypatch.setenv('STREAMLIT_GLOBAL_DEVELOPMENT_MODE', 'true')
    monkeypatch.setattr(os, 'chdir', Mock())
    original_sys_path = list(sys.path)
    original_sys_argv = list(sys.argv)
    calls = []

    def fake_main():
        calls.append((os.environ['STREAMLIT_GLOBAL_DEVELOPMENT_MODE'], list(sys.argv)))

    stub_streamlit = types.ModuleType('streamlit')
    stub_web = types.ModuleType('streamlit.web')
    stub_cli = types.ModuleType('streamlit.web.cli')
    stub_cli.main = fake_main
    stub_web.cli = stub_cli
    stub_streamlit.web = stub_web
    monkeypatch.setitem(sys.modules, 'streamlit', stub_streamlit)
    monkeypatch.setitem(sys.modules, 'streamlit.web', stub_web)
    monkeypatch.setitem(sys.modules, 'streamlit.web.cli', stub_cli)
    try:
        assert mod.serve(8509) == 0
        assert calls[0][0] == 'false'
        assert '--global.developmentMode=false' in calls[0][1]
    finally:
        sys.path[:] = original_sys_path
        sys.argv[:] = original_sys_argv


def test_always_sets_explicit_dev_flag_even_without_environment_override(monkeypatch):
    mod = load_entry()
    monkeypatch.delenv('STREAMLIT_GLOBAL_DEVELOPMENT_MODE', raising=False)
    assert '--global.developmentMode=false' in mod.streamlit_arguments(Path('frontend/app.py'), 8510)
