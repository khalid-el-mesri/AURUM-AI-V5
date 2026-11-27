from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / 'tools' / 'windows_desktop' / 'frozen_entry.py'
BUILDER = ROOT / 'tools' / 'windows_desktop' / 'build_windows_exe.py'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


entry = load(ENTRY, 'aurum_frozen_entry_test')
builder = load(BUILDER, 'aurum_builder_test')


def test_source_syntax():
    ast.parse(ENTRY.read_text(encoding='utf-8'))
    ast.parse(BUILDER.read_text(encoding='utf-8'))


def test_port_arg_validation():
    assert '--streamlit-server' in entry.executable_command(8501)
    for p in (-1, 0, 65536):
        try:
            entry.executable_command(p)
        except ValueError:
            pass
        else:
            assert False


def test_arguments_loopback():
    args = entry.streamlit_arguments(Path('frontend/app.py'), 12345)
    assert '--server.address=127.0.0.1' in args
    assert '--server.enableXsrfProtection=true' in args
    assert '--server.enableCORS=true' in args
    assert '--server.port=12345' in args


def test_pyinstaller_child_dispatch(monkeypatch):
    monkeypatch.setattr(entry.sys, 'frozen', True, raising=False)
    assert entry.executable_command(8001) == [entry.sys.executable, '--streamlit-server', '8001']


def test_manifest_excludes_credentials(tmp_path):
    folder = tmp_path / 'frontend'
    folder.mkdir()
    (folder / 'app.py').write_text('pass')
    (folder / 'secrets.toml').write_text('token=123')
    (folder / 'notes.txt').write_text('hello')
    (folder / '.env').write_text('password=secret')
    (folder / 'cert.pem').write_text('private')
    (folder / 'users.db').write_text('private')
    (folder / 'password_unlock.py').write_text('pass')
    cache = folder / '__pycache__'
    cache.mkdir()
    (cache / 'cached.py').write_text('pass')
    names = {Path(f).name for f, _ in builder.enumerate_runtime_data(tmp_path)}
    assert names == {'app.py', 'notes.txt', 'password_unlock.py'}


def test_import_scan(tmp_path):
    p = tmp_path / 'frontend'
    p.mkdir()
    (p / 'app.py').write_text('import pandas as pd\nfrom reportlab.pdfgen import canvas\n')
    mods = builder.scan_imports(tmp_path, builder.enumerate_runtime_data(tmp_path))
    assert 'pandas' in mods and 'reportlab' in mods


def test_spec_has_ico_and_mode(tmp_path):
    spec = builder.make_spec(tmp_path, [], ['streamlit', 'webview'], False)
    assert 'aurum_ai.ico' in spec
    assert "console=False" in spec
    assert 'COLLECT' in spec


def test_entry_server_cli_main_invocation(monkeypatch, tmp_path):
    import types
    (tmp_path / 'frontend').mkdir()
    (tmp_path / 'frontend' / 'app.py').write_text('pass')
    monkeypatch.setattr(entry, 'root_dir', lambda: tmp_path)
    monkeypatch.setattr(entry.os, 'chdir', lambda path: None)
    sentinel = []
    fake = types.SimpleNamespace(main=lambda: sentinel.append(tuple(sys.argv)))
    monkeypatch.setitem(sys.modules, 'streamlit.web.cli', fake)
    monkeypatch.setitem(sys.modules, 'streamlit.web', types.SimpleNamespace(cli=fake))
    monkeypatch.setitem(sys.modules, 'streamlit', types.SimpleNamespace())
    old_args = sys.argv
    try:
        assert entry.serve(8501) == 0
        assert sentinel and sentinel[0][0:2] == ('streamlit', 'run')
    finally:
        sys.argv = old_args
