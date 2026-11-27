"""Regression checks for Argon2 imports hidden by Streamlit's runpy pages.

Offline checks. Windows PyInstaller executable still requires end-to-end testing.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

BUILDER = Path(__file__).resolve().parents[1] / 'tools' / 'windows_desktop' / 'build_windows_exe.py'
module_spec = importlib.util.spec_from_file_location('aurum_argon2_builder_test', BUILDER)
builder = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(builder)


def make_sample(tmp_path):
    (tmp_path / 'frontend').mkdir()
    (tmp_path / 'frontend' / 'app.py').write_text('import streamlit as st\n', encoding='utf-8')
    return tmp_path


def test_build_script_has_valid_python_syntax():
    ast.parse(BUILDER.read_text(encoding='utf-8'))


def test_argon2_imported_as_hidden_module(tmp_path):
    spec = builder.make_spec(make_sample(tmp_path), [], [], console=True)
    assert "('argon2', '_argon2_cffi_bindings')" in spec
    assert "hiddenimports += ['argon2', '_argon2_cffi_bindings', 'cffi', 'frontend.auth.security']" in spec
    assert "'frontend.auth'" in spec
    assert "collect_submodules(" in spec


def test_generated_spec_compiles(tmp_path):
    text = builder.make_spec(make_sample(tmp_path), [], [], console=True)
    compile(text, '<generated pyinstaller spec>', 'exec')


def test_backend_includes_native_libraries(tmp_path):
    spec = builder.make_spec(make_sample(tmp_path), [], [], console=True)
    assert "binaries += collect_dynamic_libs('_argon2_cffi_bindings')" in spec
    assert 'collect_dynamic_libs' in spec


def test_includes_argon2_distribution_metadata(tmp_path):
    spec = builder.make_spec(make_sample(tmp_path), [], [], console=False)
    for name in ('argon2-cffi', 'argon2-cffi-bindings', 'cffi'):
        assert repr(name) in spec


def test_preserves_optimized_compilation(tmp_path):
    spec = builder.make_spec(make_sample(tmp_path), [], [], console=False)
    assert 'collect_all(' not in spec
    assert "'torch', 'torchvision'" in spec
    assert 'COLLECT(exe,' in spec
    assert "icon=" in spec


def test_excludes_test_subpackages(tmp_path):
    spec = builder.make_spec(make_sample(tmp_path), [], [], console=False)
    assert "part in ('test', 'tests', 'testing')" in spec


def test_does_not_package_account_database_or_keys(tmp_path):
    root = make_sample(tmp_path)
    auth = root / 'frontend' / 'auth'
    auth.mkdir()
    for item in ('secrets.toml', 'accounts.db', 'token.json', 'private.pem'):
        (auth / item).write_text('private', encoding='utf-8')
    (auth / 'security.py').write_text('from argon2 import PasswordHasher\n', encoding='utf-8')
    files = {Path(source).name for source, _ in builder.enumerate_runtime_data(root)}
    assert 'security.py' in files
    assert all(name not in files for name in ('secrets.toml', 'accounts.db', 'token.json', 'private.pem'))


def test_auth_modules_must_exist_before_build(tmp_path, monkeypatch):
    root = make_sample(tmp_path)
    (root / 'assets' / 'branding').mkdir(parents=True)
    (root / 'assets' / 'branding' / 'aurum_ai.ico').write_bytes(b'ico-placeholder')
    entry = root / 'tools' / 'windows_desktop'
    entry.mkdir(parents=True)
    (entry / 'frozen_entry.py').write_text('pass\n', encoding='utf-8')
    monkeypatch.setattr(builder, 'pkg_present', lambda name: name != 'argon2')
    with pytest.raises(RuntimeError, match='argon2-cffi'):
        builder.prepare(root, check_only=True, console=True)


def test_password_hasher_round_trip_in_test_environment():
    # This verifies the local backend, not a Windows bundle, and uses a dummy password.
    argon2 = pytest.importorskip('argon2')
    ph = argon2.PasswordHasher(time_cost=1, memory_cost=8192, parallelism=1)
    generated = ph.hash('aurum-test-only-not-a-real-password')
    assert ph.verify(generated, 'aurum-test-only-not-a-real-password')
    assert generated.startswith('$argon2id$')
