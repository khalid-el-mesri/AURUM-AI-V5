"""Offline structural tests, no Inno Setup compiler required."""
from pathlib import Path
import importlib.util
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

pre = load("phase23_preflight", "tools/windows_desktop/installer_preflight_phase23.py")
builder = load("phase23_builder", "tools/windows_desktop/build_installer_phase23.py")

def seed(tmp_path, with_internal=True):
    p = tmp_path / "dist/AURUM-AI V5"
    p.mkdir(parents=True)
    (p / "AURUM-AI V5.exe").write_bytes(b"test")
    if with_internal:
        (p / "_internal").mkdir()
        (p / "_internal/x.bin").write_bytes(b"binary")
    (tmp_path / "assets/branding").mkdir(parents=True)
    (tmp_path / "assets/branding/aurum_ai.ico").write_bytes(b"icon")
    (tmp_path / "installer").mkdir()
    (tmp_path / "installer/AURUM_AI_V5.iss").write_text(
        'Source: "..\\dist\\AURUM-AI V5\\*"; Excludes: "*.sqlite3"; Flags: recursesubdirs createallsubdirs\n'
        'PrivilegesRequired=lowest\nDefaultDirName={localappdata}\\Programs\\AURUM-AI V5', encoding="utf-8")
    return p

def test_expected_installer_and_shortcut():
    iss = (ROOT / "installer/AURUM_AI_V5.iss").read_text()
    assert "{autoprograms}" in iss
    assert "{autodesktop}" in iss
    assert "UninstallDisplayIcon" in iss
    assert "SetupIconFile" in iss

def test_auth_and_env_files_excluded():
    iss = (ROOT / "installer/AURUM_AI_V5.iss").read_text()
    for p in ("*.sqlite3", "*.db", ".env", "secrets.toml"):
        assert p in iss

def test_install_only_dist_directory():
    iss = (ROOT / "installer/AURUM_AI_V5.iss").read_text()
    assert iss.count('Source: "..\\dist\\AURUM-AI V5\\*"') == 1

def test_uninstaller_preserves_appdata():
    iss = (ROOT / "installer/AURUM_AI_V5.iss").read_text()
    assert "[UninstallDelete]" not in iss
    assert "PrivilegesRequired=lowest" in iss

def test_preflight_success(tmp_path):
    seed(tmp_path)
    result = pre.audit(tmp_path)
    assert result.ok
    assert result.file_count == 2

def test_preflight_missing_exe(tmp_path):
    p = seed(tmp_path)
    (p / "AURUM-AI V5.exe").unlink()
    assert not pre.audit(tmp_path).ok

def test_preflight_missing_internal(tmp_path):
    seed(tmp_path, with_internal=False)
    assert not pre.audit(tmp_path).ok

def test_preflight_secret_belongs_to_exclude_set(tmp_path):
    p = seed(tmp_path)
    (p / "_internal/users.sqlite3").write_bytes(b"example")
    (p / "_internal/.env").write_text("FAKE=FAKE")
    res = pre.audit(tmp_path)
    assert res.ok
    assert len(res.excluded) == 2

def test_helper_recognizes_case():
    assert pre.is_excluded(Path("abc/USERS.SQLITE3"))
    assert pre.is_excluded(Path("abc/Secrets.toml"))
    assert not pre.is_excluded(Path("abc/certifi.pem"))

def test_build_runs_iscc_and_checks_output(tmp_path, monkeypatch):
    seed(tmp_path)
    seen = []
    def fake_run(args, cwd, check):
        seen.append((args,cwd,check))
        output = tmp_path/"dist_installer/AURUM-AI-V5-Setup-0.9.0.exe"
        output.parent.mkdir()
        output.write_bytes(b"setup")
    monkeypatch.setattr(builder.subprocess, "run", fake_run)
    result = builder.build(tmp_path, Path("C:/ISCC.exe"))
    assert result.is_file() and len(seen)==1

def test_no_innosetup_binary_included():
    assert not any(p.name.lower()=="iscc.exe" for p in ROOT.rglob("*"))
