"""Unit tests on synthetic fixtures only. No real accounts or EXE needed."""
from __future__ import annotations
import importlib.util
import os
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'tools/windows_desktop/audit_distribution_phase25.py'
spec = importlib.util.spec_from_file_location('aurum_phase25_audit', SCRIPT)
mod = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


@pytest.fixture
def sample(tmp_path, monkeypatch):
    root = tmp_path/'AURUM-AI'
    dist = root/'dist/AURUM-AI V5'
    (dist/'_internal/certifi').mkdir(parents=True)
    (dist/'_internal/frontend/auth').mkdir(parents=True)
    (dist/'AURUM-AI V5.exe').write_bytes(b'frozen-demo')
    (dist/'_internal/certifi/cacert.pem').write_text('-----BEGIN CERTIFICATE-----\nPUBLIC\n')
    (root/'dist_installer').mkdir()
    (root/'dist_installer/AURUM-AI-V5-Setup-0.9.0.exe').write_bytes(b'dummy-installer')
    (root/'installer').mkdir()
    (root/'installer/AURUM_AI_V5.iss').write_text(
        '[Setup]\nPrivilegesRequired=lowest\n'
        '[Files]\nSource: "..\\dist\\AURUM-AI V5\\*"; DestDir: "{app}"; '
        'Excludes: "*.sqlite3"; Flags: recursesubdirs createallsubdirs\n'
    )
    (root/'tools/windows_desktop').mkdir(parents=True)
    (root/'tools/windows_desktop/frozen_entry.py').write_text(
        "HOST = '127.0.0.1'\ncmd = f'--server.address={HOST}'\n"
    )
    (root/'frontend/auth').mkdir(parents=True)
    (root/'frontend/auth/security.py').write_text('from argon2 import PasswordHasher\n')
    installed = tmp_path/'installed'
    installed.mkdir()
    (installed/'AURUM-AI V5.exe').write_bytes(b'frozen-demo')
    local = tmp_path/'local'
    (local/'AURUM-AI/auth').mkdir(parents=True)
    (local/'AURUM-AI/auth/users.sqlite3').write_bytes(b'not a real database; audit must not read it')
    monkeypatch.setenv('LOCALAPPDATA', str(local))
    return root, dist, installed


def test_baseline_passes_with_nonblocking_warnings(sample):
    root, dist, installed = sample
    result = mod.audit(root, installed)
    assert not result.blockers
    assert any('identique' in msg for msg in result.good)


def test_public_certifi_ignored(sample):
    root, dist, installed = sample
    assert mod.sensitive_file(Path('_internal/certifi/cacert.pem')) is None
    assert not mod.audit(root, installed).blockers


@pytest.mark.parametrize('name', ['users.sqlite3', '.env', '.env.production', 'secrets.toml',
                                  'token.json', 'client_secret.key', 'identity.p12', 'id_rsa'])
def test_sensitive_file_blocks(sample, name):
    root, dist, installed = sample
    (dist/'_internal'/name).write_text('sentinel')
    result = mod.audit(root, installed)
    assert any(name in blocker for blocker in result.blockers)


def test_no_secret_values_are_printed(sample):
    root, dist, installed = sample
    (dist/'_internal/frontend/auth/email_service.py').write_text("SMTP_PASSWORD = 'SUPERSECRET_REALVALUE'\n")
    result = mod.audit(root, installed)
    joined = ' '.join(result.blockers)
    assert 'SUPERSECRET_REALVALUE' not in joined
    assert 'email_service.py' in joined


def test_literal_checker_ignores_env_read():
    assert not mod.literal_secret_assignment('import os\nSMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")\n')


def test_literal_checker_reports_hardcode_without_value():
    result = mod.literal_secret_assignment('PASSWORD = "some-user-entered-passphrase"\n')
    assert result == [(1, 'PASSWORD')]


def test_legacy_auth_backup_blocks(sample):
    root, dist, installed = sample
    (dist/'_internal/frontend/auth/login_ui_backup.py').write_text('')
    assert any('Ancienne copie Python' in issue for issue in mod.audit(root, installed).blockers)


def test_exe_difference_blocks(sample):
    root, dist, installed = sample
    (installed/'AURUM-AI V5.exe').write_bytes(b'an older build')
    assert any('differents' in msg for msg in mod.audit(root, installed).blockers)


def test_nonlocal_host_blocks(sample):
    root, dist, installed = sample
    (root/'tools/windows_desktop/frozen_entry.py').write_text("HOST = '0.0.0.0'\ncmd='--server.address=0.0.0.0'\n")
    assert any('HOST' in msg for msg in mod.audit(root, installed).blockers)


def test_missing_installer_blocks(sample):
    root, dist, installed = sample
    (root/'dist_installer/AURUM-AI-V5-Setup-0.9.0.exe').unlink()
    assert any('Installateur absent' in msg for msg in mod.audit(root, installed).blockers)


def test_uninstall_user_data_rule_blocks(sample):
    root, dist, installed = sample
    with (root/'installer/AURUM_AI_V5.iss').open('a') as stream:
        stream.write('[UninstallDelete]\nType: filesandordirs; Name: "{localappdata}\\AURUM-AI"\n')
    assert any('desinstallation suspecte' in msg for msg in mod.audit(root, installed).blockers)


def test_no_modification_of_persistent_database(sample):
    root, dist, installed = sample
    local = Path(os.environ['LOCALAPPDATA'])/'AURUM-AI/auth/users.sqlite3'
    before = local.read_bytes()
    mod.audit(root, installed)
    assert local.read_bytes() == before


def test_simulated_installer_never_invoked(sample, monkeypatch):
    # Audit only reads metadata; there is deliberately no subprocess/launch in module.
    assert not hasattr(mod, 'subprocess')


def test_cli_exit_code_two_on_blocker(sample, capsys):
    root, dist, installed = sample
    (dist/'_internal/secrets.toml').write_text('secret')
    code = mod.main(['--root', str(root), '--installed-dir', str(installed)])
    assert code == 2
    captured = capsys.readouterr().out
    assert 'BLOQUANT' in captured


def test_cli_exit_code_zero_with_only_warnings(sample, capsys):
    root, dist, installed = sample
    assert mod.main(['--root', str(root), '--installed-dir', str(installed)]) == 0
