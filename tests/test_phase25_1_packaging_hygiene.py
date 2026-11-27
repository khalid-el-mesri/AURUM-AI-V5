"""Synthetic tests; no real user database, credentials or executable accessed."""
import ast
import importlib.util
from pathlib import Path
import sys
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'tools/windows_desktop/fix_phase25_1_packaging.py'
spec = importlib.util.spec_from_file_location('aurum_packaging_hygiene_25_1', SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

BUILDER = """from pathlib import Path
import os

def safe_file(path: Path) -> bool:
    if path.name.lower() == '.env':
        return False
    name = path.name.lower()
    if path.suffix.lower() != '.py' and ('secret' in name or 'password' in name):
        return False
    return True

def make_spec():
    spec = f'''from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules
hiddenimports += collect_submodules(
    'frontend.auth',
    filter=lambda name: not any(
        part in ('test', 'tests', 'testing') for part in name.split('.')
    ),
    on_error='warn once',
)
'''
    return spec
"""
ISS = ('[Setup]\nPrivilegesRequired=lowest\n'
       '[Files]\nSource: "..\\dist\\AURUM-AI V5\\*"; DestDir: "{app}"; '
       'Excludes: "*.sqlite3,.env,secrets.toml"; Flags: ignoreversion recursesubdirs createallsubdirs\n')

@pytest.fixture
def project(tmp_path):
    root = tmp_path / 'AURUM-AI'
    (root / 'tools/windows_desktop').mkdir(parents=True)
    (root / 'installer').mkdir(parents=True)
    (root / mod.BUILDER).write_text(BUILDER, encoding='utf-8')
    (root / mod.ISS).write_text(ISS, encoding='utf-8')
    (root / 'dist/AURUM-AI V5/_internal/frontend/auth').mkdir(parents=True)
    (root / 'dist/AURUM-AI V5/_internal/frontend/auth/login_ui_backup.py').write_text('archived version')
    (root / 'dist/AURUM-AI V5/_internal/frontend/auth/security.py').write_text('current version')
    return root

@pytest.mark.parametrize('stem', ('app_before_v5', 'login_ui_backup', 'idle_guard_before_unlock',
                 'autonomous_agents_before_phase6', 'database-old', 'config.copy', 'models_archive_2026'))
def test_backup_classification(stem):
    assert mod.LEGACY_RE.search(stem)

@pytest.mark.parametrize('stem', ('login_ui','email_service','password_unlock','blockchain_ledger'))
def test_active_classification(stem):
    assert not mod.LEGACY_RE.search(stem)

def test_safe_file_excludes_backups_but_preserves_active_auth():
    patched = mod.fix_builder(BUILDER)
    ns = {}
    exec(patched, ns)
    assert ns['safe_file'](Path('frontend/auth/security.py'))
    assert ns['safe_file'](Path('frontend/auth/password_unlock.py'))
    for p in ('frontend/auth/login_ui_backup.py', 'frontend/app_before_v5.py',
              'frontend/NYSE_V3_Backtesting_backup.py', 'frontend/v5_pages/ai_model_lab_before_phase15.py'):
        assert not ns['safe_file'](Path(p))

def test_hiddenimports_reject_old_auth_modules():
    ns = {}
    exec(mod.fix_builder(BUILDER), ns)
    tree = ast.parse(ns['make_spec']())
    auth_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Name) and n.func.id == 'collect_submodules'
                  and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == 'frontend.auth']
    assert len(auth_calls) == 1
    callback = next(k.value for k in auth_calls[0].keywords if k.arg == 'filter')
    func = eval(compile(ast.Expression(body=callback), '<lambda>', 'eval'))
    assert func('frontend.auth.security')
    assert func('frontend.auth.accounts')
    assert func('frontend.auth.login_ui')
    assert not func('frontend.auth.login_ui_backup')
    assert not func('frontend.auth.login_ui_before_unlock')
    assert not func('frontend.auth.tests')

def test_iss_adds_defensive_patterns_without_dropping_existing_exclusions():
    changed = mod.fix_iss(ISS)
    assert '*.sqlite3' in changed and 'secrets.toml' in changed
    assert '*_backup.py' in changed and '*_before_*.py' in changed
    assert changed.count('Source: "..\\dist\\AURUM-AI V5\\*"') == 1
    assert 'PrivilegesRequired=lowest' in changed
    assert '[UninstallDelete]' not in changed

def test_readonly_check_does_not_modify_project(project):
    before = [(p, p.read_bytes()) for p in (project / mod.BUILDER, project / mod.ISS)]
    assert mod.run(project, apply=False) == 0
    assert all(p.read_bytes() == content for p, content in before)
    assert not list(project.rglob('*.bak'))

def test_apply_creates_backups_and_modifies_only_two_files(project):
    assert mod.run(project, apply=True) == 0
    for p in (mod.BUILDER, mod.ISS):
        assert (project / p).exists()
        assert len(list((project / p).parent.glob((project / p).name + '.before_phase25_1_*.bak'))) == 1
    assert (project / 'dist/AURUM-AI V5/_internal/frontend/auth/login_ui_backup.py').exists()

def test_repeated_apply_idempotent(project):
    assert mod.run(project, apply=True) == 0
    b0 = (project / mod.BUILDER).read_bytes()
    i0 = (project / mod.ISS).read_bytes()
    assert mod.run(project, apply=True) == 0
    assert (project / mod.BUILDER).read_bytes() == b0
    assert (project / mod.ISS).read_bytes() == i0
    assert len(list((project / mod.BUILDER).parent.glob('*.before_phase25_1_*.bak'))) == 1

def test_disallow_unknown_builder_without_changes(project):
    b = project / mod.BUILDER
    b.write_text('print("unknown")', encoding='utf-8')
    original_iss = (project / mod.ISS).read_bytes()
    assert mod.run(project, apply=True) == 2
    assert (project / mod.ISS).read_bytes() == original_iss

def test_disallow_unknown_installer_without_changes(project):
    i = project / mod.ISS
    i.write_text('[Files]\nSource: "C:\\SomeOtherFolder\\*"\n', encoding='utf-8')
    original_builder = (project / mod.BUILDER).read_bytes()
    assert mod.run(project, apply=True) == 2
    assert (project / mod.BUILDER).read_bytes() == original_builder

def test_reports_dist_backups_without_removing_them(project):
    entries = mod.legacy_in_dist(project)
    assert entries == ['_internal/frontend/auth/login_ui_backup.py']
    assert (project / 'dist/AURUM-AI V5/_internal/frontend/auth/login_ui_backup.py').is_file()

def test_builder_parses_after_repair():
    ast.parse(mod.fix_builder(BUILDER))

def test_missing_builder_refuses(project):
    (project / mod.BUILDER).unlink()
    assert mod.run(project, apply=False) == 2

def test_private_sqlite_unmodified_by_patch(project):
    db = project / 'runtime/auth/users.sqlite3'
    db.parent.mkdir(parents=True)
    db.write_bytes(b'private do not touch')
    assert mod.run(project, apply=True) == 0
    assert db.read_bytes() == b'private do not touch'
