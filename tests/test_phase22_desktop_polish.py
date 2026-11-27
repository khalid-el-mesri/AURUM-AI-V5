"""Non-destructive tests for Phase22 AURUM-AI desktop polish."""
import importlib.util
import shutil
from pathlib import Path
import zipfile

import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[0]
BASE = Path(__file__).resolve().parent


def load_module(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / relative)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


polish = load_module('tools/windows_desktop/polish_desktop_phase22.py','phase22_polish')
audit = load_module('tools/windows_desktop/desktop_smoke_audit.py','phase22_audit')


APP_TEXT = '''import streamlit as st

def check_local_session(): return True
def end_local_session(): pass
st.session_state["aurum_v5_mode"] = True
if not check_local_session():
    st.stop()
end_local_session()
st.navigation(navigation, position="sidebar")
'''

LOGIN_TEXT = '''from pathlib import Path
ROOT = Path(__file__)
def authenticate_user(): pass
def register_and_send_otp(): pass
def confirm_email_registration(): pass
def unlock_local_session(): pass
with left:
    st.markdown(
        """
        <div class="aurum-brand">
            ◇ AURUM<span>-AI</span>
        </div>
        <div class="aurum-eyebrow">INVESTMENT INTELLIGENCE</div>
        """, unsafe_allow_html=True
    )
'''

NAV_TEXT = '''
''' .join(
    f'page("{name}", "Title", "i", "{url}")' for url,name in {
        "login":"auth.py", **audit.EXPECTED
    }.items()
)


def orig(file: str):
    if file == "frontend/app.py": return APP_TEXT + "\n" + NAV_TEXT
    if file == "frontend/auth/login_ui.py": return LOGIN_TEXT
    raise ValueError(file)


def test_app_patch_is_idempotent():
    original = orig('frontend/app.py')
    out = polish.patch_app(original)
    assert out != original
    assert 'st.set_option("client.toolbarMode", "minimal")' in out
    assert polish.patch_app(out) == out


def test_app_patch_preserves_auth_logic():
    original = orig('frontend/app.py')
    out = polish.patch_app(original)
    assert 'if not check_local_session():' in out
    assert 'end_local_session()' in out
    assert 'st.navigation(navigation' in out


def test_app_patch_rejects_unknown_shell():
    with pytest.raises(ValueError):
        polish.patch_app('print(42)')


def test_login_patch_is_idempotent():
    original = orig('frontend/auth/login_ui.py')
    out = polish.patch_login(original)
    assert out != original
    assert 'aurum_ai_wordmark.png' in out
    assert 'aurum_ai_icon.png' in out
    assert polish.patch_login(out) == out


def test_login_patch_preserves_authentication():
    original = orig('frontend/auth/login_ui.py')
    out = polish.patch_login(original)
    for term in ['authenticate_user','register_and_send_otp','confirm_email_registration','unlock_local_session']:
        assert term in out
    assert '<div class="aurum-brand">' not in out


def test_login_patch_rejects_unsupported():
    with pytest.raises(ValueError):
        polish.patch_login('with left:\n    st.markdown("Hi")\n')


def make_tree(tmp_path, apptext=None, logintext=None):
    root = tmp_path / 'aurum'
    (root / 'frontend' / 'auth').mkdir(parents=True)
    (root / 'assets' / 'branding').mkdir(parents=True)
    (root / 'frontend' / 'app.py').write_text(apptext or orig('frontend/app.py'),encoding='utf-8')
    (root / 'frontend' / 'auth' / 'login_ui.py').write_text(logintext or orig('frontend/auth/login_ui.py'),encoding='utf-8')
    for asset in ('aurum_ai_icon.png','aurum_ai_wordmark.png','aurum_ai.ico'):
        (root / 'assets' / 'branding' / asset).write_bytes(b'placeholder test asset')
    return root


def test_check_does_not_write(tmp_path):
    root = make_tree(tmp_path)
    b1 = (root/'frontend/app.py').read_bytes()
    b2 = (root/'frontend/auth/login_ui.py').read_bytes()
    result = polish.run(root, apply=False)
    assert all(result.values())
    assert (root/'frontend/app.py').read_bytes() == b1
    assert (root/'frontend/auth/login_ui.py').read_bytes() == b2
    assert list(root.rglob('*.bak')) == []


def test_apply_writes_backup_and_no_secrets(tmp_path):
    root = make_tree(tmp_path)
    polish.run(root, apply=True)
    assert len(list(root.rglob('*.bak'))) == 2
    assert polish.APP_MARKER in (root/'frontend/app.py').read_text()
    assert polish.LOGIN_MARKER in (root/'frontend/auth/login_ui.py').read_text()
    assert not (root/'runtime/auth/users.sqlite3').exists()


def test_apply_is_idempotent(tmp_path):
    root = make_tree(tmp_path)
    polish.run(root, apply=True)
    b = list(root.rglob('*.bak'))
    polish.run(root, apply=True)
    assert len(list(root.rglob('*.bak'))) == len(b)


def test_preflight_stages_both_before_any_write(tmp_path):
    root = make_tree(tmp_path,logintext='with left:\n    st.markdown("Hello")\n')
    a = (root/'frontend/app.py').read_bytes()
    with pytest.raises(ValueError):
        polish.run(root,apply=True)
    assert (root/'frontend/app.py').read_bytes() == a
    assert not list(root.rglob('*.bak'))


def test_expected_routes_count():
    assert len(audit.EXPECTED) == 19
    routes = audit.inspect_routes(orig('frontend/app.py'))
    assert all(routes.get(u)==f for u,f in audit.EXPECTED.items())
    assert routes.get('login') == 'auth.py'


def test_route_checker_missing(tmp_path):
    root = tmp_path/'aurum'
    (root/'frontend').mkdir(parents=True)
    (root/'frontend/app.py').write_text('print(1)')
    problems = audit.audit(root)
    assert any('Route absente' in p for p in problems)


def test_route_checker_valid_sources(tmp_path):
    root = tmp_path/'aurum'
    (root/'frontend/v5_pages').mkdir(parents=True)
    (root/'frontend/auth').mkdir(parents=True)
    (root/'assets/branding').mkdir(parents=True)
    (root/'frontend/app.py').write_text(orig('frontend/app.py'), encoding='utf-8')
    for f in (*audit.EXPECTED.values(),'auth.py'):
        (root/'frontend/v5_pages'/f).write_text('# mocked')
    for f in ('frontend/auth/database.py','frontend/auth/login_ui.py','assets/branding/aurum_ai_icon.png','assets/branding/aurum_ai_wordmark.png'):
        (root/f).write_text('mocked')
    assert audit.audit(root)==[]
    assert audit.audit(root,packaged=True) != []


def test_does_not_collect_password_content(tmp_path):
    root = make_tree(tmp_path)
    assert 'sqlite3' not in polish.patch_app(orig('frontend/app.py'))
    assert 'password_hash' not in polish.patch_login(orig('frontend/auth/login_ui.py'))
