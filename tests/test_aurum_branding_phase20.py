"""Phase 20 official bull-logo integration regression checks."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import shutil

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / "tools" / "windows_desktop" / "apply_branding.py"
spec = importlib.util.spec_from_file_location("aurum_branding_patcher", PATCHER)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

APP_EXAMPLE = '''from __future__ import annotations
import streamlit as st
ROOT = "project"
st.set_page_config(
    page_title="AURUM-AI",
    page_icon="old",
    layout="wide",
)
st.session_state["aurum_v5_mode"] = True
if not True:
    pass
selected_page = st.navigation({})
selected_page.run()
'''


def create_root(tmp_path):
    (tmp_path / "frontend").mkdir()
    shutil.copy(ROOT / "frontend" / "branding.py", tmp_path / "frontend" / "branding.py")
    (tmp_path / "frontend" / "app.py").write_text(APP_EXAMPLE, encoding="utf-8")
    (tmp_path / "tools" / "windows_desktop").mkdir(parents=True)
    shutil.copy(ROOT / "tools" / "windows_desktop" / "desktop_launcher_reference.py", tmp_path / "tools" / "windows_desktop" / "desktop_launcher_reference.py")
    (tmp_path / "assets" / "branding").mkdir(parents=True)
    for p in (ROOT / "assets" / "branding").iterdir():
        shutil.copy(p, tmp_path / "assets" / "branding" / p.name)
    return tmp_path


def test_logo_images_valid():
    for fn in ("aurum_ai_logo.png", "aurum_ai_wordmark.png", "aurum_ai_icon.png"):
        with Image.open(ROOT / "assets" / "branding" / fn) as im:
            im.verify()


def test_win_icon_contains_256_and_32():
    with Image.open(ROOT / "assets" / "branding" / "aurum_ai.ico") as im:
        sizes = im.info["sizes"]
        assert (256, 256) in sizes
        assert (32, 32) in sizes


def test_brand_app_changes_only_favicon_and_adds_logo():
    changed = mod.brand_app(APP_EXAMPLE)
    assert "page_icon=aurum_favicon()" in changed
    assert "apply_aurum_branding()" in changed
    assert 'st.session_state["aurum_v5_mode"] = True' in changed
    assert "selected_page.run()" in changed
    ast.parse(changed)


def test_brand_app_idempotent():
    once = mod.brand_app(APP_EXAMPLE)
    assert mod.brand_app(once) == once


def test_brand_app_fails_closed_if_unrecognized():
    with pytest.raises(ValueError):
        mod.brand_app("import streamlit as st\nst.write('X')\n")


def test_brand_launcher_preserves_streamlit_host():
    base = (ROOT / "tools" / "windows_desktop" / "desktop_launcher_reference.py").read_text(encoding="utf-8")
    result = mod.brand_launcher(base)
    assert 'HOST = "127.0.0.1"' in result
    assert '"--server.enableXsrfProtection=true"' in result
    assert '"--server.address={HOST}"' in result
    assert 'kwargs["icon"] = str(icon_path)' in result
    ast.parse(result)


def test_launcher_idempotent():
    base = (ROOT / "tools" / "windows_desktop" / "desktop_launcher_reference.py").read_text(encoding="utf-8")
    once = mod.brand_launcher(base)
    assert mod.brand_launcher(once) == once


def test_apply_creates_backup_and_does_not_touch_data(tmp_path):
    root = create_root(tmp_path)
    payload = root / "artifacts" / "experiments" / "immutable.txt"
    payload.parent.mkdir(parents=True)
    payload.write_text("unchanged", encoding="utf-8")
    mod.run(root, apply=False)
    assert "aurum_favicon()" not in (root / "frontend" / "app.py").read_text()
    mod.run(root, apply=True)
    assert (root / "desktop_launcher.py").is_file()
    assert 'webview.start(**kwargs)' in (root / "desktop_launcher.py").read_text()
    assert "aurum_favicon()" in (root / "frontend" / "app.py").read_text()
    assert payload.read_text() == "unchanged"
    assert list((root / "frontend").glob("app.py.before_branding_*.bak"))
    mod.run(root, apply=True)
    assert len(list((root / "frontend").glob("app.py.before_branding_*.bak"))) == 1


def test_missing_assets_fail_without_writes(tmp_path):
    root = create_root(tmp_path)
    (root / "assets" / "branding" / "aurum_ai_icon.png").unlink()
    before = (root / "frontend" / "app.py").read_bytes()
    with pytest.raises(FileNotFoundError):
        mod.run(root, apply=True)
    assert (root / "frontend" / "app.py").read_bytes() == before
