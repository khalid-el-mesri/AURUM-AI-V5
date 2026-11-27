"""AURUM-AI V5 desktop UI polish. Safe preflight & opt-in, no auth/data writes.

Usage from project root:
  python tools/windows_desktop/polish_desktop_phase22.py --check
  python tools/windows_desktop/polish_desktop_phase22.py --apply
"""
from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
import re
import shutil
from datetime import datetime, timezone

APP_MARKER = '# AURUM PHASE22: minimal client toolbar'
LOGIN_MARKER = '# AURUM PHASE22: official bull + wordmark in secure access'


def patch_app(text: str) -> str:
    """Only add the supported Streamlit client toolbar setting to the main shell."""
    ast.parse(text)
    if APP_MARKER in text or 'st.set_option("client.toolbarMode", "minimal")' in text or "st.set_option('client.toolbarMode', 'minimal')" in text:
        return text
    anchor = 'st.session_state["aurum_v5_mode"] = True'
    if text.count(anchor) != 1:
        raise ValueError('App V5 non reconnue : st.session_state aurum_v5_mode absent ou duplique.')
    if 'st.set_option("client.toolbarMode"' in text or "st.set_option('client.toolbarMode'" in text:
        raise ValueError('Configuration de toolbar deja personnalisee : verification manuelle requise.')
    replacement = anchor + (
        '\n\n' + APP_MARKER + '\n'
        'st.set_option("client.toolbarMode", "minimal")'
    )
    result = text.replace(anchor, replacement, 1)
    ast.parse(result)
    return result


_BRAND = re.compile(
    r'(?m)^[ \t]*<div class="aurum-brand">[ \t]*\r?\n'
    r'[ \t]*[^\r\n]*AURUM<span>-AI</span>[ \t]*\r?\n'
    r'[ \t]*</div>[ \t]*\r?\n',
)

_LOGIN_INSERT = '''    # AURUM PHASE22: official bull + wordmark in secure access
    _aurum_icon = ROOT / "assets" / "branding" / "aurum_ai_icon.png"
    _aurum_wordmark = ROOT / "assets" / "branding" / "aurum_ai_wordmark.png"
    if _aurum_icon.is_file() and _aurum_wordmark.is_file():
        _aurum_symbol, _aurum_label = st.columns(
            [1, 3.3], gap="small", vertical_alignment="center"
        )
        with _aurum_symbol:
            st.image(str(_aurum_icon), width="stretch")
        with _aurum_label:
            st.image(str(_aurum_wordmark), width="stretch")
    else:
        st.markdown("## AURUM-AI")

'''


def patch_login(text: str) -> str:
    """Replace only the existing login-page brand header, not login/OTP code."""
    ast.parse(text)
    if LOGIN_MARKER in text:
        return text
    anchor = 'with left:\n'
    if text.count(anchor) != 1:
        # Support original Windows CRLF even after Python universal newlines.
        raise ValueError('Ecran de login non reconnu : bloc with left absent ou duplique.')
    updated, count = _BRAND.subn('', text)
    if count != 1:
        raise ValueError('Entete logo login non reconnu ou modifie : aucun fichier modifie.')
    result = updated.replace(anchor, anchor + '\n' + _LOGIN_INSERT, 1)
    ast.parse(result)
    return result


def _stage_write(path: Path, text: str) -> None:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    backup = path.with_name(f'{path.name}.before_phase22_{stamp}.bak')
    shutil.copy2(path, backup)
    temporary = path.with_name(f'.{path.name}.phase22.tmp')
    try:
        temporary.write_text(text, encoding='utf-8', newline='')
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f'Modifie : {path}')
    print(f'Sauvegarde : {backup}')


def run(root: Path, apply: bool) -> dict[str, bool]:
    root = root.resolve()
    app = root / 'frontend' / 'app.py'
    login = root / 'frontend' / 'auth' / 'login_ui.py'
    logos = [root / 'assets' / 'branding' / n for n in (
        'aurum_ai_icon.png', 'aurum_ai_wordmark.png', 'aurum_ai.ico')]
    missing = [str(p) for p in (app, login, *logos) if not p.is_file()]
    if missing:
        raise FileNotFoundError('Fichiers attendus manquants : ' + ', '.join(missing))
    originals = {p: p.read_text(encoding='utf-8-sig') for p in (app, login)}
    updated = {app: patch_app(originals[app]), login: patch_login(originals[login])}
    modifications = {str(p.relative_to(root)): updated[p] != originals[p] for p in (app, login)}
    if apply:
        for path in (app, login):
            if updated[path] != originals[path]:
                _stage_write(path, updated[path])
        print('Finition appliquee. Pas de modification de database.py ou des donnees.')
    else:
        print('Controle OK. Aucune modification effectuee.')
        for p, changed in modifications.items():
            print(f'  {p}: {"a modifier" if changed else "deja applique"}')
    return modifications


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AURUM-AI V5 Phase22 desktop polish')
    exclusive = parser.add_mutually_exclusive_group(required=True)
    exclusive.add_argument('--check', action='store_true')
    exclusive.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    try:
        run(Path(__file__).resolve().parents[2], apply=args.apply)
    except (OSError, SyntaxError, ValueError) as exc:
        parser.exit(1, f'Impossible de modifier en securite : {exc}\n')
