"""Safe, repeatable optional branding patch for AURUM-AI V5.

Usage from project root:
  python tools/windows_desktop/apply_branding.py --apply
  python tools/windows_desktop/apply_branding.py --check

Only touches frontend/app.py and (when present) desktop_launcher.py; original
files are backed up as .bak before the first modification. Does NOT modify any
model, historical dataset, experiment, security gateway, or research artifact.
"""
from __future__ import annotations

import argparse
import ast
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


MARKER = "from frontend.branding import apply_aurum_branding, aurum_favicon"
LOGO_CALL = "apply_aurum_branding()"


def brand_app(text: str) -> str:
    """Conservatively adjust only page favicon + sidebar logo."""
    if LOGO_CALL in text and MARKER in text:
        ast.parse(text)
        return text
    if MARKER in text or LOGO_CALL in text:
        raise ValueError("Branding partiel detecte dans frontend/app.py : intervention manuelle requise")
    if text.count("import streamlit as st") != 1:
        raise ValueError("Impossible de trouver l'import Streamlit unique")
    if text.count('st.session_state["aurum_v5_mode"] = True') != 1:
        raise ValueError("Point de navigation/authentification V5 non reconnu")
    # Avoid matching other subpage config declarations or comments.
    hit = re.search(r'(?m)^([ \t]*)page_icon[ \t]*=[ \t]*[^\r\n]+,[ \t]*$', text)
    if not hit or text.count("page_icon=") != 1:
        raise ValueError("Parametre unique page_icon non reconnu ; aucun fichier change")
    text = text.replace(
        "import streamlit as st", "import streamlit as st\n" + MARKER, 1
    )
    text, count = re.subn(
        r'(?m)^([ \t]*)page_icon[ \t]*=[ \t]*[^\r\n]+,[ \t]*$',
        r'\1page_icon=aurum_favicon(),',
        text, count=1,
    )
    assert count == 1
    text = text.replace(
        'st.session_state["aurum_v5_mode"] = True',
        'st.session_state["aurum_v5_mode"] = True\n\n' + LOGO_CALL,
        1,
    )
    ast.parse(text)
    return text


def brand_launcher(text: str) -> str:
    """Use pywebview native .ico on versions that support icon=; old versions fallback."""
    if "# AURUM-AI branded desktop icon" in text:
        ast.parse(text)
        return text
    needle = "            webview.start(debug=False)"
    if text.count(needle) != 1:
        raise ValueError("Launcher non reconnu : impossible de modifier webview.start")
    replacement = '''            # AURUM-AI branded desktop icon (supported by recent pywebview).
            import inspect
            icon_path = project_root() / "assets" / "branding" / "aurum_ai.ico"
            kwargs = {"debug": False}
            if icon_path.is_file() and "icon" in inspect.signature(webview.start).parameters:
                kwargs["icon"] = str(icon_path)
            webview.start(**kwargs)'''
    text = text.replace(needle, replacement)
    ast.parse(text)
    return text


def write_change(path: Path, updated: str) -> bool:
    before = path.read_text(encoding="utf-8-sig")
    if before == updated:
        return False
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.name}.before_branding_{stamp}.bak")
    # backup in same folder, not a .py file discovered by Streamlit automatically
    shutil.copy2(path, backup)
    # Preserve UTF-8 (without BOM) content and normal newline convention.
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(updated)
    print(f"Modifie : {path}\nSauvegarde : {backup}")
    return True


def run(root: Path, apply: bool) -> None:
    assets = root / "assets" / "branding"
    for fn in ("aurum_ai_logo.png", "aurum_ai_wordmark.png", "aurum_ai_icon.png", "aurum_ai.ico"):
        if not (assets / fn).is_file():
            raise FileNotFoundError(f"Logo absent : {assets / fn}")
    app = root / "frontend" / "app.py"
    launcher = root / "desktop_launcher.py"
    template = root / "tools" / "windows_desktop" / "desktop_launcher_reference.py"
    if not app.is_file():
        raise FileNotFoundError(f"Application V5 absente : {app}")
    # Preflight all changes before touching any file.
    updated_app = brand_app(app.read_text(encoding="utf-8-sig"))
    if launcher.is_file():
        updated_launcher = brand_launcher(launcher.read_text(encoding="utf-8-sig"))
        copy_launcher = False
    elif template.is_file():
        updated_launcher = brand_launcher(template.read_text(encoding="utf-8-sig"))
        copy_launcher = True
    else:
        raise FileNotFoundError("desktop_launcher.py et template introuvables")
    if not apply:
        print("Controle OK : fichiers reconnus et syntaxe Python valide.")
        print("Modifications non appliquees (utilisez --apply).")
        return
    write_change(app, updated_app)
    if copy_launcher:
        shutil.copy2(template, launcher)
        print(f"Launcher Phase 20 installe depuis le template : {launcher}")
    write_change(launcher, updated_launcher)
    print("Branding AURUM-AI pret. Aucun fichier de donnees ou modele modifie.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AURUM-AI V5 branding installer")
    options = parser.add_mutually_exclusive_group(required=True)
    options.add_argument("--apply", action="store_true", help="Sauvegarder puis appliquer")
    options.add_argument("--check", action="store_true", help="Simuler sans ecrire")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    run(root, apply=args.apply)
