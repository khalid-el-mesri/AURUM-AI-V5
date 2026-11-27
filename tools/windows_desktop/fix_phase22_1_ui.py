"""AURUM-AI V5 phase 22.1: narrowly scoped, reversible Streamlit UI fixes.

Read-only by default. --apply backs up original page source files first.
Does not open or modify the ledger, datasets, authentication DB, or models.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[2]
LEDGER_REL = Path("frontend/v5_pages/blockchain_ledger.py")
TWIN_REL = Path("frontend/v5_pages/digital_twin.py")
EXPERIMENTS_REL = Path("frontend/v5_pages/experiments.py")

# The original page passes unsafe_allow_html=True already. The indentation of its
# HTML causes CommonMark to display it as preformatted code. Dedent on rendering.
LEDGER_HERO = '''def _hero() -> None:
    """Render the original cryptographic ledger banner as HTML, not a code block."""
    from textwrap import dedent

    html = dedent("""\\
        <style>
        .aurum-chain-hero {
            background: linear-gradient(115deg, #0a344a, #0a1727 60%, #06121f);
            border: 1px solid #23739b;
            border-radius: 16px;
            padding: 22px 26px;
            margin: 10px 0 14px;
            color: #edf6ff;
        }
        .aurum-chain-hero .eyebrow {
            color: #35ccff;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 2.7px;
        }
        .aurum-chain-hero h1 {
            font-size: 33px;
            margin: 9px 0 5px;
            line-height: 1.2;
            color: #fff;
        }
        .aurum-chain-hero p {
            margin: 0;
            color: #aed0ea;
            font-size: 14px;
        }
        </style>
        <div class="aurum-chain-hero">
            <div class="eyebrow">◇ AURUM-AI V5 · CRYPTOGRAPHIC AUDIT TRAIL</div>
            <h1>🔗 Blockchain Ledger</h1>
            <p>SHA-256 · Genesis &amp; Hash Links · Recorded Events · Historical Integrity</p>
        </div>
    """).strip()
    st.markdown(html, unsafe_allow_html=True)


'''

TWIN_OLD = re.compile(r'\{"Contrôle":\s*k,\s*"Valeur":\s*v\}\s+for\s+k,\s*v\s+in\s+report\.items\(\)')
EXPERIMENTS_OLD = re.compile(r'\{"Champ":\s*k,\s*"Valeur":\s*v\}\s+for\s+k,\s*v\s+in\s+scalar\.items\(\)')


def patch_ledger(source: str) -> tuple[str, bool]:
    """Replace ONLY the Ledger's _hero function if it is the original version."""
    if '"""Render the original cryptographic ledger banner as HTML, not a code block."""' in source:
        return source, False
    start_marker = "def _hero() -> None:"
    end_marker = "def _hash_card("
    if source.count(start_marker) != 1 or source.count(end_marker) != 1:
        raise ValueError("Blockchain Ledger: _hero/_hash_card introuvable ou ambigu")
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    original = source[start:end]
    if "st.markdown('''<style>" not in original or "unsafe_allow_html=True" not in original:
        raise ValueError("Blockchain Ledger: _hero ne correspond pas au code attendu; rien modifier")
    return source[:start] + LEDGER_HERO + source[end:], True


def patch_twin(source: str) -> tuple[str, bool]:
    """Only presentation frame values become strings; source report remains unchanged."""
    replacement = '{"Contrôle": str(k), "Valeur": str(v)} for k, v in report.items()'
    updated, n = TWIN_OLD.subn(replacement, source)
    if n > 1:
        raise ValueError("Digital Twin: plus d'une transformation possible")
    if not n:
        if replacement in source:
            return source, False
        raise ValueError("Digital Twin: tableau Valeur non reconnu; rien modifier")
    return updated, True


def patch_experiments(source: str) -> tuple[str, bool]:
    """Cast heterogeneous scalar metadata for a purely visual dataframe."""
    replacement = '{"Champ": str(k), "Valeur": str(v)} for k, v in scalar.items()'
    updated, n = EXPERIMENTS_OLD.subn(replacement, source)
    if n > 1:
        raise ValueError("Experiments: plus d'une transformation possible")
    if not n:
        if replacement in source:
            return source, False
        raise ValueError("Experiments: tableau Valeur non reconnu; rien modifier")
    return updated, True


def _read_preserve(path: Path) -> tuple[str, bool, str]:
    raw = path.read_bytes()
    bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig" if bom else "utf-8")
    newline = "\r\n" if b"\r\n" in raw else "\n"
    return text.replace("\r\n", "\n"), bom, newline


def prepare(root: Path = ROOT) -> list[tuple[Path, str, bool, str, bool]]:
    jobs = [(LEDGER_REL, patch_ledger), (TWIN_REL, patch_twin),
            (EXPERIMENTS_REL, patch_experiments)]
    prepared = []
    for relative, patcher in jobs:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Fichier manquant: {path}")
        original, bom, newline = _read_preserve(path)
        updated, changed = patcher(original)
        compile(updated, str(path), "exec")
        prepared.append((path, updated, bom, newline, changed))
    return prepared


def run(root: Path, apply: bool = False) -> list[tuple[Path, bool]]:
    prepared = prepare(root)
    output = [(path, changed) for path, _, _, _, changed in prepared]
    if apply:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        for path, updated, bom, newline, changed in prepared:
            if not changed:
                continue
            backup = path.with_name(f"{path.name}.before_phase22_1_{stamp}.bak")
            shutil.copy2(path, backup)
            data = updated.replace("\n", newline).encode("utf-8")
            path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + data)
            print(f"Modifie : {path}\nSauvegarde : {backup}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="Vérifier, sans modification")
    action.add_argument("--apply", action="store_true", help="Sauvegarder et appliquer")
    args = parser.parse_args()
    try:
        changes = run(ROOT, apply=args.apply)
    except (ValueError, FileNotFoundError, UnicodeError, OSError, SyntaxError) as exc:
        print(f"CONTROLE BLOQUANT : {exc}")
        return 1
    for path, changed in changes:
        print(f"{'Modifications nécessaires' if changed else 'Déjà corrigé'} : {path.relative_to(ROOT)}")
    if not args.apply:
        print("Lecture seule. Aucun fichier modifie. Utilisez --apply après vérification.")
    else:
        print("Aucun ledger, dataset, fichier de comptes ou modele modifie.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
