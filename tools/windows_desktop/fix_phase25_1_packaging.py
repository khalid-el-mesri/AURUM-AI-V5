"""Remove historical Python backups from future AURUM-AI Windows bundles.

Safe for existing user data: only modifies the build generator and the Inno
Setup manifest, with backups. Never edits/destroys user SQLite or runtime data.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import sys
import tempfile

BUILDER = Path('tools/windows_desktop/build_windows_exe.py')
ISS = Path('installer/AURUM_AI_V5.iss')
MARKER = 'LEGACY_BACKUP_PATTERN = re.compile('
FILTER_MARKER = "any(piece in ('backup', 'before', 'old', 'bak', 'copy', 'archive')"
LEGACY_RE = re.compile(r'(?:^|[_\-.])(?:backup|before|old|bak|copy|archive)(?:[_\-.]|$)', re.I)
ISS_EXCLUSIONS = ('*_before_*.py', '*_backup.py', '*_old.py', '*_bak.py',
                  '*_copy.py', '*_archive.py', '*-backup.py', '*-before-*.py',
                  'backup_*.py', 'before_*.py', 'old_*.py', 'copy_*.py',
                  'archive_*.py')


class UnexpectedBuildScript(RuntimeError):
    pass


def fix_builder(original: str) -> str:
    text = original
    if 'def safe_file(path: Path) -> bool:' not in text:
        raise UnexpectedBuildScript('safe_file() introuvable. Ne pas ecraser cette version du build.')
    if MARKER not in text:
        import_anchor = 'from pathlib import Path\n'
        if import_anchor not in text:
            raise UnexpectedBuildScript('Import Path introuvable.')
        if 'import re\n' not in text:
            pos = text.find('import os\n')
            if pos < 0:
                raise UnexpectedBuildScript('Import os introuvable.')
            text = text[:pos] + text[pos:].replace('import os\n', 'import os\nimport re\n', 1)
        declaration = "LEGACY_BACKUP_PATTERN = re.compile(r'(?:^|[_\\-.])(?:backup|before|old|bak|copy|archive)(?:[_\\-.]|$)', re.I)"
        text = text.replace('import re\n', 'import re\n\n' + declaration + '\n', 1)
    needle = '    name = path.name.lower()\n'
    replacement = (needle + "    # Keep source backups locally, never in the distributed application.\n"
                   "    if path.suffix.lower() == '.py' and LEGACY_BACKUP_PATTERN.search(path.stem):\n"
                   "        return False\n")
    if 'LEGACY_BACKUP_PATTERN.search(path.stem)' not in text:
        start = text.index('def safe_file(path: Path) -> bool:')
        end = text.find('\ndef ', start+1)
        part = text[start:] if end < 0 else text[start:end]
        if part.count(needle) != 1:
            raise UnexpectedBuildScript('Point d\'insertion safe_file inconnu; aucune modification.')
        patched = part.replace(needle, replacement, 1)
        text = text[:start]+patched+('' if end < 0 else text[end:])
    if FILTER_MARKER not in text:
        # The generated spec is a literal embedded in make_spec(). Restrict its
        # auth hiddenimports collection, rather than indiscriminately deleting
        # modules that may be required at runtime.
        auth_anchor = "hiddenimports += collect_submodules(\n    'frontend.auth',\n"
        after = "        part in ('test', 'tests', 'testing') for part in name.split('.')\n"
        if auth_anchor not in text:
            raise UnexpectedBuildScript('Filtre hiddenimports frontend.auth introuvable; aucune modification.')
        start = text.index(auth_anchor)
        end = text.find('\n)', start)
        if end < 0:
            raise UnexpectedBuildScript('Fin de filtre hiddenimports introuvable.')
        part = text[start:end]
        if part.count(after) != 1:
            raise UnexpectedBuildScript('Filtre hiddenimports non reconnu; aucune modification.')
        changed = (
            "        (part in ('test', 'tests', 'testing') or\n"
            "         any(piece in ('backup', 'before', 'old', 'bak', 'copy', 'archive')\n"
            "             for piece in part.replace('-', '_').split('_')))\n"
            "        for part in name.split('.')\n"
        )
        text = text[:start] + part.replace(after, changed) + text[end:]
    try:
        ast.parse(text)
    except SyntaxError as exc:
        raise UnexpectedBuildScript(f'La modification serait invalide: {exc}') from exc
    return text


def fix_iss(original: str) -> str:
    text = original
    lines = text.splitlines(keepends=True)
    targets = [i for i, line in enumerate(lines) if line.lstrip().startswith('Source: "..\\dist\\AURUM-AI V5\\*";')]
    if len(targets) != 1:
        raise UnexpectedBuildScript('Source Inno Setup non reconnue; aucune modification.')
    i = targets[0]
    line = lines[i]
    marker = 'Excludes: "'
    if marker not in line:
        raise UnexpectedBuildScript('Excludes: Inno Setup introuvable.')
    front, remainder = line.split(marker, 1)
    if '"' not in remainder:
        raise UnexpectedBuildScript('Fin des exclusions Inno Setup introuvable.')
    current, tail = remainder.split('"', 1)
    clauses = [x.strip() for x in current.split(',') if x.strip()]
    existing = {x.lower() for x in clauses}
    clauses += [x for x in ISS_EXCLUSIONS if x.lower() not in existing]
    lines[i] = front + marker + ','.join(clauses) + '"' + tail
    return ''.join(lines)


def legacy_in_dist(root: Path) -> list[str]:
    d = root / 'dist' / 'AURUM-AI V5'
    if not d.is_dir():
        return []
    return sorted(str(p.relative_to(d)).replace('\\', '/') for p in d.rglob('*.py')
                  if LEGACY_RE.search(p.stem))


def _atomic_write(path: Path, content: str) -> None:
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                                     dir=path.parent, prefix='.aurum_new_', delete=False) as file:
        file.write(content)
        tmp = Path(file.name)
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def run(root: Path, apply: bool = False) -> int:
    root = Path(root).resolve()
    builder = root / BUILDER
    iss = root / ISS
    for p in (builder, iss):
        if not p.is_file():
            print(f'ERREUR : fichier absent : {p}')
            return 2
    orig_builder = builder.read_text(encoding='utf-8-sig')
    orig_iss = iss.read_text(encoding='utf-8-sig')
    try:
        new_builder = fix_builder(orig_builder)
        new_iss = fix_iss(orig_iss)
    except UnexpectedBuildScript as exc:
        print(f'ERREUR : {exc}')
        return 2
    stale = legacy_in_dist(root)
    print(f'Anciens fichiers Python actuellement presents dans dist : {len(stale)}')
    for entry in stale[:10]:
        print('  -', entry)
    if len(stale) > 10:
        print(f'  ... {len(stale) - 10} autres (non affiches)')
    changes = [(builder, orig_builder, new_builder), (iss, orig_iss, new_iss)]
    modified = [p for p, old, new in changes if old != new]
    print('Fichiers de configuration a modifier :', len(modified))
    for p in modified:
        print('  -', p.relative_to(root))
    if not apply:
        print('LECTURE SEULE : utilisez --apply pour corriger les futurs builds.')
        return 0
    if not modified:
        print('Deja applique; aucune modification supplementaire.')
        return 0
    suffix = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    for p, old, new in changes:
        if old == new:
            continue
        backup = p.with_name(p.name + f'.before_phase25_1_{suffix}.bak')
        backup.write_bytes(p.read_bytes())
        _atomic_write(p, new)
        print('Modifie :', p)
        print('Sauvegarde :', backup)
    print('Le dossier dist reste ANCIEN jusqu\'a une nouvelle compilation PyInstaller.')
    print('L\'application installee reste inchangee jusqu\'a sa reinstallation.')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='AURUM-AI V5 - Phase 25.1 package hygiene')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true', help='diagnostic en lecture seule')
    mode.add_argument('--apply', action='store_true', help='modifier uniquement builder et .iss')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    return run(args.root, apply=args.apply)


if __name__ == '__main__':
    raise SystemExit(main())
