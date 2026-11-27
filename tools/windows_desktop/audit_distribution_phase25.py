"""AURUM-AI V5 Phase 25: read-only Windows distribution pre-release audit.

Checks the onedir PyInstaller tree, its installed copy (if present), source
packaging definitions, and obvious secret-bearing files or assignments.

It never reads SQLite table content, exports credentials, updates code, or
uninstalls anything. This is NOT a binary secrets audit or a runtime test.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import re

EXPECTED_EXE = 'AURUM-AI V5.exe'
EXPECTED_SETUP = 'AURUM-AI-V5-Setup-0.9.0.exe'
TEXT_EXTS = {'.py', '.json', '.toml', '.ini', '.cfg', '.yaml', '.yml', '.iss', '.ps1'}
SUSPECT_EXTS = {'.sqlite', '.sqlite3', '.db', '.p12', '.pfx', '.ppk', '.key', '.kdbx'}
SECRET_NAMES = {'secrets.toml', 'credentials.json', 'credential.json', 'service-account.json',
                'service_account.json', 'token.json', 'id_rsa', 'id_ed25519', 'known_passwords.txt'}
SECRETS_LIKE = re.compile(r'(?:password|passwd|smtp_pass|api_key|private_key|client_secret|access_token|auth_token|secret)', re.I)
BACKUP_PATTERN = re.compile(r'(?:^|[_\-.])(?:backup|before|old|bak|copy|archive)(?:[_\-.]|$)', re.I)
PLACEHOLDER = re.compile(r'^(?:changeme|example|placeholder|dummy|test|none|null|your[_\- ]|replace[_\- ])', re.I)
IGNORE_DIRS = {'.venv', '.git', '__pycache__', '.pytest_cache', 'node_modules', 'dist', 'build'}


@dataclass
class AuditResult:
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    good: list[str] = field(default_factory=list)

    @property
    def safe_to_proceed(self) -> bool:
        return not self.blockers


def sensitive_file(relative: Path) -> str | None:
    """Classify path by name only. Public certifi bundle is permitted."""
    path = relative.as_posix().lower()
    name = relative.name.lower()
    if path.endswith('_internal/certifi/cacert.pem'):
        return None
    if name.startswith('.env') or name in SECRET_NAMES:
        return 'configuration ou donnees sensibles'
    if relative.suffix.lower() in SUSPECT_EXTS:
        return 'base de donnees ou cle privee potentielle'
    if relative.suffix.lower() in ('.pem', '.crt'):
        return 'certificat/cle PEM a examiner'
    if 'private' in name and ('key' in name or name.endswith('.json')):
        return 'materiel cryptographique potentiel'
    return None


def is_legacy_python(relative: Path) -> bool:
    return relative.suffix.lower() == '.py' and bool(BACKUP_PATTERN.search(relative.stem))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _walk_regular_files(directory: Path):
    for p in directory.rglob('*'):
        if p.is_file() or p.is_symlink():
            yield p


def literal_secret_assignment(source: str) -> list[tuple[int, str]]:
    """Detect assignments to sensitive names with literal non-placeholder text.

    Does not expose the literal value; AST avoids matching comments/logs.
    This is only a heuristic: dynamic secrets or compiled PYZ remain opaque.
    """
    try:
        tree = ast.parse(source)
    except (ValueError, SyntaxError, UnicodeError):
        return []
    issues: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        pairs: list[tuple[ast.AST, ast.AST]] = []
        if isinstance(node, ast.Assign):
            pairs = [(target, node.value) for target in node.targets]
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            pairs = [(node.target, node.value)]
        for target, value in pairs:
            if isinstance(target, ast.Name):
                variable = target.id
            elif isinstance(target, ast.Attribute):
                variable = target.attr
            else:
                continue
            if variable.lower().endswith(('password_hash', 'token_hash')):
                continue
            if not SECRETS_LIKE.search(variable):
                continue
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                literal = value.value.strip()
                if len(literal) >= 5 and not PLACEHOLDER.match(literal):
                    issues.append((node.lineno, variable))
    return issues


def audit(root: Path, installed: Path | None = None) -> AuditResult:
    root = Path(root).resolve()
    dist = root / 'dist' / 'AURUM-AI V5'
    setup = root / 'dist_installer' / EXPECTED_SETUP
    iss = root / 'installer' / 'AURUM_AI_V5.iss'
    output = AuditResult()

    if not (dist / EXPECTED_EXE).is_file():
        output.blockers.append('Executable PyInstaller absent dans dist/AURUM-AI V5')
    if not (dist / '_internal').is_dir():
        output.blockers.append('Dossier _internal absent : la distribution onedir est incomplete')
    if not setup.is_file():
        output.blockers.append(f'Installateur absent : dist_installer/{EXPECTED_SETUP}')
    else:
        output.good.append(f'Installateur present ({setup.stat().st_size / 1024**2:.1f} Mio)')
    if not iss.is_file():
        output.blockers.append('Script Inno Setup absent : installer/AURUM_AI_V5.iss')
    else:
        content = iss.read_text(encoding='utf-8-sig', errors='replace')
        if 'Source: "..\\dist\\AURUM-AI V5\\*"' not in content:
            output.blockers.append('Source [Files] Inno Setup inattendue : inspecter les fichiers empaquetes')
        if not re.search(r'PrivilegesRequired\s*=\s*lowest\b', content, flags=re.I):
            output.warnings.append('PrivilegesRequired=lowest absent : verifier la portee de l’installation')
        if re.search(r'\[UninstallDelete\][\s\S]*?(?:\{localappdata\}\\AURUM-AI|users\.sqlite3)', content, re.I):
            output.blockers.append('Regle de desinstallation suspecte : pourrait effacer les comptes persistants')
        if 'recursesubdirs' not in content.lower():
            output.blockers.append('Inno Setup doit inclure les sous-dossiers _internal')
        if 'Excludes:' not in content:
            output.blockers.append('Exclusions explicites manquantes dans Inno Setup')

    if dist.is_dir():
        scanned = 0
        backups = 0
        for path in _walk_regular_files(dist):
            rel = path.relative_to(dist)
            scanned += 1
            if path.is_symlink():
                output.blockers.append(f'Lien symbolique a verifier : {rel.as_posix()}')
                continue
            reason = sensitive_file(rel)
            if reason:
                output.blockers.append(f'Fichier sensible potentiellement embarque : {rel.as_posix()} ({reason})')
            if is_legacy_python(rel):
                backups += 1
                if backups <= 12:
                    output.blockers.append(f'Ancienne copie Python embarquee : {rel.as_posix()}')
            if path.suffix.lower() == '.py' and path.stat().st_size <= 2_000_000:
                try:
                    issues = literal_secret_assignment(path.read_text(encoding='utf-8-sig'))
                except (OSError, UnicodeError):
                    issues = []
                for line, name in issues:
                    output.blockers.append(
                        f'Secret litteral potentiel dans fichier embarque : {rel.as_posix()}:{line} ({name}, valeur masquee)'
                    )
        if backups > 12:
            output.blockers.append(f'{backups - 12} autres copies Python historiques embarquees')
        output.good.append(f'Fichiers controles dans dist : {scanned}')

    launcher = root / 'tools' / 'windows_desktop' / 'frozen_entry.py'
    if launcher.is_file():
        launcher_content = launcher.read_text(encoding='utf-8-sig', errors='replace')
        if not re.search(r"\bHOST\s*=\s*['\"]127\.0\.0\.1['\"]", launcher_content):
            output.blockers.append('Connexion locale non confirmee : HOST doit etre 127.0.0.1 dans frozen_entry.py')
        else:
            output.good.append('Lanceur : HOST=127.0.0.1 dans le code (runtime non teste)')
        if '--server.address=' not in launcher_content:
            output.blockers.append('Adresse ecoute Streamlit non definie explicitement dans le lanceur')
    else:
        output.warnings.append('frozen_entry.py absent : adresse de liaison locale a verifier manuellement')

    # Review project auth source: these modules may be packed into PYZ as well as datas.
    auth = root / 'frontend' / 'auth'
    if auth.is_dir():
        for module in sorted(auth.glob('*.py')):
            if is_legacy_python(module.relative_to(auth)):
                continue
            if module.stat().st_size > 2_000_000:
                continue
            for line, name in literal_secret_assignment(module.read_text(encoding='utf-8-sig', errors='replace')):
                output.blockers.append(
                    f'Secret litteral potentiel dans source auth : frontend/auth/{module.name}:{line} ({name}, valeur masquee)'
                )
    else:
        output.warnings.append('Sources frontend/auth absentes : audit du code authentification incomplet')

    if installed is None:
        local = os.environ.get('LOCALAPPDATA')
        installed = Path(local) / 'Programs' / 'AURUM-AI V5' if local else None
    if installed is not None:
        installed = Path(installed)
        inst_exe = installed / EXPECTED_EXE
        if inst_exe.is_file() and (dist / EXPECTED_EXE).is_file():
            if sha256(inst_exe) == sha256(dist / EXPECTED_EXE):
                output.good.append('Executable installe : empreinte SHA-256 identique a dist')
            else:
                output.blockers.append('Executable installe et executable dist differents (versions divergentes)')
        elif installed.exists():
            output.blockers.append('Installation detectee mais executable attendu absent')
        else:
            output.warnings.append('Installation Windows non detectee : comparaison installe/dist impossible')

    local = os.environ.get('LOCALAPPDATA')
    if local:
        persistent = Path(local) / 'AURUM-AI' / 'auth' / 'users.sqlite3'
        if persistent.is_file():
            output.good.append('Base auth persistante externe au dossier installe detectee (contenu NON lu)')
        else:
            output.warnings.append('Base auth persistante non trouvee : tester la creation/connexion sur Windows')

    output.warnings.append('Ce diagnostic NE prouve PAS l’absence de secrets dans PYZ/EXE ou les dependances.')
    output.warnings.append('Avant partage : machine Windows propre + exports + deconnexion + lock 5 min + desinstallation.')
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='AURUM-AI Phase 25 - audit de distribution en lecture seule')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--installed-dir', type=Path, default=None)
    args = parser.parse_args(argv)
    result = audit(args.root, args.installed_dir)
    print('AURUM-AI V5 | Phase 25 - audit de distribution (lecture seule)')
    print('AUCUN mot de passe, hash ou donnees utilisateurs ne sont affiches.')
    for msg in result.good:
        print('OK :', msg)
    for msg in result.blockers[:40]:
        print('BLOQUANT :', msg)
    if len(result.blockers) > 40:
        print(f'BLOQUANT : {len(result.blockers)-40} autres constats non affiches')
    for msg in result.warnings:
        print('A VERIFIER :', msg)
    print(f'Resultat : {len(result.blockers)} bloquant(s), {len(result.warnings)} avertissement(s).')
    return 0 if result.safe_to_proceed else 2


if __name__ == '__main__':
    raise SystemExit(main())
