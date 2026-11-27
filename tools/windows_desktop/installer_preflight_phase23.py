"""Non-mutating installer readiness audit for a PyInstaller --onedir distribution.

This verifies *files*, not the absence of secrets embedded in code/binaries, the
success of interactive workflows, or the safety of distributing an installer.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

EXCLUDED_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".p12", ".pfx"}
EXCLUDED_NAMES = {"secrets.toml", ".env"}
REQUIRED = ("dist/AURUM-AI V5/AURUM-AI V5.exe", "assets/branding/aurum_ai.ico", "installer/AURUM_AI_V5.iss")


@dataclass(frozen=True)
class Audit:
    ok: bool
    problems: tuple[str, ...]
    excluded: tuple[str, ...]
    file_count: int


def is_excluded(relative_path: Path) -> bool:
    name = relative_path.name.lower()
    return name in EXCLUDED_NAMES or name.startswith(".env.") or relative_path.suffix.lower() in EXCLUDED_SUFFIXES


def audit(root: Path) -> Audit:
    root = Path(root).resolve()
    problems: list[str] = []
    for entry in REQUIRED:
        if not (root / entry).is_file():
            problems.append(f"Fichier obligatoire absent: {entry}")
    dist_dir = root / "dist" / "AURUM-AI V5"
    excluded: list[str] = []
    n = 0
    if dist_dir.is_dir():
        for path in dist_dir.rglob("*"):
            if not path.is_file():
                continue
            n += 1
            rel = path.relative_to(dist_dir)
            if is_excluded(rel):
                excluded.append(str(rel))
        if n <= 1:
            problems.append("Distribution incomplet: aucune ressource de support (_internal) detectee")
        if not (dist_dir / "_internal").is_dir():
            problems.append("Dossier PyInstaller _internal introuvable (build --onedir requis)")
    script = root / "installer" / "AURUM_AI_V5.iss"
    if script.is_file():
        source = script.read_text(encoding="utf-8-sig")
        required_patterns = ("recursesubdirs", "createallsubdirs", "Excludes:", "PrivilegesRequired=lowest", "{localappdata}")
        problems.extend(f"Directive installateur manquante: {p}" for p in required_patterns if p not in source)
        # Check for some dangerous source selectors. The installer must only include dist.
        for forbidden in ('Source: "..\\runtime', 'Source: "..\\.venv', 'Source: "..\\frontend'):
            if forbidden.lower() in source.lower():
                problems.append(f"Source de fichiers non autorisee: {forbidden}")
    return Audit(not problems, tuple(problems), tuple(excluded), n)


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    result = audit(root)
    print("AURUM-AI V5 | preflight installeur Phase 23 (lecture seule)")
    print(f"Fichiers dans la distribution : {result.file_count}")
    print(f"Fichiers sensibles exclus par les regles d'installation : {len(result.excluded)}")
    for f in result.excluded[:15]:
        print(f"  exclusion : {f}")
    for problem in result.problems:
        print(f"ERREUR: {problem}")
    if result.ok:
        print("Structure du dossier OK (cela ne valide pas les comportements runtime ni les secrets caches dans le code).")
        print("IMPORTANT : avant toute diffusion, auditer manuellement secrets, SMTP, donnees privees et tests de l'EXE.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
