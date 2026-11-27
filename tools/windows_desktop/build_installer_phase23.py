"""Create an Inno Setup installer from an ALREADY BUILT PyInstaller --onedir app."""
from __future__ import annotations
import os
from pathlib import Path
import shutil
import subprocess
import sys


def locate_iscc() -> Path | None:
    manual = os.environ.get("INNO_SETUP_ISCC")
    if manual and Path(manual).is_file():
        return Path(manual)
    found = shutil.which("ISCC.exe") or shutil.which("iscc")
    if found:
        return Path(found)
    for var in ("ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA"):
        base = os.environ.get(var)
        if not base:
            continue
        for folder in ("Inno Setup 6", "Inno Setup 7", "Programs/Inno Setup 6"):
            candidate = Path(base) / folder / "ISCC.exe"
            if candidate.is_file():
                return candidate
    return None


def build(root: Path, compiler: Path) -> Path:
    script = root / "installer" / "AURUM_AI_V5.iss"
    subprocess.run([str(compiler), str(script)], cwd=str(root / "installer"), check=True)
    expected = root / "dist_installer" / "AURUM-AI-V5-Setup-0.9.0.exe"
    if not expected.is_file():
        raise RuntimeError(f"ISCC a termine mais fichier non trouve: {expected}")
    return expected


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from installer_preflight_phase23 import audit
    report = audit(root)
    if not report.ok:
        for problem in report.problems:
            print(f"ERREUR: {problem}")
        print("Installer non lance, corrigez les problemes structurels.")
        return 1
    compiler = locate_iscc()
    if compiler is None:
        print("ISCC.exe introuvable. Installer Inno Setup depuis https://jrsoftware.org/isinfo.php")
        print("Ou definir la variable d'environnement INNO_SETUP_ISCC vers ISCC.exe")
        return 2
    print(f"Compilateur: {compiler}")
    try:
        built = build(root, compiler)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Erreur de compilation : {exc}")
        return 3
    print(f"Installateur cree: {built}")
    print("NE PAS encore distribuer : tester installation / connexion / 20 modules / desinstallation sur PC de test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
