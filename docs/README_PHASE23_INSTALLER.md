# AURUM-AI V5 — Phase 23 : installateur Windows (preview 0.9.0)

Ce package installe **un script de compilation Inno Setup**. Ce ZIP n'est PAS l'installateur final : celui-ci doit etre construit sur Windows a partir du dossier `dist/AURUM-AI V5` deja genere par PyInstaller.

## Conditions prealables

- Windows 10/11 x64, Python `.venv` actif pour les controles.
- `dist/AURUM-AI V5/AURUM-AI V5.exe` ET le dossier `_internal` existent.
- AURUM-AI V5 fonctionne en tant que fenetre native et permet connexion, navigation, exports et deconnexion.
- `assets/branding/aurum_ai.ico` existe.
- Inno Setup 6 est installe (compiler `ISCC.exe`), voir https://jrsoftware.org/isinfo.php.

## Installation du package de build

Depuis la racine du projet AURUM-AI :

```powershell
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\AURUM_AI_V5_Phase23_InstallerKit.zip" -DestinationPath . -Force
python -m pytest tests/test_phase23_installer.py -q
python tools/windows_desktop/installer_preflight_phase23.py
python tools/windows_desktop/build_installer_phase23.py
```

Si le compilateur n'est pas trouve, definir le chemin absolu :

```powershell
$env:INNO_SETUP_ISCC = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
python tools/windows_desktop/build_installer_phase23.py
```

Resultat prevu : `dist_installer/AURUM-AI-V5-Setup-0.9.0.exe` (Windows uniquement).

## Securite et donnees

- L'installateur utilise UNIQUEMENT `dist/AURUM-AI V5`, pas les sources, `runtime/` du projet, `.venv` ni la base d'origine.
- Les fichiers `.sqlite3`, `.db`, `.sqlite`, `.env*`, `secrets.toml`, `.p12`, `.pfx` sont exclus de l'installateur.
- Les comptes utilisateurs doivent rester dans `%LOCALAPPDATA%\AURUM-AI\auth\users.sqlite3`, hors dossier du programme.
- **La desinstallation ne supprime pas les comptes**, pour ne pas entrainer de perte de donnees.
- Le controle automatique est structurel : il ne detecte pas des secrets enfouis dans l'executable ou les modules Python compiles. Examiner les credentials SMTP, tokens et fichiers de config AVANT distribution a autrui.
- Pour les comptes test avec mot de passe precedemment partage dans une capture, changer les identifiants avant diffusion.
- Pas de transactions financieres reelles, plateforme de simulation historique.

## Tests sur Windows apres compilation

1. Installer le `AURUM-AI-V5-Setup-0.9.0.exe` sur un PC de test / compte Windows distinct.
2. Verifier logo taureau, raccourci dans le menu Demarrer et raccourci Bureau si selectionne.
3. Verifier que `AURUM-AI V5.exe` demarre SANS terminal ni navigateur apparent, sans erreur `ModuleNotFoundError`.
4. Verifier login / mail verification, les 19 routes authentifiees, verrouillage d'inactivite, deconnexion.
5. Verifier les sources embarquees pour la simulation historique XOM, les exports PDF/CSV/JSON, la chaine blockchain locale.
6. Refaire la verification avec Windows sans Python installe.
7. Desinstaller via Parametres Windows; s'assurer que l'application est desinstallee mais les comptes SQLite sont preserves.
8. Documenter toute dependance machine (WebView2, Visual C++ Redistributable si necessaire).

**Ne pas distribuer a autrui avant ces controles.** Le flag 'build complete' de PyInstaller n'est pas un test runtime.
