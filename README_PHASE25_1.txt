AURUM-AI V5 — Phase 25.1 : Nettoyage de la distribution Windows
==================================================================

Cause du resultat Phase 25 : le builder PyInstaller incluait des copies
historiques .py via ses donnees et collectait certains modules frontend.auth
historiques dans les imports caches. Ces sources doivent rester sur le PC de
developpement mais ne doivent pas etre diffusees dans l'application.

Ce patch cible DEUX fichiers uniquement :
  tools/windows_desktop/build_windows_exe.py
  installer/AURUM_AI_V5.iss

Le script ne supprime pas les anciennes versions dans le projet, ne touche
PAS aux bases de comptes SQLite, aux donnees XOM ou aux fichiers d'experiences.
Une sauvegarde *.before_phase25_1_....bak est creee pour chaque fichier modifie.

1) Depuis la RACINE DU PROJET, avec .venv actif :

Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\AURUM_AI_V5_Phase25_1_PackagingHygiene.zip" -DestinationPath . -Force

python -m pytest tests/test_phase25_1_packaging_hygiene.py -q
python tools/windows_desktop/fix_phase25_1_packaging.py --check

2) Si les controles passent :

python tools/windows_desktop/fix_phase25_1_packaging.py --apply
python tools/windows_desktop/build_windows_exe.py --check

3) Fermer toute fenetre AURUM-AI et recompiler l'EXE, sans console :

python tools/windows_desktop/build_windows_exe.py --build

4) Verifier que les anciennes copies ont DISPARU du nouveau dist :

python tools/windows_desktop/fix_phase25_1_packaging.py --check

Attendu : Anciens fichiers Python actuellement presents dans dist : 0

5) Ne pas modifier la base de comptes; reconstruire l'installateur Inno Setup :

python tools/windows_desktop/installer_preflight_phase23.py
python tools/windows_desktop/build_installer_phase23.py

NB : Inno Setup genere a nouveau AURUM-AI-V5-Setup-0.9.0.exe. Il faut
l'installer pour remplacer l'ancienne version. Tant que l'ancienne version
reste installee, l'audit Phase 25 signalera une difference EXE installe/dist :
c'est normal apres recompilation et avant installation du nouveau setup.

6) Une fois le nouvel installateur reinstalle et valide sur le PC :

python tools/windows_desktop/audit_distribution_phase25.py

Attendu : 0 BLOQUANT seulement apres audit de la NOUVELLE distribution
et comparaison de la NOUVELLE installation. Les avertissements sur les secrets
cachés dans PYZ/EXE et le test de desinstallation restent valables.

SECURITE : ne pas distribuer publiquement avant un audit independant des
secrets, une verification sur PC Windows propre et les tests des modules.
Ne pas transmettre de base SQLite, de mot de passe, de token ou de .env.
