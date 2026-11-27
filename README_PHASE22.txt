AURUM-AI V5 — PHASE 22 / FINITION VISUELLE + SMOKE AUDIT
================================================================
Ce package N'EST PAS encore un installateur Windows. Il n'ecrase pas les pages existantes ;
il apporte 2 outils opt-in, 14 tests et une checklist manuelle.

Objectif
  - masquer le menu Deploy via API officielle Streamlit client.toolbarMode = minimal
  - afficher le taureau officiel + le wordmark sur l'ecran de connexion
  - verifier statiquement les 19 routes protegees + 1 page d'authentification
  - pre-valider les ressources de dist apres la prochaine compilation

PREPARATION : Fermer toutes les fenetres AURUM-AI ; conserver le dossier dist actuel.
Ouvrir PowerShell dans C:\Users\Pc\Desktop\projet de deep learning\AURUM-AI avec .venv actif.

1) Installer ce ZIP a la racine (aucun fichier d'authentification n'est ecrase par le ZIP):

 $zip = "$env:USERPROFILE\Downloads\AURUM_AI_V5_Phase22_DesktopFinishing.zip"
 Test-Path $zip
 Expand-Archive -LiteralPath $zip -DestinationPath . -Force

2) Tests independants :

 python -m pytest tests/test_phase22_desktop_polish.py -q

3) Audit sans ecriture :

 python tools/windows_desktop/polish_desktop_phase22.py --check
 python tools/windows_desktop/desktop_smoke_audit.py

4) Si l'audit est OK, appliquer les DEUX modifications (sauvegardes .bak automatiques) :

 python tools/windows_desktop/polish_desktop_phase22.py --apply

5) Verifier d'abord l'interface en mode Python (sans reconstruire) :

 python -m pytest -q --disable-warnings --maxfail=1
 python -m streamlit run frontend/app.py

Inspecter l'ecran de login (taureau, pas de losange) et le dashboard (Deploy absent).
Fermer le serveur Streamlit avant la recompilation.

6) UNIQUEMENT APRES verification de la version Python, construire une seule fois :

 python tools/windows_desktop/build_windows_exe.py --check
 python tools/windows_desktop/build_windows_exe.py --build --console

7) Verifier les ressources dans dist (apres build) :

 python tools/windows_desktop/desktop_smoke_audit.py --packaged
 & ".\dist\AURUM-AI V5\AURUM-AI V5.exe"

8) Tester manuellement chaque module dans l'EXE avec
 docs/CHECKLIST_DESKTOP_PHASE22.md

SECURITE / LIMITES
  - Aucun script de cette phase ne modifie les fichiers SQLite, les identifiants,
    les artefacts, les donnees XOM, ni le code de calcul.
  - Outil de patch conservateur : s'arrete sans modifier si le login a ete personnalise
    d'une maniere qui ne correspond pas au header prevu.
  - Le patch cree une sauvegarde .bak avant chaque modification.
  - Le controle statique ne signifie PAS que les modules sont fonctionnels dans l'EXE.
  - Ne pas inclure runtime/auth/users.sqlite3, secrets.toml, .env, credentials,
    tokens, journaux ou donnees confidentielles dans un installateur public.
  - Changer le mot de passe partage precedemment sur une capture apres la connexion.
  - La plateforme reste exclusivement historique et simulee.
