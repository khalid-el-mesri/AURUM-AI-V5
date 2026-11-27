AURUM-AI V5 — Phase 22.1 — UI Blockchain / Arrow
==================================================

But
---
1. Blockchain Ledger : empêche l'en-tête HTML d'être traité comme du code Markdown
   grâce à textwrap.dedent(...) avant st.markdown(..., unsafe_allow_html=True).
2. Digital Twin : convertit en texte UNIQUEMENT la colonne 'Valeur' du tableau
   de contrôles affiché par Streamlit (mélange int / str / bool).
3. Experiments : même protection d'affichage pour les métadonnées de l'inspecteur.

N'affecte PAS : data/01_RAW, ledger SHA-256, base SQLite, modèles, entraînements,
rapports originaux, logique d'authentification.

PowerShell à la racine du projet, .venv actif :
  Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\AURUM_AI_V5_Phase22_1_BlockchainArrow_Fix.zip" -DestinationPath . -Force
  python -m pytest tests/test_phase22_1_ui_fix.py -q
  python tools/windows_desktop/fix_phase22_1_ui.py --check
  python tools/windows_desktop/fix_phase22_1_ui.py --apply
  python -m pytest -q --disable-warnings --maxfail=1
  python -m streamlit run frontend/app.py

Contrôler ensuite :
  /blockchain-ledger (titre visuel et fonds dégradés)
  /digital-twin -> onglet Contrôles de cohérence (aucun avertissement Arrow)
  /experiments -> Inspecteur (tableau de scalaires hétérogènes).

Chaque source .py modifié reçoit une copie .bak avec horodatage.
Une fois les vérifications passées, la construction PyInstaller reste à refaire
pour incorporer les changements dans l'EXE ; NE PAS reconstruire avant les tests.

NOTE : 'Get-Content' peut afficher des accents mal encodés suivant la page de code
PowerShell alors que les sources UTF-8 sont correctes. L'outil lit/écrit en UTF-8.
