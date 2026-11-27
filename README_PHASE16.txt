AURUM-AI V5 — Phase 16.1 / Research Center / Library
====================================================

INSTALLATION (dans la racine AURUM-AI, terminal PowerShell + .venv actif)

1. Faire une copie du fichier actuel :
   Copy-Item .\frontend\v5_pages\library.py .\frontend\v5_pages\library_before_phase16.py

2. Extraire ce ZIP à la racine du projet, en conservant les dossiers :
   Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\AURUM_AI_V5_Library_Phase16.zip" -DestinationPath . -Force

3. Exécuter les tests :
   python -m pytest tests/test_research_library_audit.py -q

4. Lancer l'application :
   python -m streamlit run frontend/app.py

5. Cliquer sur "Library" dans le volet Research Center.

NOTES
- Package limité à la page Library + service isolé + tests associés.
- Ne modifie pas Experiments, Reports, data/01_RAW ou autres pages existantes.
- Le contenu JSON/CSV/texte est lu uniquement sur la ressource sélectionnée.
- Pas de recherche dans .venv, auth, cookies, secrets, données RAW.
- Un SHA-256 est calculé uniquement après avoir cliqué sur le bouton.
- Certains documents restent volontairement hors catalogue pour éviter l'exposition de données sensibles.
- Les fichiers .npz/.parquet et checkpoints sont indexés mais non chargés en RAM pour l'aperçu.
- Les avertissements Streamlit use_container_width des autres pages restent indépendants de cette phase.
