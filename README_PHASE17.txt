AURUM-AI V5 — Phase 17 / Research Center / Experiments
========================================================

BUT
Construire le registre en lecture seule des 12 expériences historiques
NYSE V3 (LSTM, GRU, CNN1D × 5/15/30/60 minutes). Le module recense
les fichiers existants, inspecte les report.json, compare les métadonnées,
et exporte un manifeste sans lancer de training ou modifier le projet.

INSTALLATION — PowerShell depuis la RACINE du projet AURUM-AI

1) Sauvegarder l'ancienne page :
   Copy-Item .\frontend\v5_pages\experiments.py .\frontend\v5_pages\experiments_before_phase17.py

2) Télécharger ce ZIP dans C:\Users\Pc\Downloads, puis extraire :
   $zip = "$env:USERPROFILE\Downloads\AURUM_AI_V5_Experiments_Phase17.zip"
   Test-Path $zip
   Expand-Archive -LiteralPath $zip -DestinationPath . -Force

3) Tests (venv actif) :
   python -m pytest tests/test_research_experiments_audit.py -q

4) Démarrer :
   python -m streamlit run frontend/app.py

5) Dans l'interface : Research Center > Experiments.

NOTES METHODOLOGIQUES
- Ne modifie ni Library, Reports, AI Model Lab, ni d'autres pages V5.
- Aucun fichier dans data/01_RAW ou autre dossier de données n'est ouvert.
- 12 runs attendus, même si certains reports ou NPZ manquent.
- Les chemins sont fixés et les liens symboliques ne sont pas suivis.
- Les fichiers NPZ et poids de modèles ne sont jamais ouverts par la page.
- Les rapports JSON sont limités à 2 Mo et leurs clés sensibles masquées.
- Les résultats sont historiques et issus de Validation, pas de Test.
- Les SHA-256 demandés sur la page vérifient des octets locaux : ils ne
  démontrent ni l'authenticité externe ni l'absence de fuite temporelle.
- Exports JSON/CSV créés uniquement en mémoire : aucune écriture projet.
- Un test symlink peut être IGNORÉ sous Windows sans privilège symlink.

SI LE ZIP N'EST PAS DANS DOWNLOADS :
Chercher son chemin exact avant de lancer Expand-Archive :
Get-ChildItem "$env:USERPROFILE\Downloads" -Filter '*Experiments*zip'
