AURUM-AI V5 | PHASE 4 — FEATURE ENGINEERING EXPLORER
=====================================================

INSTALLATION (PowerShell, à la racine de AURUM-AI)

1. Arrêter Streamlit : Ctrl+C.
2. Sauvegarder la page existante :
   Copy-Item .\frontend\v5_pages\feature_engineering.py .\frontend\v5_pages\feature_engineering_before_phase4.py
3. Extraire le ZIP téléchargé à la racine du projet :
   Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\AURUM_AI_V5_FeatureEngineering.zip" -DestinationPath . -Force
4. Exécuter :
   python -m pytest tests/test_feature_engineering_audit.py -q
5. Démarrer :
   python -m streamlit run frontend/app.py
6. Se connecter puis ouvrir Feature Engineering.

Le ZIP apporte UNIQUEMENT :
- frontend/v5_pages/feature_engineering.py (remplace la page vide)
- frontend/services/feature_engineering_audit.py (nouveau moteur lecture seule)
- tests/test_feature_engineering_audit.py (nouveaux tests)
- README_FEATURE_ENGINEERING_V5.txt

NE REMPLACE PAS :
- frontend/app.py, auth/, modèles, pipelines Deep Learning,
- data/01_raw, 02_clean, 03_features, 04_windows, 05_model_data.

Les fichiers Parquet historiques observés :
- xom_features.parquet : 161023 lignes / 13 colonnes
- xom_features_v2.parquet : 161023 / 17
- xom_features_nyse_v3.parquet : 161001 / 18

La différence de 22 lignes reste à expliquer via le pipeline de construction.
Les fenêtres existantes portent X, y et des horodatages ;
leur simple inspection ne prouve pas l'absence de data leakage.
Les transformations fit/normalisation Train seul devront être vérifiées
à partir des scripts de préparation avant de certifier cet aspect.

DÉPENDANCES : pandas, numpy, plotly, streamlit, pyarrow, pytest (tests).
Si pyarrow manque : python -m pip install pyarrow

Aucun nouvel entraînement. Aucun write sur les fichiers source.
Les exports de rapport JSON et d'un échantillon CSV sont servis en mémoire.
