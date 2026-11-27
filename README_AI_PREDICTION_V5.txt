AURUM-AI V5 — Phase 5 : AI Prediction, read-only Validation
========================================================

INSTALLATION (PowerShell, depuis la racine AURUM-AI)

1. Arrêter Streamlit avec Ctrl+C.
2. Sauvegarder :
   Copy-Item .\frontend\v5_pages\ai_prediction.py .\frontend\v5_pages\ai_prediction_before_phase5.py
3. Extraire cette archive à la RACINE du projet :
   Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\AURUM_AI_V5_AIPrediction.zip" -DestinationPath . -Force
4. Tests :
   python -m pytest tests/test_ai_prediction_engine.py -q
5. Lancer :
   python -m streamlit run frontend/app.py
6. Ouvrir l'onglet "AI Prediction".

Fichiers ajoutés/remplacés :
- frontend/v5_pages/ai_prediction.py (remplace uniquement le placeholder)
- frontend/services/ai_prediction_engine.py (nouveau, lecture seule)
- tests/test_ai_prediction_engine.py (nouveau)

Prérequis : npz existants sous artifacts/experiments/nyse_v3/{model}_{horizon}m_seed42/validation_predictions.npz
Les 12 expériences sont disponibles selon les sorties du terminal de l'utilisateur.
Ce module ne charge PAS de checkpoints, ne réalise PAS d'inférence live, n'utilise PAS le jeu Test,
et n'écrit PAS dans le dataset RAW ou les artefacts d'entraînement.

LABEL : y_true / y_pred sont des rendements logarithmiques (ln(next_close/current_close)).
Les prix prédits dérivés sont = current_close * exp(y_pred).
Les métriques sont calculées uniquement sur la Validation, comparées à une baseline Zero Return.
Les contrôles temporels sont structuraux ; ils ne constituent pas une preuve exhaustive anti-fuite.
