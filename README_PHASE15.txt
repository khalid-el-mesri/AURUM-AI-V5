AURUM-AI V5 — PHASE 15 : AI MODEL LAB
=====================================

Objectif
--------
Exploiter en lecture seule les prédictions historiques VALIDATION déjà entraînées
sur XOM (NYSE V3) : LSTM, GRU, CNN1D ; horizons 5, 15, 30, 60 minutes.

Prérequis :
- L'installation des phases précédentes, notamment :
  frontend/services/ai_prediction_engine.py (Phase 5)
  frontend/v5_pages/_common.py
  frontend/app.py
- Les fichiers artifacts/experiments/nyse_v3/{modele}_{horizon}m_seed42/
  validation_predictions.npz et, si disponible, report.json.
- numpy, pandas, plotly, Streamlit, pytest dans .venv.

Installation (PowerShell, depuis la racine AURUM-AI)
----------------------------------------------------
1. Enregistrer AURUM_AI_V5_AIModelLab_Phase15.zip dans Téléchargements.
2. Sauvegarder la page précédente :
   Copy-Item .\frontend\v5_pages\ai_model_lab.py .\frontend\v5_pages\ai_model_lab_before_phase15.py
3. Installer le ZIP :
   Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\AURUM_AI_V5_AIModelLab_Phase15.zip" -DestinationPath . -Force
4. Lancer les tests :
   python -m pytest tests/test_ai_model_lab_audit.py tests/test_ai_prediction_engine.py -q
5. Lancer AURUM-AI :
   python -m streamlit run frontend/app.py
6. Ouvrir http://localhost:8501/ai-model-lab

Le ZIP remplace uniquement :
  frontend/v5_pages/ai_model_lab.py
  frontend/services/ai_model_lab_audit.py
  tests/test_ai_model_lab_audit.py
et ajoute ce README_PHASE15.txt à la racine.

Fonctionnalités
---------------
- Inventaire de 12 expériences Validation et de leurs rapports JSON.
- MAE/RMSE calculées depuis les vrais NPZ ; Zero Return sur les mêmes lignes.
- Diagnostics d'erreurs, extrêmes, MAE, RMSE et accord directionnel descriptif.
- Séries UTC observées/prédites et prix futurs reconstruits.
- Audit structurel temporel hérité de Phase 5 ; métadonnées seulement.
- Empreintes SHA-256 locales, lecture des report.json, exports CSV/JSON en mémoire.

Garde-fous
----------
- Aucune valeur inventée ; les expériences absentes/corrompues sont signalées.
- Aucune donnée du jeu Test, aucun réentraînement, aucun ordre, aucune transaction.
- Les étiquettes sont des log-rendements ; le gain vs Zero Return n'implique
  ni rentabilité, ni absence de fuite de données, ni généralisation Test.
- Les chiffres dans les copies d'écran de l'utilisateur n'ont pas été utilisés
  pour créer de nouveaux fichiers prédictions.
