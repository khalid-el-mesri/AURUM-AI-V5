AURUM-AI V5 — Data Quality phase 2
==================================

S'INSTALLE PAR-DESSUS V5 PHASE 1 (installation déjà validée)

FICHIERS
  frontend/v5_pages/data_quality.py        Page Streamlit Data Quality réelle
  frontend/services/data_quality_audit.py Moteur Python autonome, lecture seule
  tests/test_data_quality_audit.py        Tests unitaires

INSTALLATION (PowerShell, à la racine de AURUM-AI)

1. Arrêter Streamlit : Ctrl + C.
2. Sauvegarder la page existante :
   Copy-Item .\frontend\v5_pages\data_quality.py .\frontend\v5_pages\data_quality_before_phase2.py
3. Placer ce ZIP dans Downloads, puis :
   Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\AURUM_AI_V5_DataQuality.zip" -DestinationPath . -Force
4. Exécuter les tests (si pytest est installé) :
   python -m pytest tests/test_data_quality_audit.py -q
5. Relancer l'application V5 :
   python -m streamlit run frontend/app.py
6. Se connecter, puis ouvrir 'Data Quality' dans la barre latérale.

FICHIER ATTENDU
  data/01_raw/xomususd-m5-bid-2017-01-26-2026-10-01.csv
  Colonnes : timestamp,open,high,low,close
  timestamp : entier Unix en millisecondes ; cadence nominale M5.

GARANTIES ET LIMITES
- Pas de modification des fichiers RAW, CLEAN ou des modèles déjà entraînés.
- Aucun chiffre d'audit pré-calculé : résultats calculés à partir du CSV sur ton PC.
- Les intervalles >5min sont décrits, pas automatiquement qualifiés de défauts.
- Le calendrier officiel NYSE, les jours fériés et la DST ne sont pas intégrés.
- L'analyse de fréquence suppose une grille UTC fixe à 5 minutes.
- Pour vérifier l'identité du CSV, calculer son SHA-256 dans Export Report.
- La protection de la page est assurée par le gateway V5 et demand_session().

Note : la fonctionnalité Streamlit st.plotly_chart(width='stretch') requiert une version récente
(la même famille que V5, qui avertit déjà sur use_container_width).
