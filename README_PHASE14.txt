AURUM-AI V5 — PHASE 14 / BLOCKCHAIN EXPLORER
================================================

OBJECTIF
- Nouvelle page autonome et sécurisée (n'utilise plus v5_legacy/executive_dashboard.py)
- Consultation, recherche, inspection et export des blocs historiques
- Contrôle d'intégrité effectué par BlockchainLedger.verify_integrity()
- Vérification des empreintes des fichiers d'origine par comparaison à audit_summary.json
- Lecture seule, simulation académique, aucune écriture sur les datasets ou sur le ledger

FICHIERS AJOUTÉS / REMPLACÉS
- frontend/v5_pages/blockchain_explorer.py (remplace la passerelle 389 octets)
- frontend/services/blockchain_explorer_audit.py (nouveau)
- tests/test_blockchain_explorer_audit.py (nouveau)

INSTALLATION (PowerShell, depuis le dossier racine AURUM-AI)
1) Sauvegarder l'ancien Explorer :
   Copy-Item .\frontend\v5_pages\blockchain_explorer.py .\frontend\v5_pages\blockchain_explorer_before_phase14.py

2) Décompresser l'archive avec -DestinationPath . -Force
   ATTENTION : le ZIP doit réellement exister dans le dossier Downloads avant la commande.

3) Lancer les tests :
   python -m pytest tests/test_blockchain_explorer_audit.py tests/test_blockchain_ledger_audit.py tests/test_blockchain_ledger.py tests/test_blockchain_integration.py -q

4) Démarrer l'app :
   python -m streamlit run frontend/app.py

5) Ouvrir Blockchain Explorer dans le menu.

ARCHITECTURE
- Le fichier blockchain/ledger.py reste inchangé.
- Les artefacts historiques attendus sont :
  reports/blockchain/consensus_trend_60m_validation/ledger.json
  reports/blockchain/consensus_trend_60m_validation/audit_summary.json
  reports/backtesting/combined_holding_30m_60m/{60m_decisions.csv,60m_transactions.csv,comparison_summary.json}
- Les exports sont générés à la volée en mémoire.

LIMITES
- SHA-256 local n'est pas une blockchain distribuée ni un certificat d'authenticité indépendant.
- L'audit de SHA-256 dépend de la confiance dans les fichiers sources et le code local.
- Le rapport compare les octets des fichiers à leurs empreintes stockées; il ne redémarre
  pas les modèles ni les agents et ne valide pas la rentabilité.
- Vérifiez visuellement la page et exécutez les tests dans VOTRE environnement AURUM-AI.
