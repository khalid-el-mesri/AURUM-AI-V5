AURUM-AI V5 | Phase 6 — Autonomous Agents (Historical Research Viewer)
===================================================================

Fichiers à copier à la racine du projet AURUM-AI :
  frontend/v5_pages/autonomous_agents.py          <- remplace uniquement le placeholder existant
  frontend/services/autonomous_agents_audit.py    <- nouveau moteur d'analyse en lecture seule
  tests/test_autonomous_agents_audit.py           <- nouveaux tests

IMPORTANT : Le ZIP ne modifie aucun script original de agents/, risk/ ou deep_learning/.
Il utilise :
  agents/strategy_agent.py
  artifacts/experiments/nyse_v3/*/validation_predictions.npz
et, si présent :
  reports/blockchain/consensus_trend_60m_validation/ledger.json

Installation PowerShell depuis la racine (avec .venv) :
  Copy-Item .\frontend\v5_pages\autonomous_agents.py .\frontend\v5_pages\autonomous_agents_before_phase6.py
  Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\AURUM_AI_V5_AutonomousAgents.zip" -DestinationPath . -Force
  python -m pytest tests/test_autonomous_agents_audit.py -q
  python -m streamlit run frontend/app.py

Menu V5 > Autonomous Agents

SAFETY & ACADEMIC SCOPE
- Validation historique, pas de prédiction live, pas d'exécution d'ordre.
- Le StrategyAgent régresse des rendements en décisions BUY/SELL/HOLD sur un seuil (0.00030 par défaut).
- Le ClassificationStrategyAdapter + ExecutionAgent demande des classes 0/1/2, PAS des rendements bruts.
- Aucune connexion à un courtier. Aucun fichier historique modifié.
- Le panneau ledger affiche des décisions enregistrées mais n'effectue pas de vérification cryptographique.
- Les fichiers source fournis n'incluaient pas risk/risk_agent.py ; son code doit être inspecté avant toute simulation complète.
- Les contrôles ciblés ne certifient pas l'absence totale de data leakage.
