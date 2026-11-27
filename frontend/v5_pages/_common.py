"""V5 common display helpers; no financial values are invented."""
from __future__ import annotations
import streamlit as st
from frontend.auth.idle_guard import check_local_session

MODULES = {
    "market_analysis": ("Market Analysis", "📈", "Exploration des prix OHLC, rendements, volatilité et indicateurs XOM.", "Dataset brut/nettoyé + moteur d'indicateurs"),
    "data_quality": ("Data Quality", "🔍", "Audit des valeurs manquantes, doublons, gaps chronologiques et intégrité OHLC.", "Rapports Data Quality traçables"),
    "feature_engineering": ("Feature Engineering", "🧮", "Construction sans fuite temporelle de RSI, ATR, moyennes mobiles et fenêtres.", "Pipeline des features"),
    "ai_prediction": ("AI Prediction", "🎯", "Prévision à 15/30/60 minutes, avec réel vs prédit et métriques fiables.", "Prédictions exportées des modèles"),
    "autonomous_agents": ("Autonomous Agents", "🤖", "Prévision → stratégie → risque → exécution simulée → portefeuille.", "Moteur multi-agents validé"),
    "agent_decision_room": ("Agent Decision Room", "🗣️", "Décision X-Ray : raisons, accord/désaccord et refus par l'agent Risk.", "Journal des décisions multi-agents"),
    "digital_twin": ("Digital Twin", "🔁", "Vue du flux événementiel entre marché, IA, agents, risque et portefeuille.", "État de simulation / snapshots réels"),
    "portfolio_simulator": ("Portfolio Simulator", "💼", "Courbe d'équité, expositions, P&L et historique des trades simulés.", "Simulation portefeuille et snapshots"),
    "risk_management": ("Risk Management", "🛡️", "Drawdown, exposition, limites de pertes, règles d'approbation/rejet.", "Métriques et journaux Risk Agent"),
    "stress_testing": ("Stress Testing", "⚡", "Scénarios de forte volatilité, chute brutale et séries de pertes.", "Moteur de scénarios de stress"),
    "what_if_simulator": ("What-If Simulator", "🎛️", "Tester capital initial, seuils, coûts et gestion du risque en mode 100 % simulé.", "Recalcul déterministe des scénarios"),
    "blockchain_ledger": ("Blockchain Ledger", "🔗", "Journal des décisions signé par hash, chaîne et preuves d'intégrité.", "Écriture/lecture vérifiées du ledger"),
    "library": ("Library", "📁", "Inventaire des datasets, artefacts, modèles et documentation expérimentale.", "Index des artefacts du projet"),
    "experiments": ("Experiments", "🧪", "Paramètres, horizons, versions et comparaison reproductible des expériences.", "Registry des runs / rapports"),
    "reports": ("Reports", "📄", "Export des rapports IA, stratégie, risque, blockchain et portefeuille.", "Génération de rapports validée"),
}


def demand_session():
    if not check_local_session():
        st.error("Session non autorisée. Connectez-vous pour accéder à cette page.")
        st.stop()


def placeholder(slug):
    demand_session()
    title, emoji, description, requirement = MODULES[slug]
    st.markdown("""<style>
      .v5-intro {background:linear-gradient(125deg,#092840,#071827);border:1px solid #1d5576;
        padding:30px;border-radius:18px;margin:14px 0 24px 0}
      .v5-eyebrow {letter-spacing:3px;font-size:11px;font-weight:800;color:#43cfff}
      .v5-headline {font-size:39px;font-weight:900;color:#f0f7ff;margin-top:12px}
      .v5-sub {color:#bbd5e6;margin-top:8px;font-size:15px}
    </style>""", unsafe_allow_html=True)
    st.markdown(f"""<div class="v5-intro"><div class="v5-eyebrow">◇ AURUM-AI V5 · XOM HISTORICAL RESEARCH</div>
      <div class="v5-headline">{emoji} {title}</div><div class="v5-sub">{description}</div></div>""", unsafe_allow_html=True)
    st.info("**Module prévu — pas encore connecté au backend.** Aucun chiffre simulé ou inventé n'est affiché ici.")
    st.markdown("#### Étape d'intégration")
    st.write(requirement)
    st.caption("Plateforme académique en mode Simulation uniquement · Aucune transaction réelle.")
    if slug in {"risk_management", "portfolio_simulator", "blockchain_ledger", "experiments"}:
        st.markdown("Les résultats actuellement disponibles sont consultables dans les pages **Executive Dashboard**, **Backtesting · NYSE V3** ou **Blockchain Explorer** de la navigation.")
