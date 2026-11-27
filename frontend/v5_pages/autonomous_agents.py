"""AURUM-AI V5 — historical Autonomous Agents research viewer (read-only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.services.autonomous_agents_audit import (
    available_experiments, load_validation_npz, signal_decisions,
    signal_summary, timing_audit, price_target_consistency,
    read_historical_ledger,
)

EXP_ROOT = ROOT / "artifacts" / "experiments" / "nyse_v3"
LEDGER = ROOT / "reports" / "blockchain" / "consensus_trend_60m_validation" / "ledger.json"


@st.cache_data(show_spinner=False)
def load_cached(filename: str, mtime_ns: int):
    return load_validation_npz(Path(filename))


@st.cache_data(show_spinner=False)
def read_ledger_cached(filename: str, mtime_ns: int):
    return read_historical_ledger(Path(filename))


def main():
    st.markdown('''<style>
    .aurum-agents-head {background:linear-gradient(100deg,#0b2943,#061829);border:1px solid #20628a;
      border-radius:16px;padding:22px 26px;margin:1rem 0 0.7rem 0;color:#edf7ff}
    .aurum-agents-head .kicker{color:#39cdfd;font-size:11px;letter-spacing:3px;font-weight:750}
    .aurum-agents-head h1{font-size:34px;margin:12px 0 6px;color:#f5f9ff}
    .aurum-agents-head .sub{color:#b6d9f3;margin:0}
    </style>''', unsafe_allow_html=True)
    st.markdown('''<div class="aurum-agents-head"><div class="kicker">◇ AURUM-AI V5 · RESEARCH AGENTS</div>
    <h1>🤖 Autonomous Agents</h1><p class="sub">Forecast → Strategy → Risk → Virtual Execution → Portfolio</p></div>''', unsafe_allow_html=True)
    st.info("Mode recherche, lecture seule. Prévisions Validation historiques ; aucune transaction, aucun ordre, aucun réentraînement. Les décisions de stratégie affichées ici sont des PROPOSITIONS et non des exécutions.")

    experiments = available_experiments(EXP_ROOT)
    if not experiments:
        st.error(f"Aucune prédiction historique trouvée sous {EXP_ROOT}. Les artefacts Validation sont nécessaires.")
        st.stop()

    models = sorted({m for m, h in experiments}, key=lambda x: ["LSTM","GRU","CNN1D"].index(x))
    controls = st.columns([2, 1, 1])
    with controls[0]:
        model = st.selectbox("Modèle de régression", models, index=models.index("CNN1D") if "CNN1D" in models else 0, key="aa_model")
    horizons = sorted(h for m, h in experiments if m == model)
    with controls[1]:
        horizon = st.selectbox("Horizon (min)", horizons, index=(horizons.index(60) if 60 in horizons else 0), key="aa_horizon")
    with controls[2]:
        threshold = st.number_input("Seuil stratégique (log-return)", min_value=0.00001, max_value=0.01,
                                    value=0.00030, step=0.00005, format="%.5f", key="aa_threshold")

    prediction_path = experiments[(model, horizon)]
    try:
        arrays = load_cached(str(prediction_path), prediction_path.stat().st_mtime_ns)
        frame = signal_decisions(arrays, threshold)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        st.error(f"Chargement ou vérification impossible : {exc}")
        st.stop()
    stats = signal_summary(frame)

    c = st.columns(5)
    c[0].metric("Observations Validation", f"{stats['observations']:,}".replace(",", " "))
    c[1].metric("BUY proposés", f"{stats['BUY']:,}".replace(",", " "))
    c[2].metric("SELL proposés", f"{stats['SELL']:,}".replace(",", " "))
    c[3].metric("HOLD proposés", f"{stats['HOLD']:,}".replace(",", " "))
    c[4].metric("MAE rendements log", f"{stats['mae_log_return']:.8f}")
    st.caption(f"Source : {prediction_path.relative_to(ROOT)} · Le seuil 0,00030 correspond à la configuration de StrategyAgent dans test_strategy_with_cnn1d.py ; les autres seuils sont des scénarios exploratoires.")

    tab_workflow, tab_signals, tab_ledger, tab_checks, tab_export = st.tabs([
        "🧭 Pipeline", "📈 Forecast → Strategy", "🔗 Risque & historique", "🛡️ Contrôles", "📄 Export recherche"
    ])
    with tab_workflow:
        st.subheader("Architecture existante (code source)")
        for label, desc in [
            ("1 · Forecast", "12 fichiers validation_predictions.npz : LSTM, GRU, CNN1D sur 5/15/30/60 minutes"),
            ("2 · StrategyAgent", "Rendement log prédit : BUY si au-dessus du seuil, SELL en dessous de son opposé, sinon HOLD"),
            ("3 · ClassificationStrategyAdapter", "Pipeline DISTINCT : prédictions classifiées 0 DOWN / 1 NEUTRAL / 2 UP, en tenant compte des positions"),
            ("4 · RiskAgent", "Appelé depuis agents/execution_agent.py ; le code du module risk/risk_agent.py n'était pas inclus dans l'archive fournie"),
            ("5 · ExecutionAgent", "Accepte des classes, consulte le risque et applique des transactions uniquement à un portefeuille VIRTUEL"),
            ("6 · PortfolioAgent", "Tient les liquidités, les quantités, les frais et les rendements simulés"),
        ]:
            with st.expander(label, expanded=False):
                st.write(desc)
        st.warning("Important : nous n'envoyons PAS les rendements régressifs de cette page directement à l'ExecutionAgent qui attend une classe 0/1/2. Aucune simulation de nouveaux ordres n'est effectuée ici.")

    with tab_signals:
        st.subheader("Propositions de décision sur les prédictions historiques")
        counts = pd.DataFrame({"Décision": ["BUY", "SELL", "HOLD"],
                               "Observations": [stats["BUY"], stats["SELL"], stats["HOLD"]]})
        fig = go.Figure(go.Bar(x=counts["Décision"], y=counts["Observations"],
                               marker_color=["#25d6a0", "#ff7184", "#37bafa"]))
        fig.update_layout(template="plotly_dark", paper_bgcolor="#061628", plot_bgcolor="#061628",
                          height=315, margin=dict(l=20,r=20,t=22,b=30), yaxis_title="Observations")
        st.plotly_chart(fig, width="stretch")
        n = st.slider("Dernières propositions à consulter", 10, 150, 30, 10, key="aa_n")
        st.dataframe(frame.tail(n).iloc[::-1], width="stretch", hide_index=True)
        st.caption("Les actions sont dérivées du modèle choisi et du seuil affiché. Elles ne tiennent compte ni de la position, ni des contraintes de risque, ni des coûts de transaction.")

    with tab_ledger:
        st.subheader("Décisions historiques consignées dans le ledger")
        st.caption("Source DISTINCTE du pipeline de prédictions sélectionné : aucune correspondance événement par événement n'est supposée.")
        if LEDGER.is_file():
            try:
                ledger_df = read_ledger_cached(str(LEDGER), LEDGER.stat().st_mtime_ns)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                st.warning(f"Impossible de lire le ledger : {exc}")
            else:
                st.metric("Blocs interprétés", len(ledger_df))
                if not ledger_df.empty:
                    statuses = ledger_df["risk_status"].replace("", pd.NA).dropna().value_counts().rename_axis("Risk Status").reset_index(name="Événements")
                    if not statuses.empty:
                        st.write("Statuts Risk Agent enregistrés")
                        st.dataframe(statuses, width="stretch", hide_index=True)
                    types = ["Tous"] + sorted(str(x) for x in ledger_df["event_type"].dropna().unique())
                    event = st.selectbox("Filtrer les événements", types, key="aa_event")
                    selected = ledger_df if event == "Tous" else ledger_df[ledger_df["event_type"].astype(str) == event]
                    st.dataframe(selected.tail(150).iloc[::-1], width="stretch", hide_index=True)
                    st.caption("Lecture seule : cette page ne revalide pas la chaîne cryptographique. Utiliser Blockchain Explorer pour les vérifications d'intégrité.")
                else:
                    st.info("Ledger chargé, mais aucun bloc exploitable trouvé.")
        else:
            st.warning(f"Ledger historique introuvable : {LEDGER.relative_to(ROOT)}. Cette section sera disponible lorsqu'il sera présent.")
        st.info("Le code de l'agent de risque n'a pas été fourni dans ce ZIP. Pour rejouer une vraie chaîne Forecast → Strategy → Risk → Execution sans modifier les résultats existants, une validation séparée du pipeline est nécessaire.")

    with tab_checks:
        st.subheader("Intégrité des prédictions Validation")
        checks = timing_audit(arrays, horizon)
        checks["log_returns_inconsistent_with_prices"] = price_target_consistency(arrays)
        st.dataframe(pd.DataFrame({"Contrôle": list(checks), "Nombre": list(checks.values())}),
                     width="stretch", hide_index=True)
        st.metric("MAE Zero Return (baseline)", f"{stats['zero_return_mae']:.8f}")
        st.metric("Précision directionnelle brute", f"{stats['direction_accuracy_percent']:.2f}%")
        if any(v for k, v in checks.items() if k != "observations"):
            st.warning("Des écarts ont été détectés dans les métadonnées. Vérifier le générateur de labels et les sources avant d'interpréter les résultats.")
        else:
            st.success("Aucune anomalie détectée par ces contrôles ciblés.")
        st.caption("Ces tests ne démontrent pas l'absence totale de data leakage ni une rentabilité future ; le découpage Train/Validation et les normalisations doivent être audités séparément.")

    with tab_export:
        st.subheader("Export des propositions en lecture seule")
        st.download_button("📥 Télécharger les signaux CSV (Validation)", data=frame.to_csv(index=False).encode("utf-8"),
                           file_name=f"aurum_{model.lower()}_{horizon}m_strategy_proposals_validation.csv",
                           mime="text/csv", key="aa_csv")
        metadata = {"project":"AURUM-AI V5", "split":"validation", "mode":"READ_ONLY_NO_EXECUTION",
                    "experiment":prediction_path.parent.name,"threshold_log_return":float(threshold),
                    "metrics":stats,"timing_audit":timing_audit(arrays,horizon),
                    "warning":"Strategy proposals only. No risk approval, no execution, no portfolio update."}
        st.download_button("📥 Télécharger le manifeste JSON", data=json.dumps(metadata, indent=2).encode("utf-8"),
                           file_name=f"aurum_{model.lower()}_{horizon}m_agent_audit.json", mime="application/json", key="aa_manifest")


main()
