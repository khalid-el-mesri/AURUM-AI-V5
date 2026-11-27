"""AURUM-AI V5 — Risk Management | historical read-only observatory.

Entry point: python -m streamlit run frontend/app.py
The application router imports this file. No agent is called on historical rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.services.risk_management_audit import (
    RUN_DIR, DECISIONS_FILE, TRADES_FILE, SUMMARY_FILE,
    load_sources, exact_matches, audit_history, risk_timeline, report_json,
)


@st.cache_data(show_spinner=False)
def _get_data(root: str, mtime_d: int, mtime_t: int, mtime_s: int):
    return load_sources(Path(root))


def _chart_equity(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for n, (_, seg) in enumerate(frame.groupby("segment", sort=False)):
        fig.add_trace(go.Scatter(
            x=seg["market_time_utc"], y=seg["equity_after_usd"],
            mode="lines+markers" if len(seg) <= 15 else "lines",
            name="Equity après décision" if n == 0 else "Segment séparé",
            showlegend=n == 0,
            line={"color": "#25D2EA", "width": 2}, marker={"size": 5},
            hovertemplate="%{x}<br>%{y:.2f} USD<extra></extra>",
        ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#061524", plot_bgcolor="#061524",
        height=375, margin={"l": 14, "r": 12, "t": 15, "b": 55},
        xaxis_title="Horodatage historique (UTC)",
        yaxis_title="Equity enregistrée (USD)",
        legend={"orientation": "h", "y": 1.10},
    )
    return fig


def _flat_table(mapping: dict) -> pd.DataFrame:
    return pd.DataFrame([{"Indicateur": str(k), "Valeur": str(v)} for k, v in mapping.items()])


def main() -> None:
    st.markdown("""
    <style>
    .rm-hero{padding:24px 28px;background:linear-gradient(110deg,#113B53,#061626);
      border:1px solid #1A7195;border-radius:16px;margin:10px 0 12px}
    .rm-eye{font-size:11px;color:#56DFF9;letter-spacing:2px;font-weight:750}
    .rm-hero h1{font-size:35px;color:#F4F9FF;margin:8px 0 6px}
    .rm-hero p{color:#BDD6E7;margin:0}
    </style>""", unsafe_allow_html=True)
    st.markdown("""<div class="rm-hero"><div class="rm-eye">◇ AURUM-AI V5 · RISK INTELLIGENCE</div>
    <h1>🛡️ Risk Management</h1>
    <p>Risk Agent V2 · Persistent Kill Switch · Validation Audit · Research Only</p></div>""",
    unsafe_allow_html=True)
    st.info("**Lecture seule : Consensus + Trend 60m / Validation.** Les événements viennent des rapports "
            "historiques ; aucun agent de trading ou moteur d'exécution n'est relancé. "
            "Les contrôles ne prouvent pas la rentabilité ni l'absence de data leakage.")
    base = ROOT / RUN_DIR
    paths = [base / DECISIONS_FILE, base / TRADES_FILE, base / SUMMARY_FILE]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        st.error("Fichiers nécessaires absents : " + ", ".join(missing))
        return
    try:
        decisions, trades, summary = _get_data(str(ROOT), *(p.stat().st_mtime_ns for p in paths))
        audit = audit_history(decisions, trades, summary)
        joined, orphan = exact_matches(decisions, trades)
        timeline = risk_timeline(decisions)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        st.error(f"Impossible d'auditer l'expérience historique : {exc}")
        return

    st.caption("Source : `reports/backtesting/combined_holding_30m_60m/` — "
               "`60m_decisions.csv`, `60m_transactions.csv`, `comparison_summary.json`.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Décisions historiques", str(audit["decisions"]))
    c2.metric("Trades virtuels rapprochés", f'{audit["matched"]}/{audit["transactions"]}')
    c3.metric("Max drawdown rapporté", f'{audit["max_observed_drawdown_pct"]:.3f} %')
    c4.metric("État final enregistré", audit["recorded_system_final"])
    if sum(audit["checks"].values()) == 0:
        st.success("Rapprochement des transactions, frais et capital final : aucun écart selon les contrôles implémentés.")
    else:
        st.warning("Anomalies de rapprochement signalées : consulter l'onglet Audit. "
                   "Ne pas interpréter les événements manquants comme des transactions.")

    tabs = st.tabs(["📊 Risk Overview", "🛡️ Risk Agent", "🔒 Kill Switch",
                    "🕒 Risk Timeline", "🔍 Audit", "📥 Export"])

    with tabs[0]:
        st.subheader("Exposition historique et capital simulé")
        a, b, c = st.columns(3)
        a.metric("Capital initial (summary)", f'{audit["initial_capital_usd"]:,.2f} USD')
        b.metric("Capital final (summary)", f'{audit["final_equity_usd"]:,.2f} USD')
        c.metric("Rendement historique (summary)", f'{audit["net_return_pct"]:+.3f} %')
        st.plotly_chart(_chart_equity(timeline), width="stretch")
        st.caption("Points de capital uniquement après les décisions consignées. Les intervalles de plus "
                   "de 7 jours ne sont pas connectés. Le drawdown ci-dessus provient du résumé du backtest "
                   "(observations 5 min), pas de cette série parcellaire.")
        st.markdown("#### Statuts de risque enregistrés")
        st.bar_chart(pd.Series(audit["status_counts"], name="Événements"))
        st.caption("Les statuts APPROVED / NO_ACTION sont issus du rapport historique, "
                   "pas d'une nouvelle validation des ordres.")

    with tabs[1]:
        st.subheader("Risk Agent V2 — limites implémentées")
        limits = audit["limits"]
        ca, cb, cc, cd = st.columns(4)
        ca.metric("Ordre BUY maximal", f'{100*limits["max_order_fraction"]:.1f} %')
        cb.metric("Exposition maximale", f'{100*limits["max_exposure_fraction"]:.1f} %')
        cc.metric("Perte journalière", f'{100*limits["max_daily_loss_fraction"]:.1f} %')
        cd.metric("Drawdown maximal", f'{100*limits["max_drawdown_fraction"]:.1f} %')
        st.markdown("#### Règles vérifiées dans `risk/risk_agent.py`")
        st.markdown("""- **HOLD** : renvoie `NO_ACTION` sans créer d'ordre.
- **BUY** : contrôle la taille positive de l'ordre, la perte journalière, le drawdown, la limite par ordre et l'exposition après ordre.
- **SELL** : autorisé pour réduire une position existante, même après dépassement de pertes, mais la quantité demandée ne doit pas dépasser les titres détenus.
- **Données invalides** : fractions non finies ou hors intervalle [0, 1] rejetées.
""")
        st.warning("Le CSV historique ne contient pas les snapshots complets "
                   "`daily_loss_fraction`, `current_exposure_fraction`, `proposed_order_fraction` "
                   "et `held_fraction` utilisés par RiskAgent. Ces limites sont la **configuration "
                   "du code actuel**, pas une preuve indépendante que chaque décision ancienne y satisfaisait.")
        st.dataframe(_flat_table(audit["status_counts"]), width="stretch", hide_index=True)
        st.markdown("#### Justifications de risque consignées")
        rationales = decisions.groupby(["risk_status", "risk_reason"], dropna=False).size().reset_index(name="Événements")
        st.dataframe(rationales, width="stretch", hide_index=True)

    with tabs[2]:
        st.subheader("Persistent Kill Switch — état historique")
        ca, cb, cc = st.columns(3)
        ca.metric("État final (summary)", audit["recorded_system_final"])
        cb.metric("Transitions consignées", str(len(audit["kill_switch_transitions"])))
        cc.metric("Ordres BLOCKED consignés", str(audit["blocked_recorded"]))
        if audit["halt_timestamp_ms"] is None:
            st.info("Aucun instant de HALT n'est enregistré dans le résumé de cette expérience. "
                    "Cela ne démontre pas l'absence de risque futur.")
        else:
            st.warning(f'Arrêt consigné : {pd.to_datetime(audit["halt_timestamp_ms"], unit="ms", utc=True)}')
        st.markdown("#### Comportement défini dans le code existant")
        st.markdown("""1. Passage de `ACTIVE` à `HALTED` lorsque le drawdown atteint **10 %**.
2. L'état `HALTED` reste actif même si le drawdown redescend (pas de reset automatique).
3. Les nouveaux `BUY` sont bloqués ; l'adaptateur traduit `HALTED` en `REJECTED` pour l'ExecutionAgent.
4. Les `SELL` réducteurs restent soumis aux vérifications de détention du Risk Agent.
""")
        st.dataframe(_flat_table(audit["state_counts"]), width="stretch", hide_index=True)
        if audit["kill_switch_transitions"]:
            st.dataframe(pd.DataFrame(audit["kill_switch_transitions"]), width="stretch", hide_index=True)
        st.caption("Ce panneau montre les états **enregistrés** et la logique du code. Il ne modifie "
                   "pas l'état d'un Kill Switch actif, et n'injecte aucun choc hypothétique dans le backtest.")

    with tabs[3]:
        st.subheader("Chronologie des décisions — temps de marché")
        choices = list(range(len(joined)))
        position = st.slider("Événement à explorer (1 = plus ancien)", 1, len(joined), len(joined),
                             key="rm_event_position") - 1
        item = joined.iloc[position]
        row_a, row_b, row_c, row_d = st.columns(4)
        row_a.metric("Action", str(item["strategy_action"]))
        row_b.metric("Risk Status", str(item["risk_status"]))
        row_c.metric("Execution Status", str(item["execution_status"]))
        row_d.metric("Trade rapproché", "OUI" if item["matched_trade"] else "NON")
        st.caption(f'Événement {position+1}/{len(joined)} · {item["market_time_utc"]} (UTC)')
        st.info("`risk_timestamp_utc` correspond au moment de l'audit de l'agent (souvent en 2026), "
                "et non à l'horodatage de marché (par exemple 2022).")
        observed = {
            "Timestamp marché (ms)": item["timestamp_ms"],
            "Horodatage d'audit": item["risk_timestamp_utc"],
            "Action stratégie": item["strategy_action"],
            "Risk Status": item["risk_status"],
            "Risk Reason": item["risk_reason"],
            "System State": item["system_state"],
            "Execution Status": item["execution_status"],
            "Trade correspondant": "Oui" if item["matched_trade"] else "Non",
            "Frais rapprochés (USD)": item.get("matched_fee_usd"),
            "Capital après décision (USD)": item["equity_after_usd"],
        }
        st.dataframe(_flat_table({k: "Non renseigné" if pd.isna(v) else v
                                  for k, v in observed.items()}), width="stretch", hide_index=True)
        filters = st.selectbox("Filtrer le journal par statut de risque",
                               ["Tous"] + sorted(decisions["risk_status"].dropna().unique().tolist()),
                               key="rm_status_filter")
        journal = joined if filters == "Tous" else joined.loc[joined["risk_status"] == filters]
        st.dataframe(journal[["market_time_utc", "strategy_action", "risk_status", "risk_reason",
                              "execution_status", "system_state", "matched_trade"]].tail(100).iloc[::-1],
                     width="stretch", hide_index=True)

    with tabs[4]:
        st.subheader("Audit de cohérence — même simulation 60m")
        checks = audit["checks"]
        st.dataframe(pd.DataFrame([{"Contrôle": k, "Anomalies": v} for k, v in checks.items()]),
                     width="stretch", hide_index=True)
        st.write(f'Frais observés dans les transactions : **{audit["fee_total_usd"]:,.4f} USD**')
        st.write("Méthode : appariement strict des transactions aux décisions `EXECUTED` "
                 "par `(timestamp_ms, action, occurrence)` ; tolérance sur le cash : 0,01 USD.")
        st.warning("**Limites :** le CSV n'expose ni les pertes journalières calculées à chaque bougie, "
                   "ni tous les snapshots utilisés par les agents. Le résumé indique que ce test "
                   "sur Validation est exploratoire, non indépendant, et utilise des hypothèses de frais "
                   "et de prix idéalisées. Aucun contrôle ici ne démontre l'absence de fuite de données.")
        if audit["validation_limitations"]:
            with st.expander("Limitations de l'expérience fournies dans le résumé"):
                for statement in audit["validation_limitations"]:
                    st.write("• " + str(statement))

    with tabs[5]:
        st.subheader("Exports historiques — sans modification du projet")
        st.download_button("📥 Décisions et risques (CSV)",
                           data=decisions.to_csv(index=False).encode("utf-8-sig"),
                           file_name="aurum_risk_60m_decisions.csv", mime="text/csv",
                           key="rm_export_decisions")
        st.download_button("📥 Journal rapproché (CSV)",
                           data=joined.to_csv(index=False).encode("utf-8-sig"),
                           file_name="aurum_risk_60m_reconciled.csv", mime="text/csv",
                           key="rm_export_reconciled")
        st.download_button("📥 Rapport d'audit (JSON)",
                           data=report_json(audit),
                           file_name="aurum_risk_60m_audit.json", mime="application/json",
                           key="rm_export_json")
        st.caption("Les fichiers RAW, les états Risk/Kill Switch et les expériences historiques "
                   "ne sont jamais modifiés par cette page.")


main()
