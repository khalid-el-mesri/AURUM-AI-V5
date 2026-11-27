"""AURUM-AI V5 Phase 11 — Stress Testing, research-only diagnostic interface.

Launch the existing main router: python -m streamlit run frontend/app.py
Never import/re-run an ExecutionAgent here; active user risk state is untouched.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.services.risk_management_audit import (RUN_DIR, DECISIONS_FILE, TRADES_FILE, SUMMARY_FILE)
from frontend.services.stress_testing_engine import (
    PRESETS, additional_cost_stress, historical_inputs, paired_trades,
    trade_streak_report, simulate_shock_path, json_report,
)
from risk.risk_agent import RiskLimits


@st.cache_data(show_spinner=False)
def _load(root: str, m1: int, m2: int, m3: int):
    return historical_inputs(Path(root))


def _line(df: pd.DataFrame, y: str, x: str, title: str, ytitle: str, color: str = "#32DCE9"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x], y=df[y], mode="lines+markers", line={"color": color, "width": 2.5},
                             marker={"size": 5}, name=title,
                             hovertemplate="%{x}<br>%{y:,.3f}<extra></extra>"))
    fig.update_layout(template="plotly_dark", paper_bgcolor="#061726", plot_bgcolor="#061726",
                      height=360, margin={"l": 10, "r": 15, "t": 22, "b": 35}, showlegend=False,
                      xaxis_title=x, yaxis_title=ytitle,
                      xaxis={"gridcolor": "#193347"}, yaxis={"gridcolor": "#193347"})
    return fig


def _text_table(dic: dict):
    return pd.DataFrame([{"Indicateur": str(k), "Valeur": str(v)} for k,v in dic.items()])


def main():
    st.markdown("""<style>
    .st-hero{padding:24px 27px;background:linear-gradient(115deg,#102F46,#061426);
      border:1px solid #1878A0;border-radius:15px;margin:6px 0 12px}
    .st-overline{font-size:11px;letter-spacing:2px;font-weight:800;color:#4ADFF5}
    .st-hero h1{font-size:34px;margin:8px 0 7px;color:#F1F7FF}
    .st-hero p{margin:0;color:#B9D6E8}
    </style>""", unsafe_allow_html=True)
    st.markdown("""<div class="st-hero"><div class="st-overline">◇ AURUM-AI · RISK RESILIENCE LAB</div>
    <h1>⚡ Stress Testing</h1>
    <p>Historical Validation Audit · Counterfactual Costs · Synthetic Market Scenarios · Isolated Kill Switch</p>
    </div>""", unsafe_allow_html=True)
    st.info("**Lecture seule — aucune exécution réelle ou simulée du moteur de trading.** Les scénarios de choc sont "
            "**hypothétiques**, pas des observations historiques. Les frais supplémentaires sont une analyse de "
            "sensibilité sur les mêmes 90 opérations clôturées, sans recalculer les décisions des agents.")
    paths = [ROOT / RUN_DIR / n for n in (DECISIONS_FILE, TRADES_FILE, SUMMARY_FILE)]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        st.error("Fichiers Validation absents : " + ", ".join(missing))
        return
    try:
        decisions, trades, summary, audit = _load(str(ROOT), *(p.stat().st_mtime_ns for p in paths))
        pairs = paired_trades(trades)
        streak = trade_streak_report(pairs)
    except (ValueError, TypeError, KeyError, IndexError, OSError, AssertionError) as exc:
        st.error(f"Audit des données impossible : {exc}")
        return
    historical = summary["results"]["60m"]
    initial = float(historical["initial_capital_usd"])
    reference_price = float(trades.iloc[0]["price"])
    st.caption("Référence historique : `reports/backtesting/combined_holding_30m_60m/` · "
               "**Consensus + Trend 60m — Validation** · sans fusion avec NYSE V3 / LSTM.")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Trades clôturés historiques", str(len(pairs)))
    c2.metric("Rendement historique enregistré", f'{historical["net_return_pct"]:+.3f} %')
    c3.metric("Max DD historique rapporté", f'{historical["max_observed_drawdown_pct"]:.3f} %')
    c4.metric("Kill Switch historique", str(historical["final_system_state"]))
    st.success("Les décisions et transactions de la référence 60m sont rapprochées : "
               f'{audit["matched"]}/{audit["transactions"]}, aucun écart dans les contrôles de Phase 10.')
    tabs = st.tabs(["📊 Overview", "💸 Cost Drag", "🌪️ Shock & Regimes", "🔒 Kill Switch",
                    "📉 Loss Streaks", "🔍 Integrity", "📥 Export"])

    with tabs[0]:
        st.subheader("Base historique — aucune altération")
        a,b,c,d = st.columns(4)
        a.metric("Capital initial", f"{initial:,.2f} USD")
        b.metric("Capital final", f'{historical["final_equity_usd"]:,.2f} USD')
        c.metric("Frais déjà enregistrés", f'{historical["total_fees_usd"]:,.2f} USD')
        d.metric("Perte max. consécutive", f'{streak["max_consecutive_realized_losses"]} trades')
        base_frame, base_report = additional_cost_stress(pairs, initial, 0)
        st.plotly_chart(_line(base_frame, "equity_after_trade_usd", "trade_id", "Capital après clôture", "USD"), width="stretch")
        st.caption("Cette courbe est construite uniquement aux **clôtures des trades**. Son drawdown ne peut "
                   "remplacer le drawdown du résumé historique calculé sur les observations du backtest.")
        st.dataframe(pairs.tail(20), width="stretch", hide_index=True, height=310)

    with tabs[1]:
        st.subheader("Sensibilité aux frais de glissement supplémentaires")
        st.caption("Les frais historiques de 10 bps par côté sont déjà inclus dans le P&L enregistré. "
                   "Ce curseur ajoute des bps **en plus** à l'achat et à la vente, sans modifier prix, signaux ni taille des trades.")
        extra_bps = st.slider("Glissement additionnel par côté (basis points)", 0, 100, 10, step=5, key="st_extra_bps")
        adjusted, cost_report = additional_cost_stress(pairs, initial, extra_bps)
        p,q,r,s = st.columns(4)
        p.metric("Surcoût hypothétique", f'{cost_report["total_additional_cost_usd"]:,.2f} USD')
        q.metric("Capital final hypothétique", f'{cost_report["final_trade_close_equity_usd"]:,.2f} USD')
        r.metric("Rendement hypothétique", f'{cost_report["trade_close_return_pct"]:+.3f} %')
        s.metric("DD aux clôtures seulement", f'{cost_report["max_trade_close_drawdown_pct"]:.3f} %')
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=adjusted["trade_id"],y=initial+adjusted["realized_net_pnl_usd"].cumsum(),
                                 mode="lines",name="Historique aux clôtures",line={"color":"#26D9E6","width":2.3}))
        fig.add_trace(go.Scatter(x=adjusted["trade_id"],y=adjusted["equity_after_trade_usd"],
                                 mode="lines",name="Hypothétique + glissement",line={"color":"#FFBA52","width":2.3}))
        fig.update_layout(template="plotly_dark",paper_bgcolor="#061726",plot_bgcolor="#061726",
                          height=365,margin={"l":12,"r":12,"t":20,"b":35},xaxis_title="Trade clôturé",
                          yaxis_title="Capital aux clôtures (USD)",legend={"orientation":"h","y":1.1})
        st.plotly_chart(fig,width="stretch")
        st.dataframe(adjusted[["trade_id","realized_net_pnl_usd","extra_slippage_usd",
                                   "stressed_net_pnl_usd","equity_after_trade_usd"]].tail(30),
                     width="stretch",hide_index=True,height=300)
        st.warning("Cette sensibilité **ne réexécute pas** le Risk Agent ou Kill Switch. Elle ne constitue "
                   "pas un nouveau backtest avec prise en compte dynamique de la trésorerie.")

    with tabs[2]:
        st.subheader("Scénarios de marché — contrefactuels")
        a,b = st.columns(2)
        with a:
            scenario_name=st.selectbox("Trajectoire hypothétique",list(PRESETS),key="st_scenario")
        with b:
            exposure_pct=st.slider("Exposition XOM de départ (%)",5,25,25,step=5,key="st_exposure_pct")
        st.caption("Capital initial du scénario : 10 000 USD ; prix d'ancrage = premier achat observé "
                   f'({reference_price:,.3f} USD). L’exposition choisie est **hypothétique**, '
                   'pas l’exposition reconstituée de l’historique. Aucun coût ni rebalancement sur ce chemin.')
        shock_frame,shock_report=simulate_shock_path(PRESETS[scenario_name],exposure_pct/100,initial,reference_price)
        a,b,c = st.columns(3)
        a.metric("Capital final hypothétique",f'{shock_report["hypothetical_end_equity_usd"]:,.2f} USD')
        b.metric("Drawdown maximal hypothétique",f'{shock_report["hypothetical_max_drawdown_pct"]:.2f} %')
        c.metric("État du gate isolé",shock_report["isolated_final_gate_state"])
        st.plotly_chart(_line(shock_frame,"hypothetical_equity_usd","step",scenario_name,"Equity synthétique USD"),width="stretch")
        st.dataframe(shock_frame,width="stretch",hide_index=True,height=260)
        st.warning("Ce chemin synthétique ne contient pas de bougies OHLC, n'est pas une prévision, "
                   "et ne tient pas compte de liquidité, spreads, slippage, ou d'une modification de stratégie.")

    with tabs[3]:
        st.subheader("Kill Switch isolé — validation de la persistance")
        st.caption("Évaluation du code réel `RiskKillSwitch` dans une **nouvelle instance mémoire**, "
                   "sur les mêmes points du scénario sélectionné. Rien n'est envoyé au moteur historique.")
        limits=RiskLimits()
        x,y,z=st.columns(3)
        x.metric("Seuil de drawdown",f"{limits.max_drawdown_fraction*100:.0f} %")
        y.metric("Nouveaux BUY en fin de scénario",shock_report["hypothetical_new_buy_status"])
        z.metric("SELL réducteur en fin de scénario",shock_report["hypothetical_reducing_sell_status"])
        if shock_report["hypothetical_gate_transitions"]:
            st.error("Le seuil de drawdown est atteint dans ce scénario fictif. "
                     "Le Kill Switch passe à HALTED, sans réarmement automatique.")
            st.dataframe(pd.DataFrame(shock_report["hypothetical_gate_transitions"]),width="stretch",hide_index=True)
        else:
            st.info("Pas de transition HALTED avec ces paramètres ; un BUY peut malgré tout être "
                    "refusé par les autres règles (exposition, perte journalière, taille d'ordre).")
        st.dataframe(_text_table({"BUY (raison)":shock_report["hypothetical_new_buy_reason"],
                                  "SELL (raison)":shock_report["hypothetical_reducing_sell_reason"]}),
                     hide_index=True,width="stretch")
        st.caption("La perte journalière dans ce test correspond à la baisse du capital depuis le début "
                   "d'une session **entièrement hypothétique**. Ce n'est pas la perte journalière historique.")

    with tabs[4]:
        st.subheader("Séries de pertes réellement consignées — 60m Validation")
        a,b,c = st.columns(3)
        a.metric("Trades perdants", str(streak["losing_trades"]))
        b.metric("Pire série de pertes",str(streak["max_consecutive_realized_losses"]))
        c.metric("Pire P&L de trade",f'{streak["worst_trade_pnl_usd"]:,.2f} USD')
        st.plotly_chart(_line(pairs,"realized_net_pnl_usd","trade_id","P&L réalisé par trade","USD","#EFC16A"),width="stretch")
        st.info(f'La plus longue série de pertes occupe les trades n° '
                f'{streak["loss_streak_start_trade"]} à {streak["loss_streak_end_trade"]}. '
                "Ceci décrit les trades clôturés, pas l'évolution continue des pertes intraday.")
        st.dataframe(pairs[["trade_id","entry_time_utc","exit_time_utc","realized_net_pnl_usd"]],
                     width="stretch",hide_index=True,height=300)

    with tabs[5]:
        st.subheader("Intégrité de la simulation historique de référence")
        st.dataframe(pd.DataFrame([{"Contrôle":k,"Anomalies":int(v)} for k,v in audit["checks"].items()]),
                     width="stretch",hide_index=True)
        st.markdown("**Garde-fous méthodologiques**")
        st.write("• Stress de coûts : prix et quantités historiques figés ; aucun nouveau signal généré.")
        st.write("• Scénarios : chemins de prix inventés explicitement, et non présentés comme données XOM.")
        st.write("• Kill Switch : code réel invoqué uniquement dans un objet isolé sur snapshots synthétiques.")
        st.write("• Données Validation : exploration non indépendante, sans garantie de performance future.")
        with st.expander("Limites documentées de l'expérience originale"):
            for lim in summary.get("limitations",[]): st.write("• "+str(lim))

    with tabs[6]:
        st.subheader("Exports de recherche — sans écriture sur le projet")
        st.download_button("📥 Trades historiques clôturés (CSV)",pairs.to_csv(index=False).encode("utf-8-sig"),
                           file_name="aurum_stress_historical_60m_pairs.csv",mime="text/csv")
        st.download_button("📥 Sensibilité coûts (CSV)",adjusted.to_csv(index=False).encode("utf-8-sig"),
                           file_name="aurum_stress_cost_sensitivity.csv",mime="text/csv")
        st.download_button("📥 Scénario de choc (CSV)",shock_frame.to_csv(index=False).encode("utf-8-sig"),
                           file_name="aurum_stress_synthetic_shock.csv",mime="text/csv")
        obj={"phase":"11", "source":"Consensus + Trend 60m / Validation", "historical":audit,
             "historical_streak":streak,"extra_cost_stress":cost_report,
             "synthetic_scenario_name":scenario_name,"synthetic_shock":shock_report}
        st.download_button("📥 Rapport structuré (JSON)",json_report(obj),
                           file_name="aurum_stress_research_report.json",mime="application/json")
        st.caption("Le ZIP d'installation n'écrase que la page placeholder et ajoute les modules Phase 11. "
                   "Ces boutons ne modifient aucun historique.")


main()
