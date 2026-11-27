"""AURUM-AI V5 — historical Digital Twin, read only. Entry via frontend/app.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.services.digital_twin_audit import (
    ALT_EQUITY_REL, ALT_TRADES_REL, DECISIONS_REL, TRANSACTIONS_REL,
    alternate_audit, attach_recorded_trades, audit_consensus, event_replay,
    load_consensus, load_lstm_alternate, segmented_equity, summary_json,
)


@st.cache_data(show_spinner=False)
def _read_consensus(root: str, decisions_mtime: int, trades_mtime: int):
    return load_consensus(Path(root))


@st.cache_data(show_spinner=False)
def _read_alternate(root: str, equity_mtime: int, trades_mtime: int):
    return load_lstm_alternate(Path(root))


def _utc(ms: int | float) -> str:
    return pd.to_datetime(ms, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M UTC")


def _value(obj, default="Non enregistré") -> str:
    if obj is None or pd.isna(obj) or str(obj).strip() == "":
        return default
    return str(obj)


def _field_group(title: str, pairs: list[tuple[str, object]]):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        for label, obj in pairs:
            st.markdown(f"**{label} :** {_value(obj)}")


def _plot_recorded_equity(records: pd.DataFrame, title: str, name: str, value_key: str,
                          gap_days: float = 7.0) -> go.Figure:
    fig = go.Figure()
    if records.empty:
        return fig
    frame = records.copy()
    if "segment" not in frame:
        frame["segment"] = (frame["timestamp_ms"].diff().fillna(0) > gap_days * 86400000).cumsum()
    for n, (_, segment) in enumerate(frame.groupby("segment", sort=False)):
        fig.add_trace(go.Scattergl(
            x=segment["datetime_utc"], y=segment[value_key], mode="lines+markers"
            if len(segment) < 30 else "lines", name=name if n == 0 else f"{name} (segment {n + 1})",
            line=dict(color="#22c6f0", width=2), marker=dict(size=4), showlegend=n == 0,
            hovertemplate="%{x}<br>%{y:.2f} USD<extra></extra>",
        ))
    fig.update_layout(
        title=title, template="plotly_dark", paper_bgcolor="#061522", plot_bgcolor="#061522",
        height=440, margin=dict(l=30, r=20, t=60, b=32),
        xaxis_title="Horodatage historique UTC", yaxis_title="Capital simulé (USD)",
        legend=dict(orientation="h", y=1.07),
    )
    return fig


def main():
    st.markdown('''<style>
    .dt-hero{padding:24px 28px;background:linear-gradient(105deg,#10354f,#051522);
    border:1px solid #1d688c;border-radius:16px;margin:14px 0}
    .dt-hero .eyebrow{font-size:11px;color:#49d8f5;letter-spacing:2px;font-weight:750}
    .dt-hero h1{font-size:35px;color:#f4f9ff;margin:8px 0 5px}
    .dt-hero p{margin:0;color:#b6d4ea}
    </style>''', unsafe_allow_html=True)
    st.markdown('''<div class="dt-hero"><div class="eyebrow">◇ AURUM-AI V5 · SIMULATION INTELLIGENCE</div>
    <h1>🔁 Digital Twin</h1><p>Historical Replay · Agents · Virtual Portfolio · Temporal Audit</p></div>''',
    unsafe_allow_html=True)
    st.info("**Jumeau historique, lecture seule.** Aucun ordre n'est lancé. Le replay ne réexécute "
            "pas les agents : il affiche uniquement les décisions et états conservés. "
            "Les expériences Consensus 60m et LSTM 60m sont indépendantes et ne sont jamais fusionnées.")

    dpath = ROOT / DECISIONS_REL
    tpath = ROOT / TRANSACTIONS_REL
    missing = [str(p.relative_to(ROOT)) for p in (dpath, tpath) if not p.is_file()]
    if missing:
        st.error("Sources historiques Consensus + Trend 60m manquantes : " + ", ".join(missing))
        return
    try:
        decisions, trades = _read_consensus(str(ROOT), dpath.stat().st_mtime_ns, tpath.stat().st_mtime_ns)
        report = audit_consensus(decisions, trades)
        events = attach_recorded_trades(decisions, trades)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        st.error(f"Sources illisibles ou schéma incompatible : {exc}")
        return

    st.caption("Source principale : `reports/backtesting/combined_holding_30m_60m/` — modèle `Consensus + Trend 60m`.")
    a, b, c, d = st.columns(4)
    a.metric("Décisions enregistrées", f"{report['decisions']:,}".replace(",", " "))
    b.metric("Transactions simulées", f"{report['transactions']:,}".replace(",", " "))
    c.metric("Transactions reliées", report["matched_transactions"])
    d.metric("Événements EXECUTED", report["recorded_executed"])
    if (report["unmatched_trade_rows"] or report["unmatched_executed_decisions"]
            or report["cash_mismatches_gt_1_cent"] or report["decision_reverse_timestamps"]):
        st.warning("Le rapprochement des événements signale des anomalies. "
                   "Ne pas interpréter les étapes manquantes comme des transactions exécutées.")
    else:
        st.success("Rapprochement des transactions par horodatage et action : aucun écart détecté parmi les critères testés.")

    tabs = st.tabs([
        "📊 Executive Twin", "▶️ Historical Replay", "🤖 Agent Pipeline",
        "💼 Trade Tape", "🧪 LSTM (indépendant)", "🛡️ Data Integrity", "📥 Export",
    ])
    equity = segmented_equity(decisions)
    with tabs[0]:
        st.subheader("Capital virtuel enregistré — Consensus + Trend 60m")
        if equity.empty:
            st.warning("Aucune valeur de capital disponible.")
        else:
            st.plotly_chart(_plot_recorded_equity(
                equity, "Equity après les décisions historiques", "Equity enregistrée", "equity_after_usd"),
                width="stretch")
            st.caption("Points = états de portefeuille réellement enregistrés après une décision. "
                       "Les intervalles supérieurs à 7 jours sont représentés par des segments distincts. "
                       "Aucune donnée intermédiaire n'est interpolée.")
            first, last = equity.iloc[0], equity.iloc[-1]
            col1, col2 = st.columns(2)
            col1.metric("Premier capital reporté", f"{first['equity_after_usd']:,.2f} USD")
            col2.metric("Dernier capital reporté", f"{last['equity_after_usd']:,.2f} USD")
            st.caption("Ces valeurs ne sont pas présentées comme un rendement d'investissement : "
                       "le capital initial doit être attesté par l'expérience source.")
        st.subheader("Répartition des décisions consignées")
        st.bar_chart(decisions["strategy_action"].fillna("NON RENSEIGNÉ").value_counts())
        st.dataframe(decisions.tail(50).iloc[::-1], width="stretch", hide_index=True)

    # Replay selection shared across the panel but computed from one immutable local run.
    with tabs[1]:
        st.subheader("Rejeu chronologique — enregistrements existants")
        if events.empty:
            st.warning("Aucune décision enregistrée.")
        else:
            idx = st.slider("Position historique (1 = première décision)", 1, len(events), len(events),
                            key="digital_twin_replay_position") - 1
            selected = event_replay(decisions, trades, idx)
            st.caption(f"Événement {idx + 1}/{len(events)} · {_utc(selected['timestamp_ms'])}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Stratégie", _value(selected["strategy_action"]))
            col2.metric("Risk Status", _value(selected["risk_status"]))
            col3.metric("Execution Status", _value(selected["execution_status"]))
            col4.metric("Trade relié", "OUI" if selected["matched_trade"] else "NON")
            st.warning("`risk_timestamp_utc` est le temps d'audit du moteur, pas le temps de marché historique. "
                       "Il peut se situer en 2026 lorsque l'événement de marché est en 2022.")
            st.markdown("### Chaîne de décision observée")
            _field_group("1 · Forecast (enregistré)", [
                ("Classe prédite", selected["predicted_class"]),
                ("Timestamp de signal", selected["signal_timestamp_ms"]),
                ("Modèle source", selected["model"]),
            ])
            _field_group("2 · Strategy", [
                ("Signal", selected["strategy_action"]),
                ("Justification enregistrée", selected["strategy_reason"]),
            ])
            _field_group("3 · Risk / Kill Switch", [
                ("Statut", selected["risk_status"]),
                ("Raison enregistrée", selected["risk_reason"]),
                ("État du système", selected["system_state"]),
                ("Horodatage d'audit", selected["risk_timestamp_utc"]),
            ])
            _field_group("4 · Exécution virtuelle", [
                ("Statut historique", selected["execution_status"]),
                ("Motif", selected["execution_reason"]),
                ("Transaction correspondante", "Présente dans le CSV" if selected["matched_trade"] else "Non rapprochée"),
                ("Prix transaction USD", selected["trade_price"]),
                ("Quantité transaction", selected["trade_quantity"]),
                ("Frais USD", selected["trade_fee_usd"]),
            ])
            _field_group("5 · Portfolio virtuel", [
                ("Cash après décision USD", selected["cash_after_usd"]),
                ("Equity après décision USD", selected["equity_after_usd"]),
                ("Position après trade (si enregistré)", selected["trade_holdings_after"]),
                ("P&L réalisé (si enregistré)", selected["realized_pnl_usd"]),
            ])
            with st.expander("Événement sélectionné — données brutes de replay"):
                st.json(selected)
            st.download_button("📥 Télécharger cet événement (JSON)", data=summary_json(selected),
                               file_name=f"aurum_twin_event_{idx + 1}.json", mime="application/json",
                               key="twin_event_export")

    with tabs[2]:
        st.subheader("Pipeline historique des agents")
        st.caption("Les étapes ci-dessous décrivent les champs historiques consultés, et non un nouvel appel aux agents.")
        for name, description in [
            ("Forecast", "predicted_class, signal_timestamp_ms, model (lorsqu'enregistrés)"),
            ("Strategy", "strategy_action et strategy_reason enregistrés"),
            ("Risk", "risk_status, risk_reason et system_state historiques"),
            ("Virtual Execution", "execution_status ; preuve de transaction seulement si rapprochement CSV"),
            ("Portfolio", "cash_after_usd, equity_after_usd, positions connues via transactions"),
        ]:
            with st.container(border=True):
                st.markdown(f"**{name}**")
                st.caption(description)
        st.info("Les règles actuelles des agents peuvent différer de celles de l'expérience historique. "
                "Le replay ne recalculera pas les décisions pour les modifier.")

    with tabs[3]:
        st.subheader("Transactions de la même expérience — Consensus + Trend 60m")
        tape = trades.copy()
        tape["datetime_utc"] = pd.to_datetime(tape["timestamp_ms"], unit="ms", utc=True)
        actions = ["Toutes"] + sorted(_value(v) for v in tape["action"].dropna().unique())
        choice = st.selectbox("Filtrer les transactions", actions, key="twin_tape_filter")
        if choice != "Toutes":
            tape = tape[tape["action"].astype(str) == choice]
        st.dataframe(tape.iloc[::-1], width="stretch", hide_index=True)
        st.caption("Un événement EXECUTED n'est lié à une transaction que lorsque l'horodatage "
                   "et l'action concordent avec le CSV de cette même expérience.")

    with tabs[4]:
        st.subheader("Expérience distincte : NYSE V3 LSTM 60m — Validation")
        st.warning("Cette courbe LSTM et ses trades NE SONT PAS les résultats de Consensus + Trend 60m. "
                   "Aucune jointure ni attribution de performance entre les deux expériences.")
        ep, tp = ROOT / ALT_EQUITY_REL, ROOT / ALT_TRADES_REL
        if not ep.is_file() or not tp.is_file():
            st.info("Sources NYSE V3 LSTM 60m absentes ; section indisponible.")
        else:
            try:
                ae, at = _read_alternate(str(ROOT), ep.stat().st_mtime_ns, tp.stat().st_mtime_ns)
                ar = alternate_audit(ae, at)
                st.metric("Points de courbe enregistrés", ar["equity_observations"])
                st.metric("Trades LSTM enregistrés", ar["trade_rows"])
                tmp = ae.copy()
                tmp["datetime_utc"] = pd.to_datetime(tmp["timestamp_ms"], unit="ms", utc=True)
                tmp["segment"] = (tmp["timestamp_ms"].diff().fillna(0) > 7 * 86400000).cumsum()
                st.plotly_chart(_plot_recorded_equity(
                    tmp, "LSTM 60m Validation — equity de cette expérience uniquement",
                    "LSTM Validation", "equity_usd"), width="stretch")
                st.dataframe(at, width="stretch", hide_index=True)
                st.caption("Le capital affiché provient uniquement de `lstm60m_validation_dashboard/equity_curve.csv`.")
            except (OSError, ValueError, TypeError, KeyError) as exc:
                st.error(f"Impossible de lire les artefacts LSTM : {exc}")

    with tabs[5]:
        st.subheader("Contrôles de cohérence — même expérience")
        table = pd.DataFrame([{"Contrôle": str(k), "Valeur": str(v)} for k, v in report.items()])
        st.dataframe(table, width="stretch", hide_index=True)
        st.caption("Les rapprochements sont exacts sur l'horodatage marché et l'action, avec numéro d'occurrence "
                   "pour les doublons éventuels. Un rapprochement réussi ne certifie ni toutes les règles d'exécution "
                   "ni l'absence totale de data leakage.")
        if report["cash_mismatches_gt_1_cent"]:
            st.error("Des montants cash divergent de plus d'un centime : vérifier la provenance des exports.")

    with tabs[6]:
        st.subheader("Exports en mémoire — sans écrire sur les datasets")
        st.download_button("📥 Événements Consensus avec rapprochements (CSV)",
                           data=events.to_csv(index=False).encode("utf-8-sig"),
                           file_name="aurum_twin_consensus_events.csv", mime="text/csv", key="twin_csv_events")
        st.download_button("📥 Audit de rapprochement (JSON)",
                           data=summary_json(report), file_name="aurum_twin_audit.json",
                           mime="application/json", key="twin_json_audit")
        st.caption("Aucun fichier source n'est réécrit. Les valeurs non enregistrées ne sont pas inventées.")


main()
