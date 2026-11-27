"""AURUM-AI V5 / Agent Decision Room — verified, read-only event X-ray.

No agent is invoked, no strategy or order is generated, no ledger file is written.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.services.decision_room_audit import (
    LEDGER_RELATIVE_PATH, block_table, chronology_audit, export_selected_event,
    extract_stage_data, ledger_summary, payload_evidence, read_verified_ledger,
    risk_limits_from_code,
)

LEDGER_FILE = ROOT / LEDGER_RELATIVE_PATH


@st.cache_data(show_spinner=False)
def load_verified(filename: str, mtime_ns: int) -> list[dict]:
    return read_verified_ledger(Path(filename))


def _info_value(obj: dict, key: str) -> str:
    value = obj.get(key)
    return "Non renseigné" if value is None or value == "" else str(value)


def _stage_panel(label: str, stage: dict, details: str):
    with st.container(border=True):
        st.markdown(f"**{label}**")
        if stage:
            st.json(stage, expanded=False)
        else:
            st.caption("Aucune information enregistrée dans ce bloc pour cette étape.")
        st.caption(details)


def main():
    st.markdown('''<style>
      .aurum-decision-hero{background:linear-gradient(105deg,#0e314e,#061522);border:1px solid #1f6286;
      border-radius:16px;padding:22px 26px;margin:12px 0 15px;color:#eaf5ff}
      .aurum-decision-hero .mini{font-size:11px;letter-spacing:3px;font-weight:750;color:#42cfff}
      .aurum-decision-hero h1{font-size:34px;color:#fff;margin:12px 0 6px}
      .aurum-decision-hero p{color:#bad8ef;margin:0}
      </style>''', unsafe_allow_html=True)
    st.markdown('''<div class="aurum-decision-hero"><div class="mini">◇ AURUM-AI V5 · HISTORICAL DECISION INTELLIGENCE</div>
    <h1>🎛️ Agent Decision Room</h1><p>Decision X-Ray · Risk Decisions · Kill Switch · Blockchain Audit</p></div>''',
                unsafe_allow_html=True)
    st.info("**Lecture seule / Simulation académique.** Les événements proviennent du ledger historique. "
            "Ils ne sont pas automatiquement liés aux prévisions LSTM, GRU ou CNN1D sélectionnées dans AI Prediction. "
            "Aucun ordre, aucune nouvelle décision et aucune transaction réelle ne sont générés ici.")

    if not LEDGER_FILE.is_file():
        st.warning(f"Ledger introuvable : `{LEDGER_RELATIVE_PATH.as_posix()}`. "
                   "La page n'affiche pas de résultats inventés ; vérifier la présence des artefacts historiques.")
        return

    try:
        blocks = load_verified(str(LEDGER_FILE), LEDGER_FILE.stat().st_mtime_ns)
    except (OSError, ValueError, TypeError, json.JSONDecodeError, KeyError) as exc:
        st.error("**Intégrité non confirmée.** Chargement refusé par la vérification du ledger. "
                 "Aucune décision historique n'est affichée comme fiable.")
        st.code(str(exc))
        return

    if not blocks:
        st.warning("Aucun bloc valide disponible.")
        return

    table = block_table(blocks)
    summary = ledger_summary(blocks)
    st.success("Chaîne locale SHA-256 vérifiée avec `BlockchainLedger.load_json()` (genesis, index, hash et chaînage). "
               "Ce contrôle établit l'intégrité du fichier selon les règles du projet, pas une preuve externe ou décentralisée.")

    cols = st.columns(4)
    cols[0].metric("Blocs vérifiés", f"{summary['block_count']:,}".replace(",", " "))
    cols[1].metric("Événements historiques", f"{summary['historical_events']:,}".replace(",", " "))
    cols[2].metric("Risk APPROVED enregistrés", summary["risk_approvals_recorded"])
    cols[3].metric("Execution EXECUTED enregistrés", summary["executed_status_recorded"])

    tabs = st.tabs(["🧭 Vue d'ensemble", "🔎 Decision X-Ray", "🛡️ Risk & Kill Switch",
                    "🕒 Chronologie", "🔗 Audit blockchain", "📥 Export"])

    with tabs[0]:
        st.subheader("Événements consignés")
        st.caption("Chaque entrée est un bloc du ledger historique, et non une nouvelle simulation.")
        if summary["event_types"]:
            counts = pd.Series(summary["event_types"], name="Nombre").sort_values(ascending=False)
            st.bar_chart(counts)
        event_types = ["Tous"] + sorted({str(x) for x in table["event_type"].dropna() if x != "GENESIS"})
        selected_type = st.selectbox("Type d'événement", event_types, key="decision_room_type")
        subset = table[table["event_type"] != "GENESIS"]
        if selected_type != "Tous":
            subset = subset[subset["event_type"] == selected_type]
        st.dataframe(subset.tail(250).iloc[::-1], width="stretch", hide_index=True)
        st.caption("Les statuts vides sont des données non enregistrées ; ils ne prouvent ni approbation ni exécution.")

    # Selection common to the X-Ray, Risk and Export views. Index comes from VERIFIED chain.
    historical = [b for b in blocks if b.get("event_type") != "GENESIS"]
    if not historical:
        for tab in tabs[1:]:
            with tab:
                st.info("Le ledger ne contient que le bloc GENESIS.")
        return
    indices = [b["index"] for b in historical]
    with tabs[1]:
        st.subheader("Sélectionner une décision historique")
        selected_index = st.selectbox(
            "Bloc enregistré", indices, index=len(indices) - 1,
            format_func=lambda i: f"#{i} · {blocks[i].get('event_type', '?')} · {blocks[i].get('timestamp_utc', '?')}",
            key="decision_room_block",
        )
        block = blocks[selected_index]
        stage_data = extract_stage_data(block)
        a, b, c = st.columns(3)
        a.metric("Action stratégique enregistrée", _info_value(stage_data["strategy"], "action"))
        b.metric("Risk Status enregistré", _info_value(stage_data["risk"], "status"))
        c.metric("Execution Status enregistré", _info_value(stage_data["execution"], "status"))
        st.dataframe(pd.DataFrame(payload_evidence(block)), width="stretch", hide_index=True)
        st.caption("« Non renseigné » signifie que le bloc ne contient pas ce champ, pas que l'agent n'a pas fonctionné.")
        _stage_panel("1 · Forecast", stage_data["forecast"], "Prévision enregistrée uniquement ; aucune prédiction recalculée.")
        _stage_panel("2 · Strategy", stage_data["strategy"], "Signal et justification enregistrés ; non reconstitués après coup.")
        _stage_panel("3 · Risk & Kill Switch", stage_data["risk"], "Résultat de risque historique, sans rejouer le moteur.")
        _stage_panel("4 · Execution virtuelle", stage_data["execution"], "Pas de nouvel ordre ; un statut manquant ne confirme aucune exécution.")
        _stage_panel("5 · Portfolio", stage_data["portfolio"], "Données du portefeuille seulement lorsqu'elles figurent dans le bloc.")
        with st.expander("Bloc intégral (preuve source)", expanded=False):
            st.json(block)

    with tabs[2]:
        st.subheader("Limites configurées dans RiskAgent V2")
        try:
            limits = risk_limits_from_code()
        except (OSError, ImportError, ValueError, TypeError) as exc:
            st.error(f"Impossible de lire les limites du Risk Agent : {exc}")
        else:
            st.dataframe(pd.DataFrame([{"Règle": k, "Limite": f"{v:.2%}"} for k, v in limits.items()]),
                         width="stretch", hide_index=True)
        st.markdown("**Kill Switch persistant** : une fois `HALTED` au seuil de drawdown, "
                    "les BUY sont bloqués. Les SELL restent soumis aux contrôles des avoirs disponibles ; "
                    "il n'y a pas de réinitialisation automatique dans le code fourni.")
        risk_df = table[(table["event_type"] != "GENESIS") & table["risk_status"].notna()]
        st.markdown("**Décisions de risque consignées**")
        st.dataframe(risk_df[["index", "timestamp_utc", "strategy_action", "risk_status",
                              "risk_reason", "kill_switch_state_recorded", "execution_status"]].tail(150).iloc[::-1],
                     width="stretch", hide_index=True)
        st.warning("Les limites ci-dessus proviennent du code courant. Sans snapshots historiques complets, "
                   "on ne peut pas reconstruire ou confirmer a posteriori chaque décision du Risk Agent. "
                   "Le statut du Kill Switch n'est affiché que s'il est explicitement enregistré.")

    with tabs[3]:
        st.subheader("Chronologie du ledger")
        audit = chronology_audit(blocks)
        st.dataframe(pd.DataFrame([{"Contrôle": k, "Nombre": v} for k, v in audit.items()]),
                     width="stretch", hide_index=True)
        st.dataframe(table[["index", "timestamp_utc", "event_type", "strategy_action",
                            "risk_status", "execution_status"]].tail(250).iloc[::-1], width="stretch", hide_index=True)
        st.caption("La chronologie est contrôlée séparément de l'intégrité SHA-256 ; un hash valide ne prouve pas un horodatage correct.")

    with tabs[4]:
        st.subheader("Vérification de la chaîne")
        st.success("Intégrité de la chaîne validée lors du chargement de l'application.")
        st.write("**Fichier analysé :**", LEDGER_RELATIVE_PATH.as_posix())
        st.write("**Empreinte du dernier bloc :**")
        st.code(blocks[-1]["hash"])
        st.write("**Bloc GENESIS :**")
        st.json(blocks[0], expanded=False)
        st.caption("Local, tamper-evident, non décentralisé. Les événements ne sont pas des preuves de performance future.")

    with tabs[5]:
        st.subheader("Exporter les preuves historiques sans écrire sur le disque")
        st.download_button("📥 Tableau des événements vérifiés (CSV)",
                           data=table.to_csv(index=False).encode("utf-8"),
                           file_name="aurum_decision_room_verified_events.csv", mime="text/csv",
                           key="decision_room_csv")
        st.download_button("📥 Bloc sélectionné (JSON)", data=export_selected_event(block),
                           file_name=f"aurum_decision_room_block_{block['index']}.json", mime="application/json",
                           key="decision_room_event_json")
        st.caption("Exports dérivés d'un ledger vérifié ; aucune modification du fichier historique.")


main()
