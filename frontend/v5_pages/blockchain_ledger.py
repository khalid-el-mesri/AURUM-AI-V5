"""AURUM-AI V5 Phase 13 — Blockchain Ledger (verified local audit, read only).

Existing historical ledger and the core blockchain/ledger.py are never modified.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.services.blockchain_ledger_audit import (
    LEDGER_RELATIVE_PATH, check_one_block, chronology_report, event_rows,
    export_block_json, export_summary_json, filter_rows, read_verified_ledger,
    source_metadata, summarize,
)

LEDGER_FILE = ROOT / LEDGER_RELATIVE_PATH


def _hero() -> None:
    """Render the original cryptographic ledger banner as HTML, not a code block."""
    from textwrap import dedent

    html = dedent("""\
        <style>
        .aurum-chain-hero {
            background: linear-gradient(115deg, #0a344a, #0a1727 60%, #06121f);
            border: 1px solid #23739b;
            border-radius: 16px;
            padding: 22px 26px;
            margin: 10px 0 14px;
            color: #edf6ff;
        }
        .aurum-chain-hero .eyebrow {
            color: #35ccff;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 2.7px;
        }
        .aurum-chain-hero h1 {
            font-size: 33px;
            margin: 9px 0 5px;
            line-height: 1.2;
            color: #fff;
        }
        .aurum-chain-hero p {
            margin: 0;
            color: #aed0ea;
            font-size: 14px;
        }
        </style>
        <div class="aurum-chain-hero">
            <div class="eyebrow">◇ AURUM-AI V5 · CRYPTOGRAPHIC AUDIT TRAIL</div>
            <h1>🔗 Blockchain Ledger</h1>
            <p>SHA-256 · Genesis &amp; Hash Links · Recorded Events · Historical Integrity</p>
        </div>
    """).strip()
    st.markdown(html, unsafe_allow_html=True)


def _hash_card(title: str, text: str):
    st.markdown(f"**{title}**")
    st.code(text, language=None)


def main() -> None:
    _hero()
    st.info("**Simulation académique · Lecture seule.** Journal d'audit local et non blockchain publique. "
            "Aucun agent, trade, modèle ou moteur d'exécution n'est lancé par cette page.")

    if not LEDGER_FILE.is_file():
        st.error(f"Fichier historique introuvable : `{LEDGER_RELATIVE_PATH.as_posix()}`. "
                 "Aucune chaîne fictive n'est générée.")
        return

    try:
        # Every fresh Streamlit run independently verifies the source's entire chain.
        snapshot = read_verified_ledger(LEDGER_FILE)
        summary = summarize(snapshot.blocks)
        chronology = chronology_report(snapshot.blocks)
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, RecursionError) as exc:
        st.error("**VÉRIFICATION ÉCHOUÉE — lecture des événements bloquée.** "
                 "Le journal n'a pas passé le contrôle strict du moteur BlockchainLedger.")
        st.code(str(exc))
        st.caption("Ne pas réparer ou recalculer automatiquement les hash. Préserver l'original pour diagnostic.")
        return

    st.success("**Chaîne vérifiée par `BlockchainLedger.load_json()` :** bloc GENESIS, indices, "
               "previous_hash et SHA-256 de chaque bloc conformes au format du projet.")

    cols = st.columns(4)
    cols[0].metric("Blocs vérifiés", f'{summary["total_blocks"]:,}'.replace(",", " "))
    cols[1].metric("Événements enregistrés", f'{summary["events"]:,}'.replace(",", " "))
    cols[2].metric("Risk APPROVED consignés", f'{summary["risk_approvals"]:,}'.replace(",", " "))
    cols[3].metric("Execution EXECUTED consignés", f'{summary["recorded_executions"]:,}'.replace(",", " "))

    st.caption(f"**Fichier :** `{LEDGER_RELATIVE_PATH.as_posix()}` · "
               f"**Taille lue :** {snapshot.size_bytes / 1024:.1f} Kio · "
               "**Source :** Consensus + Trend 60m / Validation. Les événements ne sont pas les "
               "prédictions LSTM/GRU/CNN1D actuellement sélectionnées.")
    tabs = st.tabs(["📊 Overview", "🧱 Block Inspector", "🛡️ Chain Integrity",
                    "🕓 Event History", "🧬 Provenance", "📥 Audit Export"])

    with tabs[0]:
        st.subheader("Distribution des événements historiques")
        if summary["event_types"]:
            ev = pd.DataFrame({"Type": list(summary["event_types"]),
                               "Nombre": list(summary["event_types"].values())}).set_index("Type")
            st.bar_chart(ev, color="#27c2f3")
        else:
            st.info("Aucun événement au-delà du GENESIS.")
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Actions stratégiques consignées**")
            st.dataframe(pd.DataFrame([{"Action": k, "Nombre": v} for k, v in summary["actions"].items()]),
                         width="stretch", hide_index=True)
        with cols[1]:
            st.markdown("**Bornes enregistrées**")
            st.write("Premier événement :", summary["first_event_utc"] or "Aucun")
            st.write("Dernier événement :", summary["last_event_utc"] or "Aucun")
            st.caption("L'ordre des blocs ne certifie ni l'exactitude des heures de marché ni l'exécution réelle.")
        st.markdown("**Derniers blocs enregistrés**")
        st.dataframe(pd.DataFrame(event_rows(snapshot.blocks)).tail(12).iloc[::-1],
                     width="stretch", hide_index=True)

    with tabs[1]:
        st.subheader("Inspecteur de blocs — chaîne vérifiée")
        indices = list(range(len(snapshot.blocks)))
        chosen = st.selectbox("Bloc à inspecter", indices, index=len(indices) - 1,
                              format_func=lambda i: f"#{i} · {snapshot.blocks[i]['event_type']} · {snapshot.blocks[i]['timestamp_utc']}",
                              key="chain_block_select")
        block = snapshot.blocks[chosen]
        checked = check_one_block(snapshot.blocks, chosen)
        a, b, c = st.columns(3)
        a.metric("Index", block["index"])
        b.metric("Événement", block["event_type"])
        c.metric("Contrôle local", "PASS" if checked["ok"] else "FAIL")
        st.markdown("**SHA-256 stocké et recalculé**")
        _hash_card("Empreinte stockée", checked["hash_stocke"])
        _hash_card("Empreinte recalculée", checked["hash_recalcule"])
        st.markdown("**Lien avec le bloc précédent**")
        _hash_card("previous_hash stocké", checked["previous_hash"])
        if chosen > 0:
            st.caption(f"Référence directe au hash du bloc #{chosen - 1} : "
                       f"{'conforme' if checked['checks']['previous_hash_conforme'] else 'non conforme'}.")
        else:
            st.caption("Le bloc GENESIS référence 64 caractères « 0 » selon la convention du projet.")
        if checked["following_index"] is not None:
            st.caption(f"Lien du bloc suivant #{checked['following_index']} vers ce bloc : "
                       f"{'conforme' if checked['checks']['lien_bloc_suivant_conforme'] else 'non conforme'}.")
        st.dataframe(pd.DataFrame([{"Contrôle": k, "Conforme": v} for k, v in checked["checks"].items()]),
                     width="stretch", hide_index=True)
        with st.expander("Afficher la charge utile et les métadonnées originales", expanded=False):
            st.json(block, expanded=True)
        st.caption("Aucun payload synthétique n'est injecté et aucun bloc n'est recalculé sur disque.")

    with tabs[2]:
        st.subheader("Intégrité cryptographique et qualité temporelle")
        summary_table = [
            {"Contrôle": "Schéma JSON du ledger", "État": "VALIDÉ"},
            {"Contrôle": "Bloc GENESIS canonique", "État": "VALIDÉ"},
            {"Contrôle": "Index séquentiels et liens previous_hash", "État": "VALIDÉ"},
            {"Contrôle": "Empreinte SHA-256 de tous les blocs", "État": "VALIDÉ"},
        ]
        st.dataframe(pd.DataFrame(summary_table), width="stretch", hide_index=True)
        st.markdown("**Audit complémentaire des horodatages (non cryptographique)**")
        checks = st.columns(3)
        checks[0].metric("Horodatages invalides", chronology["timestamps_invalides"])
        checks[1].metric("Heures répétées", chronology["timestamps_répétés"])
        checks[2].metric("Inversions chronologiques", chronology["inversions_chronologiques"])
        if any(chronology.values()):
            st.warning("Des anomalies chronologiques sont enregistrées. La chaîne peut rester "
                       "cryptographiquement valide malgré ces anomalies.")
        else:
            st.success("Aucune anomalie chronologique détectée avec ces critères.")
        _hash_card("Hash du dernier bloc (tête de chaîne)", summary["head_hash"])
        st.warning("Une chaîne SHA-256 **n'est pas une preuve indépendante d'authenticité** : "
                   "un auteur disposant du fichier peut réécrire les événements et recalculer tous les hash. "
                   "Sans empreinte ancrée hors du projet ou signature vérifiable, cela reste un journal local tamper-evident.")

    with tabs[3]:
        st.subheader("Consulter les événements d'origine")
        data = event_rows(snapshot.blocks)
        if not data:
            st.info("GENESIS uniquement : aucun autre événement enregistré.")
        else:
            left, mid, right = st.columns(3)
            types = sorted({str(r["Événement"]) for r in data})
            actions = sorted({str(r["Action"]) for r in data if r["Action"] is not None})
            risks = sorted({str(r["Risk status"]) for r in data if r["Risk status"] is not None})
            with left:
                et = st.selectbox("Type d'événement", ["Tous"] + types, key="blockchain_type")
            with mid:
                act = st.selectbox("Action", ["Toutes"] + actions, key="blockchain_action")
            with right:
                risk = st.selectbox("Risk status", ["Tous"] + risks, key="blockchain_risk")
            phrase = st.text_input("Recherche (hash, timestamp, statut, index…)", value="",
                                   key="blockchain_search")
            matches = filter_rows(data, None if et == "Tous" else et,
                                  None if act == "Toutes" else act,
                                  None if risk == "Tous" else risk, phrase)
            st.caption(f"**{len(matches)} / {len(data)} événements** affichés · aucun champ manquant n'est inventé.")
            st.dataframe(pd.DataFrame(matches).iloc[::-1], width="stretch", hide_index=True, height=440)

    with tabs[4]:
        st.subheader("Provenance et empreintes")
        provenance = source_metadata(snapshot)
        st.markdown("**SHA-256 du fichier JSON tel que lu**")
        st.code(snapshot.file_sha256, language=None)
        st.markdown("**Empreintes de chaîne**")
        _hash_card("GENESIS", summary["first_hash"])
        _hash_card("Dernier bloc", summary["head_hash"])
        st.json({k:v for k,v in provenance.items() if k != "limitations"}, expanded=False)
        st.warning("Le hash de fichier correspond **à cette lecture** ; il n'a pas été comparé à une "
                   "empreinte externe antérieure. Les fichiers RAW et les rapports de backtesting "
                   "ne sont pas authentifiés par cette page.")
        st.caption("La chaîne n'est ni distribuée, ni immuable au sens d'une blockchain publique. "
                   "Les hash ne garantissent pas que les événements correspondent à la réalité du marché.")

    with tabs[5]:
        st.subheader("Exports d'audit en lecture seule")
        st.download_button("📥 Rapport d'intégrité (JSON)", export_summary_json(snapshot),
                           file_name="aurum_blockchain_integrity_report.json", mime="application/json",
                           key="blockchain_report")
        rows = event_rows(snapshot.blocks, include_genesis=True)
        st.download_button("📥 Index de tous les blocs vérifiés (CSV)",
                           pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig"),
                           file_name="aurum_blockchain_verified_index.csv", mime="text/csv",
                           key="blockchain_csv")
        st.download_button("📥 Bloc sélectionné (JSON)", export_block_json(snapshot, chosen),
                           file_name=f"aurum_block_{chosen:04d}.json", mime="application/json",
                           key="blockchain_selected_block")
        st.info("Les exports décrivent les données déjà enregistrées. Rien n'est écrit sur le disque du projet "
                "et les hash historiques sont conservés.")


main()
