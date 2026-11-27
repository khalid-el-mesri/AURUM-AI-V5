"""AURUM-AI V5 — autonomous, read-only Blockchain Explorer."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from frontend.v5_pages._common import demand_session
from frontend.services.blockchain_explorer_audit import (
    PROJECT_ROOT, LEDGER_REL, AUDIT_REL,
    block_rows, event_counts, export_rows_json, filter_block_rows,
    load_snapshot, previous_link_status,
)


demand_session()

st.markdown("""
<style>
.explorer-hero {
  padding: 1.3rem 1.5rem; border-radius: 15px;
  background: linear-gradient(120deg, #071b30 0%, #063148 70%, #0d2238 100%);
  border: 1px solid #1271a0; margin-bottom: .8rem;
}
.explorer-hero .eyebrow { color:#33d4ff; font-size:.72rem; letter-spacing:.19rem; font-weight:750; }
.explorer-hero h1 { margin:.25rem 0; color:#ecf8ff; font-size:2rem; }
.explorer-hero p { color:#a0c5d8; margin:0; }
.explorer-note { color:#9fc3da; font-size:.86rem; }
</style>
<div class="explorer-hero">
  <div class="eyebrow">AURUM-AI V5 · CRYPTOGRAPHIC AUDIT TRAIL</div>
  <h1>🔗 Blockchain Explorer</h1>
  <p>Historical decisions · Linked hashes SHA-256 · Local, simulated and read-only</p>
</div>
""", unsafe_allow_html=True)

st.info(
    "Lecture seule des décisions historiques. Aucun bloc n'est ajouté, aucun ordre n'est envoyé "
    "et aucune donnée du projet n'est modifiée. Les vérifications sont locales."
)

try:
    snapshot = load_snapshot(PROJECT_ROOT)
except (OSError, ValueError, KeyError, TypeError) as exc:
    st.error(f"Impossible de charger le Blockchain Explorer : {exc}")
    st.caption(f"Chemin attendu : {LEDGER_REL}")
    st.stop()

rows = list(snapshot.rows)
blocks = snapshot.blocks
source_ok = sum(check["status"] == "CONFORME" for check in snapshot.source_checks)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Blocs (GENESIS compris)", len(blocks))
c2.metric("Événements historiques", max(0, len(blocks) - 1))
c3.metric("Intégrité SHA-256", "PASS" if snapshot.chain_valid else "ÉCHEC")
c4.metric("Sources vérifiées", f"{source_ok}/{len(snapshot.source_checks)}")

if not snapshot.chain_valid:
    st.error("La vérification cryptographique du ledger a échoué. Ne pas considérer les données comme intègres.")
elif snapshot.final_hash_matches_audit is False or source_ok < len(snapshot.source_checks):
    st.warning("Chaîne valide localement, mais une différence ou une absence existe dans les preuves d'audit associées.")
else:
    st.success("Chaîne valide selon BlockchainLedger.verify_integrity(). Vérifier aussi les limites de confiance ci-dessous.")

st.caption(f"Source : {LEDGER_REL}")

overview, search, inspector, integrity, exports = st.tabs([
    "📊 Overview", "🔎 Recherche avancée", "🧱 Inspecteur de blocs",
    "🛡️ Vérification", "📥 Export & provenance",
])

with overview:
    st.subheader("Types d'événements historiques")
    counts = event_counts(rows)
    if counts:
        counts_df = pd.DataFrame([{"Événement": name, "Nombre": number} for name, number in counts.items()])
        st.bar_chart(counts_df.set_index("Événement"), height=280)
    else:
        st.info("Aucun événement historique à afficher.")

    st.subheader("Derniers blocs enregistrés")
    latest = list(reversed(rows[-15:]))
    st.dataframe(pd.DataFrame([{
        "Bloc": item["index"], "UTC": item["time_utc"], "Événement": item["event"],
        "Action": item["action"], "Risque": item["risk"],
        "Exécution": item["execution"], "Hash": item["hash"],
    } for item in latest]), hide_index=True, width="stretch")
    st.caption("Les événements sont déjà consignés dans le ledger. Aucun événement en direct n'est généré.")

with search:
    st.subheader("Rechercher dans les blocs et leurs données")
    term = st.text_input("Index, hash, texte d'événement, transaction ou payload", key="aurum_explorer_query")
    f1, f2, f3, f4 = st.columns(4)
    def choices(key: str):
        return ["Tous"] + sorted({str(row[key]) for row in rows})
    event_choice = f1.selectbox("Événement", choices("event"), key="aurum_explorer_event")
    action_choice = f2.selectbox("Action", choices("action"), key="aurum_explorer_action")
    risk_choice = f3.selectbox("Risque", choices("risk"), key="aurum_explorer_risk")
    execution_choice = f4.selectbox("Exécution", choices("execution"), key="aurum_explorer_execution")
    include_genesis = st.checkbox("Inclure GENESIS", value=True, key="aurum_explorer_genesis")
    filtered = filter_block_rows(
        rows, query=term,
        event=None if event_choice == "Tous" else event_choice,
        action=None if action_choice == "Tous" else action_choice,
        risk=None if risk_choice == "Tous" else risk_choice,
        execution=None if execution_choice == "Tous" else execution_choice,
        include_genesis=include_genesis,
    )
    st.caption(f"{len(filtered)} bloc(s) sur {len(blocks)}")
    export_columns = ("index", "time_utc", "event", "action", "risk", "execution", "kill_switch", "hash", "previous_hash")
    display_df = pd.DataFrame([{name: row[name] for name in export_columns} for row in filtered])
    st.dataframe(display_df, hide_index=True, width="stretch", height=410)
    e1, e2 = st.columns(2)
    e1.download_button(
        "⬇️ Exporter la sélection CSV", display_df.to_csv(index=False).encode("utf-8-sig"),
        "aurum_blockchain_explorer_selection.csv", "text/csv", key="aurum_explorer_csv",
    )
    e2.download_button(
        "⬇️ Exporter les blocs sélectionnés JSON", export_rows_json(filtered, blocks),
        "aurum_blockchain_explorer_selection.json", "application/json", key="aurum_explorer_json",
    )

with inspector:
    st.subheader("Inspecteur de blocs")
    selected_position = st.selectbox(
        "Choisir un bloc",
        options=list(range(len(blocks) - 1, -1, -1)),
        format_func=lambda idx: f"Bloc {rows[idx]['index']} · {rows[idx]['event']} · {rows[idx]['hash'][:14]}",
        key="aurum_explorer_selected_block",
    )
    selected = rows[selected_position]
    block = blocks[selected_position]
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Index", str(selected["index"]))
    i2.metric("Événement", selected["event"])
    i3.metric("Action", selected["action"])
    link_status = previous_link_status(blocks, selected_position)
    i4.metric("Lien précédent", "OK" if link_status is True else "ÉCHEC" if link_status is False else "N/A")
    if link_status is False:
        st.error("Le previous_hash stocké ne correspond pas au hash du bloc précédent.")
    elif link_status is None:
        st.caption("GENESIS ou lien non vérifiable avec les champs disponibles.")
    st.text("SHA-256 stocké : " + str(selected["hash"]))
    st.text("Hash précédent : " + str(selected["previous_hash"]))
    st.subheader("Enregistrement brut (lecture seule)")
    st.json(block, expanded=2)
    st.download_button(
        "⬇️ Exporter ce bloc JSON", json.dumps(block, indent=2, ensure_ascii=False, default=str),
        f"aurum_block_{selected_position}.json", "application/json", key="aurum_explorer_single_json",
    )

with integrity:
    st.subheader("Intégrité cryptographique et qualité de la provenance")
    checks = [
        {"Contrôle": "Vérification SHA-256 et chaînage (moteur existant)", "Résultat": "PASS" if snapshot.chain_valid else "FAIL"},
        {"Contrôle": "Fichier audit_summary.json présent", "Résultat": "OUI" if snapshot.audit_present else "NON"},
        {"Contrôle": "Dernier hash concordant avec l'audit", "Résultat": "PASS" if snapshot.final_hash_matches_audit is True else "FAIL" if snapshot.final_hash_matches_audit is False else "NON VÉRIFIÉ"},
        {"Contrôle": "Liens previous_hash internes", "Résultat": str(sum(previous_link_status(blocks, pos) is False for pos in range(1, len(blocks)))) + " anomalie(s) détectée(s)"},
    ]
    st.dataframe(pd.DataFrame(checks), hide_index=True, width="stretch")
    st.subheader("Sources et empreintes SHA-256")
    st.dataframe(pd.DataFrame(snapshot.source_checks), hide_index=True, width="stretch")
    st.caption("SHA-256 du fichier ledger.json (octets du fichier) : " + snapshot.ledger_sha256)
    if snapshot.audit_sha256:
        st.caption("SHA-256 du fichier audit_summary.json : " + snapshot.audit_sha256)
    st.warning(
        "Une chaîne SHA-256 locale détecte certaines modifications, mais ne constitue pas "
        "une blockchain publique décentralisée ni une preuve indépendante d'authenticité. "
        "Un attaquant pouvant réécrire le ledger ET ses fichiers d'audit pourrait recalculer les hashes."
    )

with exports:
    st.subheader("Provenance et exports de recherche")
    st.code(str(LEDGER_REL), language="text")
    st.code(str(AUDIT_REL), language="text")
    st.caption("Les téléchargements créent des copies dans le navigateur ; aucun fichier source local n'est réécrit.")
    manifest = {
        "project": "AURUM-AI V5",
        "mode": "read_only_historical_simulation",
        "ledger_file": str(LEDGER_REL),
        "audit_file": str(AUDIT_REL),
        "ledger_sha256": snapshot.ledger_sha256,
        "audit_sha256": snapshot.audit_sha256,
        "chain_valid": snapshot.chain_valid,
        "audit_present": snapshot.audit_present,
        "final_hash_matches_audit": snapshot.final_hash_matches_audit,
        "block_count": len(blocks),
        "event_count": max(0, len(blocks) - 1),
        "source_checks": list(snapshot.source_checks),
    }
    st.download_button(
        "⬇️ Rapport de provenance JSON",
        json.dumps(manifest, ensure_ascii=False, indent=2),
        "aurum_blockchain_explorer_provenance.json", "application/json",
        key="aurum_explorer_provenance",
    )
