"""AURUM-AI V5 / Research Center / Reports (Phase 18) — read only."""
from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import streamlit as st
from frontend.v5_pages._common import demand_session
from frontend.services.research_reports import (
    flatten_scalars, html_report_bytes, inventory_csv_bytes,
    json_report_bytes, pdf_report_bytes, report_snapshot,
    safe_source, sha256, source_inventory,
)

demand_session()
ROOT = Path(__file__).resolve().parents[2]

st.markdown("""
<style>
.aurum-reports-hero{border:1px solid #1a7e9e;border-radius:15px;
background:linear-gradient(110deg,#07354b,#091825);padding:1.35rem 1.55rem;margin-bottom:.7rem}
.aurum-reports-hero .tag{color:#48ddf4;font-size:.73rem;letter-spacing:.14em;font-weight:700}
.aurum-reports-hero h2{color:#f1faff;margin:.4rem 0}
.aurum-reports-hero p{color:#bdd6e6;margin:0}
</style>
<div class="aurum-reports-hero">
<div class="tag">AURUM-AI V5 · RESEARCH CENTER</div>
<h2>📑 Reports</h2>
<p>Rapport scientifique consolidé · Historique XOM · Validation · Traçabilité des sources</p>
</div>""", unsafe_allow_html=True)

st.info("**Consultation en lecture seule.** Le reporting regroupe des artefacts historiques déjà présents. "
        "Il ne produit aucun entraînement, aucune prédiction nouvelle, aucun ordre et n'utilise pas les données Test.")

@st.cache_data(ttl=120, show_spinner=False)
def _snapshot(root: str):
    return report_snapshot(Path(root))

try:
    report = _snapshot(str(ROOT))
except (OSError, ValueError) as exc:
    st.error(f"Centre de rapports indisponible : {exc}")
    st.stop()

sources = report["sources"]
files_present = sum(s["status"] == "present" for s in sources)
total_files = len(sources)
valid_runs = report["model_reports_valid"]
decisions = report.get("historical_decisions") or {}
trades = report.get("historical_transactions") or {}
a,b,c,d = st.columns(4)
a.metric("Rapports IA lisibles", f"{valid_runs}/12")
b.metric("Sources présentes", f"{files_present}/{total_files}")
c.metric("Décisions historiques", decisions.get("rows", "N/D"))
d.metric("Transactions historiques", trades.get("rows", "N/D"))

if st.button("↻ Actualiser les rapports"):
    _snapshot.clear()
    st.rerun()

if valid_runs < 12:
    st.warning(f"{12-valid_runs} rapport(s) IA manquant(s) ou illisible(s). Aucune donnée n'est reconstituée.")

views = st.tabs(["📊 Synthèse", "🧠 Modèles IA", "💼 Simulation", "🔗 Intégrité", "📤 Exports"])

with views[0]:
    st.subheader("Rapport consolidé — sources documentaires")
    st.caption("Les états proviennent des fichiers locaux à cet instant. Le catalogue n'inclut aucun dossier privé ni dataset RAW.")
    table = pd.DataFrame([{"Source": s["key"], "Chemin relatif": s["path"],
                           "État": s["status"], "Taille (Ko)":
                           round(s["bytes"]/1024,1) if s["bytes"] is not None else None}
                          for s in sources])
    st.dataframe(table, hide_index=True, width="stretch", height=360)
    st.subheader("Périmètre et limites de la recherche")
    for limit in report["limitations"]:
        st.markdown("- " + limit)
    st.caption(f"Instantané créé à {report['generated_utc']}. "
               "Une vérification de contenu scientifique et une évaluation Test indépendantes restent nécessaires.")

with views[1]:
    st.subheader("Inventaire des entraînements historiques")
    st.dataframe(pd.DataFrame([{"Modèle": r["model"], "Horizon (min)":r["horizon_minutes"],
                               "Statut du fichier":r["file_status"],
                               "Rapport JSON lisible":r["json_valid"]}
                              for r in report["model_runs"]]), hide_index=True, width="stretch")
    st.caption("Le statut 'lisible' ne garantit ni absence de leakage ni qualité prédictive.")
    st.markdown("**Comparaison des métriques de Validation disponibles**")
    try:
        from frontend.services.model_comparison import load_model_comparison
        comparison = load_model_comparison()
        st.dataframe(comparison, hide_index=True, width="stretch", height=320)
        st.caption("Métriques reprises du service AI Model Lab. Pas de classement sur Test indépendant.")
    except (ImportError, OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        st.warning(f"La comparaison de métriques n'est pas disponible : {exc}")
    st.caption("Pour l'analyse détaillée des erreurs, utiliser la page AI Model Lab.")

with views[2]:
    st.subheader("Backtesting historique — 60 minutes")
    st.info("Les décisions et transactions sont issues de la simulation existante. "
            "Aucune nouvelle stratégie n'est optimisée ou exécutée.")
    summary = report.get("historical_summary")
    if summary is None:
        st.warning("comparison_summary.json absent ou illisible : " + str(report.get("historical_summary_error")))
    else:
        st.markdown("**Champs déclarés dans le rapport historique**")
        st.dataframe(pd.DataFrame(flatten_scalars(summary, limit=100)), hide_index=True,
                     width="stretch", height=340)
        with st.expander("Voir les métadonnées JSON du résumé (clés sensibles masquées)"):
            st.json(summary)
    col1,col2 = st.columns(2)
    for col, label, key in [(col1,"Décisions", "historical_decisions"),
                            (col2,"Transactions", "historical_transactions")]:
        with col:
            data = report.get(key)
            st.markdown(f"**{label} historiques**")
            if data is None:
                st.warning("CSV indisponible : " + str(report.get(key + "_error")))
            else:
                st.metric("Lignes source", data["rows"])
                st.dataframe(pd.DataFrame(data["preview"]), hide_index=True,
                             width="stretch", height=280)
    st.warning("Une simulation historique, même réconciliée, ne démontre pas une rentabilité future. "
               "Les résultats ne constituent pas des décisions de trading.")

with views[3]:
    st.subheader("Blockchain locale et provenance")
    if report["ledger_present"]:
        st.success("Le fichier ledger.json historique est présent.")
    else:
        st.warning("ledger.json absent : la chaîne ne peut pas être contrôlée.")
    audit = report.get("blockchain_audit")
    if audit is not None:
        st.markdown("**Audit enregistré dans le projet (déclarations de la source)**")
        st.dataframe(pd.DataFrame(flatten_scalars(audit, limit=40)), hide_index=True,
                     width="stretch", height=240)
    else:
        st.warning("Audit historique absent ou illisible : " + str(report.get("blockchain_audit_error")))
    if st.button("Vérifier la chaîne locale existante", disabled=not report["ledger_present"]):
        try:
            from blockchain.ledger import BlockchainLedger
            path = safe_source(ROOT, "ledger")
            if path.stat().st_size > 12 * 1024 * 1024:
                raise ValueError("Ledger trop volumineux pour une vérification interactive")
            chain = BlockchainLedger.load_json(path)
            valid = bool(chain.verify_integrity())
            if valid:
                st.success(f"Chaîne locale : intégrité SHA-256 PASS ({len(chain.blocks)} blocs lus).")
            else:
                st.error("La chaîne locale ne passe pas le contrôle SHA-256.")
        except (ImportError, OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            st.error(f"La vérification n'a pas pu aboutir : {exc}")
    st.markdown("**Empreintes des sources autorisées — à la demande**")
    if st.button("Calculer SHA-256 des sources présentes"):
        try:
            hashed = source_inventory(ROOT, with_hashes=True)
            st.dataframe(pd.DataFrame(hashed), hide_index=True, width="stretch")
        except (OSError, ValueError) as exc:
            st.error(f"Erreur de hachage : {exc}")
    st.warning("Une empreinte locale valide prouve la cohérence des octets contrôlés, "
               "pas l'authenticité externe des données ou l'absence de fuite temporelle.")

with views[4]:
    st.subheader("Exporter un rapport professionnel — sans écriture sur le disque du projet")
    st.caption("Les exports sont générés en mémoire à partir des fichiers autorisés déjà lus. "
               "Le PDF ne doit pas être interprété comme une certification indépendante.")
    st.download_button("⬇ Rapport scientifique (HTML imprimable)",
                       data=html_report_bytes(report), file_name="AURUM_AI_V5_Research_Report.html",
                       mime="text/html", key="aurum_reports_html")
    try:
        pdf_data = pdf_report_bytes(report)
        st.download_button("⬇ Rapport scientifique (PDF)", data=pdf_data,
                           file_name="AURUM_AI_V5_Research_Report.pdf",
                           mime="application/pdf", key="aurum_reports_pdf")
    except ImportError:
        st.warning("PDF non disponible : installer ReportLab dans le venv (`python -m pip install reportlab`). "
                   "Le rapport HTML reste utilisable avec Imprimer → PDF.")
    except (OSError, ValueError, TypeError) as exc:
        st.warning(f"PDF non généré : {exc}. Utiliser l'export HTML.")
    st.download_button("⬇ Instantané d'audit (JSON)", data=json_report_bytes(report),
                       file_name="AURUM_AI_V5_Research_Report.json", mime="application/json",
                       key="aurum_reports_json")
    st.download_button("⬇ Inventaire des sources (CSV)", data=inventory_csv_bytes(report),
                       file_name="AURUM_AI_V5_Research_Sources.csv", mime="text/csv",
                       key="aurum_reports_csv")

st.caption("AURUM-AI V5 · Historical Validation Only · XOM 5-minute source · "
           "Recherches et simulation uniquement. Aucun conseil financier, aucune exécution réelle.")
