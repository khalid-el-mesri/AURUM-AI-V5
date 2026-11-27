"""AURUM-AI V5 — Research Center / Experiments (Phase 17), read-only."""
from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import streamlit as st

from frontend.v5_pages._common import demand_session
from frontend.services.research_experiments import (
    HORIZONS, MODELS, EXPERIMENTS_DIR, Experiment, artifact_path,
    compare_report_metadata, list_experiments, manifest_json_bytes,
    read_report, registry_csv_bytes, scalar_metadata, sha256,
)

demand_session()
ROOT = Path(__file__).resolve().parents[2]

st.markdown("""
<style>
.aurum-experiments-hero {
  border:1px solid #20749a;border-radius:15px;
  background:linear-gradient(113deg,#083149 0%,#081724 100%);
  padding:1.3rem 1.55rem;margin-bottom:.7rem;
}
.aurum-experiments-hero .eyebrow {color:#4de1ff;font-size:.7rem;letter-spacing:.15em;font-weight:700}
.aurum-experiments-hero h2 {margin:.45rem 0;color:#f0f9ff}
.aurum-experiments-hero p {margin:0;color:#adcfde}
</style>
<div class="aurum-experiments-hero">
  <div class="eyebrow">AURUM-AI V5 · RESEARCH CENTER</div>
  <h2>🧪 Experiments</h2>
  <p>Registre des expériences historiques · NYSE V3 · LSTM / GRU / CNN1D</p>
</div>
""", unsafe_allow_html=True)

st.info("**Recherche en lecture seule.** 12 combinaisons attendues sur le jeu **Validation**. "
        "Aucun entraînement, aucune inférence nouvelle, aucun accès aux données Test, "
        "aucun ordre, aucune modification des artefacts.")

@st.cache_data(show_spinner=False, ttl=120)
def _load_registry(root: str):
    return list_experiments(Path(root))

try:
    runs = _load_registry(str(ROOT))
except (OSError, ValueError) as exc:
    st.error(f"Inventaire des expériences indisponible : {exc}")
    st.stop()

full = sum(x.status == "Complet" for x in runs)
reports = sum(x.report_present for x in runs)
predictions = sum(x.validation_present for x in runs)
a,b,c,d = st.columns(4)
a.metric("Expériences attendues", len(runs))
b.metric("Runs complets", f"{full}/{len(runs)}")
c.metric("Rapports JSON", f"{reports}/{len(runs)}")
d.metric("Prédictions Validation", f"{predictions}/{len(runs)}")

if not (ROOT / EXPERIMENTS_DIR).is_dir():
    st.error(f"Dossier introuvable : {EXPERIMENTS_DIR.as_posix()}")
    st.stop()

if st.button("↻ Actualiser le registre"):
    _load_registry.clear()
    st.rerun()

f1,f2 = st.columns(2)
filter_horizon = f1.selectbox("Horizon (minutes)", ["Tous"] + list(HORIZONS), key="aurum_experiment_filter_h")
filter_model = f2.selectbox("Modèle", ["Tous"] + list(MODELS), key="aurum_experiment_filter_model")
shown = [r for r in runs if (filter_horizon == "Tous" or r.horizon_minutes == filter_horizon)
         and (filter_model == "Tous" or r.model == filter_model)]

def _table_record(r: Experiment) -> dict:
    return {
        "Expérience": r.run_id, "Modèle": r.model, "Horizon (min)": r.horizon_minutes,
        "État": r.status, "Validation NPZ": "Oui" if r.validation_present else "Non",
        "Rapport JSON": "Oui" if r.report_present else "Non",
        "Checkpoints (inventaire)": r.checkpoint_count,
        "Taille NPZ (Ko)": round(r.validation_size_bytes / 1024, 1) if r.validation_size_bytes is not None else None,
        "Dernière modification (UTC)": r.last_modified_utc,
    }

view = st.tabs(["🧾 Registre", "🔬 Inspecteur", "⚖️ Comparaison", "🔏 Provenance", "⬇ Export"])
with view[0]:
    st.subheader("Registre des expériences existantes")
    st.caption("Les entrées manquantes restent visibles ; elles ne sont ni créées ni réparées automatiquement.")
    st.dataframe(pd.DataFrame([_table_record(r) for r in shown]), hide_index=True, width="stretch", height=410)
    st.markdown("**Disponibilité Validation par horizon**")
    grid = pd.DataFrame(index=list(MODELS), columns=[f"{h}m" for h in HORIZONS])
    for r in runs:
        grid.loc[r.model, f"{r.horizon_minutes}m"] = r.status
    st.dataframe(grid, width="stretch")
    if full < len(runs):
        st.warning(f"{len(runs) - full} combinaison(s) incomplète(s) ou absente(s) : vérifier les fichiers sources, sans générer de remplacement.")

with view[1]:
    st.subheader("Inspecter un run historique")
    if not shown:
        st.warning("Aucune expérience avec ces filtres.")
    else:
        selected_id = st.selectbox("Expérience", [r.run_id for r in shown], key="aurum_experiment_select")
        run = next(r for r in shown if r.run_id == selected_id)
        st.write(f"**Modèle :** {run.model} · **Horizon :** {run.horizon_minutes} min · **État :** {run.status}")
        st.caption(f"artifacts/experiments/nyse_v3/{run.run_id}/")
        if run.report_present:
            try:
                metadata = read_report(ROOT, run.model, run.horizon_minutes)
                with st.expander("Métadonnées JSON originales (clés sensibles masquées)", expanded=True):
                    st.json(metadata)
                scalar = scalar_metadata(metadata)
                st.caption(f"{len(scalar)} champ(s) scalaire(s) détecté(s) ; structure du rapport conservée.")
                if scalar:
                    st.dataframe(pd.DataFrame([{"Champ": str(k), "Valeur": str(v)} for k, v in scalar.items()]),
                                 hide_index=True, width="stretch", height=260)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                st.error(f"Rapport illisible (sans écriture) : {exc}")
        else:
            st.warning("Pas de report.json disponible pour ce run.")
        st.caption("Les métriques du rapport ne sont pas recalculées dans cette page. "
                   "Pour les analyses d'erreurs MAE/RMSE, utiliser AI Model Lab.")

with view[2]:
    st.subheader("Comparer les métadonnées de deux runs")
    st.caption("Comparaison descriptive des champs JSON ; pas de désignation automatique du meilleur modèle.")
    horizon = st.selectbox("Même horizon (minutes)", list(HORIZONS), index=3,
                            key="aurum_experiment_compare_h")
    candidates = [r for r in runs if r.horizon_minutes == horizon and r.report_present]
    if len(candidates) < 2:
        st.warning("Deux rapports JSON disponibles au même horizon sont nécessaires.")
    else:
        left_col,right_col = st.columns(2)
        left = left_col.selectbox("Run A", [r.model for r in candidates], index=0,
                                   key="aurum_experiment_a")
        alternatives = [r.model for r in candidates if r.model != left]
        right = right_col.selectbox("Run B", alternatives, key="aurum_experiment_b")
        try:
            fields = compare_report_metadata(read_report(ROOT,left,horizon),
                                             read_report(ROOT,right,horizon))
            st.dataframe(pd.DataFrame(fields), hide_index=True, width="stretch", height=385)
            st.caption("Un champ différent indique une différence documentaire, pas un gain statistique. "
                       "Les schémas hétérogènes peuvent limiter la comparaison.")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            st.warning(f"Comparaison non disponible : {exc}")

with view[3]:
    st.subheader("Provenance locale des fichiers")
    st.caption("Les noms de checkpoints sont recensés uniquement. Aucun checkpoint n'est chargé ou exécuté.")
    st.dataframe(pd.DataFrame([_table_record(r) for r in shown]), hide_index=True,
                 width="stretch", height=290)
    if shown:
        chosen_id = st.selectbox("Run à contrôler", [r.run_id for r in shown],
                                 key="aurum_experiment_audit_run")
        current = next(r for r in shown if r.run_id == chosen_id)
        if st.button("Calculer les empreintes SHA-256 de ce run"):
            try:
                for kind, present in (("report",current.report_present), ("validation",current.validation_present)):
                    if present:
                        path = artifact_path(ROOT, current.model, current.horizon_minutes, kind)
                        st.write(f"**{path.name}**")
                        st.code(sha256(path), language="text")
            except (OSError, ValueError, FileNotFoundError) as exc:
                st.error(f"Empreinte impossible : {exc}")
    st.warning("Les empreintes locales prouvent la cohérence des octets lus, "
               "pas l'authenticité externe, ni l'absence de data leakage.")

with view[4]:
    st.subheader("Exporter le registre sans modifier les sources")
    st.download_button("⬇ Registre des expériences (CSV)", data=registry_csv_bytes(shown),
                       file_name="aurum_experiments_validation_registry.csv",
                       mime="text/csv", key="aurum_experiments_export_csv")
    st.download_button("⬇ Manifeste de recherche (JSON)", data=manifest_json_bytes(ROOT, shown),
                       file_name="aurum_experiments_validation_manifest.json",
                       mime="application/json", key="aurum_experiments_export_json")
    st.caption("Exports générés en mémoire. Pas de copie des fichiers NPZ ou des poids de modèles. "
               "SHA-256 à la demande depuis l'onglet Provenance.")

st.caption("Portée : expériences Validation NYSE V3 disponibles sur disque. "
           "L'absence d'anomalies de fichiers ne suffit pas à garantir la généralisation Test, "
           "la rentabilité d'une stratégie ou l'absence de fuite temporelle.")
