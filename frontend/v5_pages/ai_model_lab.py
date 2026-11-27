"""AURUM-AI V5 — AI Model Lab, Phase 15.

Read-only exploration of the existing 12 NYSE V3 VALIDATION experiments.
Reuses the Phase-5 prediction engine. No Test data, training, inference or orders.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.v5_pages._common import demand_session
from frontend.services.ai_prediction_engine import (
    HORIZONS, MODELS, PredictionArtifactError,
    experiment_file, read_validation,
)
from frontend.services.ai_model_lab_audit import (
    EXPERIMENT_SUBDIR, comparison_for_horizon, file_sha256, inventory,
    read_report, report_path, summary_export, temporal_check, top_residuals,
    validation_diagnostics,
)


demand_session()
ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = ROOT / EXPERIMENT_SUBDIR
CYAN, GOLD, GREEN, BG = "#24d4f5", "#f4c45e", "#12d9aa", "#081525"

st.markdown("""
<style>
.lab-hero {
    background:linear-gradient(118deg,#082c44 0%,#081827 100%);
    border:1px solid #20749a;border-radius:17px;
    padding:1.6rem 1.8rem;margin-bottom:.9rem;
}
.lab-eyebrow {font-size:.73rem;letter-spacing:.22rem;font-weight:800;color:#44d9ff}
.lab-hero h1 {font-size:2.13rem;line-height:1.2;color:#ecf8ff;margin:.55rem 0 .45rem}
.lab-hero p {font-size:.94rem;color:#afd1e3;margin:0}
</style>
<div class="lab-hero">
  <div class="lab-eyebrow">◇ AURUM-AI V5 · VALIDATION INTELLIGENCE</div>
  <h1>🧠 AI Model Lab</h1>
  <p>LSTM · GRU · CNN1D · XOM / NYSE V3 · Historique & recherche quantitative</p>
</div>
""", unsafe_allow_html=True)

st.info(
    "**Lecture seule / Validation uniquement.** Comparaison des prédictions historiques stockées. "
    "Aucun modèle réentraîné, aucune inférence instantanée, aucun accès au jeu Test, "
    "aucun ordre réel ou simulé ajouté. Les labels sont des **log-rendements**, pas des prix USD."
)

if not EXPERIMENT_ROOT.is_dir():
    st.error(f"Dossier d'expériences introuvable : {EXPERIMENT_SUBDIR.as_posix()}")
    st.stop()

catalogue = inventory(EXPERIMENT_ROOT)
available = catalogue.loc[catalogue["NPZ Validation"] == "Présent"]
if available.empty:
    st.error("Aucun artefact Validation n'est disponible pour l'AI Model Lab.")
    st.stop()

n_available = len(available)
n_reports = int((catalogue["Rapport JSON"] == "Présent").sum())
a, b, c = st.columns(3)
a.metric("Expériences Validation disponibles", f"{n_available}/12")
b.metric("Rapports JSON disponibles", f"{n_reports}/12")
c.metric("Familles entraînées présentes", int(available["Modèle"].nunique()))

horizon_values = [h for h in HORIZONS if h in available["Horizon (min)"].tolist()]
horizon_col, model_col = st.columns(2)
with horizon_col:
    horizon = st.selectbox("Horizon de prédiction (minutes)", horizon_values,
                           index=len(horizon_values) - 1, key="aurum_lab_horizon")
models_for_horizon = [m for m in MODELS if not available.loc[
    (available["Modèle"] == m) & (available["Horizon (min)"] == horizon)
].empty]
with model_col:
    model = st.selectbox("Modèle à inspecter", models_for_horizon,
                         key="aurum_lab_model")


def _cache_signature(h: int) -> tuple:
    signature = []
    for name in MODELS:
        filename = experiment_file(EXPERIMENT_ROOT, name, h)
        if filename.is_file():
            stat = filename.stat()
            signature.append((name, stat.st_mtime_ns, stat.st_size))
        else:
            signature.append((name, None, None))
    return tuple(signature)


@st.cache_data(show_spinner=False)
def _comparison_cached(root_text: str, h: int, signature: tuple):
    return comparison_for_horizon(Path(root_text), h)


@st.cache_data(show_spinner=False)
def _read_cached(path_text: str, mtime: int, size: int, name: str, h: int):
    return read_validation(Path(path_text), model=name, horizon=h)


comparison, comparison_errors = _comparison_cached(
    str(EXPERIMENT_ROOT), horizon, _cache_signature(horizon)
)
selected_file = experiment_file(EXPERIMENT_ROOT, model, horizon)
selected_stat = selected_file.stat()
try:
    df = _read_cached(str(selected_file), selected_stat.st_mtime_ns,
                      selected_stat.st_size, model, horizon)
    metrics = validation_diagnostics(df)
    checks = temporal_check(df)
except (PredictionArtifactError, OSError, ValueError, KeyError, TypeError) as exc:
    st.error(f"Le fichier Validation sélectionné ne passe pas les contrôles : {exc}")
    if not comparison_errors.empty:
        st.dataframe(comparison_errors, hide_index=True, width="stretch")
    st.stop()


def _fmt(number: float | None, decimals: int = 6) -> str:
    return "N/D" if number is None else f"{number:.{decimals}f}"


st.caption(f"Source sélectionnée : `{selected_file.relative_to(ROOT).as_posix()}` · "
           f"{len(df):,} observations · log-return = ln(close future / close courant).")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Observations Validation", f"{metrics['observations']:,}")
k2.metric("MAE · log-return", _fmt(metrics["mae"], 8))
k3.metric("RMSE · log-return", _fmt(metrics["rmse"], 8))
k4.metric("Gain MAE vs Zero Return", (
    "N/D" if metrics["mae_improvement_pct"] is None
    else f"{metrics['mae_improvement_pct']:+.2f}%"
))

sections = st.tabs([
    "📊 Comparaison", "📉 Erreurs & résidus", "📈 Observé / prédit",
    "🕒 Audit temporel", "🔎 Provenance", "📥 Export",
])

with sections[0]:
    st.subheader(f"Comparaison sur la Validation — horizon {horizon} min")
    if not comparison_errors.empty:
        st.warning("Certaines expériences sont absentes ou non conformes :")
        st.dataframe(comparison_errors, hide_index=True, width="stretch")
    if comparison.empty:
        st.error("Aucune mesure fiable disponible pour cet horizon.")
    else:
        st.dataframe(
            comparison.sort_values("MAE").style.format({
                "MAE": "{:.8f}", "RMSE": "{:.8f}",
                "Zero Return MAE": "{:.8f}", "Zero Return RMSE": "{:.8f}",
                "Gain MAE (%)": "{:+.2f}", "Accord directionnel (%)": "{:.2f}",
            }, na_rep="N/D"),
            hide_index=True, width="stretch",
        )
        choice = st.radio("Indicateur pour la figure", ["MAE", "RMSE"],
                          horizontal=True, key="aurum_lab_metric")
        baseline = "Zero Return MAE" if choice == "MAE" else "Zero Return RMSE"
        fig = go.Figure()
        fig.add_trace(go.Bar(x=comparison["Modèle"], y=comparison[choice],
                             name="Modèle", marker_color=CYAN))
        fig.add_trace(go.Bar(x=comparison["Modèle"], y=comparison[baseline],
                             name="Zero Return (mêmes observations)", marker_color=GOLD))
        fig.update_layout(template="plotly_dark", barmode="group", height=400,
                          paper_bgcolor=BG, plot_bgcolor=BG,
                          margin=dict(l=12, r=12, b=25, t=20),
                          yaxis_title=f"{choice} · log-return", xaxis_title="Modèle",
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width="stretch")
        st.caption("La référence Zero Return prédit un log-rendement de zéro sur les mêmes "
                   "observations de Validation. Un gain négatif signifie que le modèle fait "
                   "moins bien que cette référence. Le plus faible MAE Validation ne constitue "
                   "pas un gagnant sur le jeu Test indépendant.")

with sections[1]:
    st.subheader("Distribution des erreurs — Validation")
    resid = df["y_pred"] - df["y_true"]
    hist = go.Figure(go.Histogram(x=resid * 100, nbinsx=65,
                                  marker_color=CYAN, name="Résidu"))
    hist.update_layout(template="plotly_dark", height=350,
                       paper_bgcolor=BG, plot_bgcolor=BG,
                       xaxis_title="(Prédit − observé) · points de log-%",
                       yaxis_title="Nombre d'observations",
                       margin=dict(l=10, r=10, t=15, b=10))
    st.plotly_chart(hist, width="stretch")
    e1, e2, e3 = st.columns(3)
    e1.metric("Biais moyen", _fmt(metrics["bias"], 8))
    e2.metric("Erreur absolue médiane", _fmt(metrics["median_absolute_error"], 8))
    e3.metric("95e percentile erreur absolue", _fmt(metrics["p95_absolute_error"], 8))
    st.markdown("**Les 20 plus grandes erreurs absolues (log-return)**")
    st.dataframe(top_residuals(df, 20), hide_index=True, width="stretch")
    st.caption("Examen descriptif sur Validation, pas d'évaluation de stabilité en production.")

with sections[2]:
    st.subheader("Comparaison locale : valeurs observées et prédites")
    n = len(df)
    count = min(250, n)
    max_start = max(n - count, 0)
    start = st.slider("Début de la fenêtre (index observation)", min_value=0,
                      max_value=max_start, value=max_start, step=1,
                      key="aurum_lab_start") if max_start else 0
    subset = df.iloc[start:start + count].copy()
    st.caption(f"Observations {start + 1:,} à {start + len(subset):,} sur {n:,}. "
               "Les horaires du marché ne sont pas continus ; aucun segment entre "
               "deux observations séparées n'est interpolé.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=subset["end_timestamp_utc"],
                             y=subset["observed_log_return_pct"], mode="markers",
                             name="Observé (log-%)", marker=dict(color=CYAN, size=6)))
    fig.add_trace(go.Scatter(x=subset["end_timestamp_utc"],
                             y=subset["predicted_log_return_pct"], mode="markers",
                             name="Prédit (log-%)", marker=dict(color=GOLD, size=6)))
    fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
                      height=400, xaxis_title="Fin de fenêtre UTC",
                      yaxis_title="Log-rendement (%)",
                      margin=dict(l=10, r=10, t=15, b=10))
    st.plotly_chart(fig, width="stretch")
    fig_close = go.Figure()
    fig_close.add_trace(go.Scatter(x=subset["target_timestamp_utc"],
                                   y=subset["next_close"], mode="markers",
                                   name="Clôture future observée", marker=dict(color=GREEN, size=6)))
    fig_close.add_trace(go.Scatter(x=subset["target_timestamp_utc"],
                                   y=subset["predicted_close"], mode="markers",
                                   name="Clôture reconstruite", marker=dict(color=GOLD, size=6)))
    fig_close.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
                            height=390, xaxis_title="Timestamp cible UTC", yaxis_title="USD",
                            margin=dict(l=10, r=10, t=15, b=10))
    st.plotly_chart(fig_close, width="stretch")
    st.caption("Clôture reconstruite = close courant × exp(log-return prédit). "
               "Ce n'est pas une nouvelle inférence et ce n'est pas une cotation actuelle.")

with sections[3]:
    st.subheader("Contrôles de disponibilité et d'horodatage")
    st.dataframe(checks, hide_index=True, width="stretch")
    anomalies = int(checks["Anomalies"].sum())
    if anomalies:
        st.warning(f"{anomalies:,} anomalies structurelles détectées ; examiner les lignes et le pipeline source.")
    else:
        st.success("Aucune anomalie dans les contrôles de métadonnées implémentés.")
    st.warning("**Limite méthodologique :** des timestamps corrects ne démontrent pas "
               "l'absence de data leakage. Le split Train/Validation/Test, "
               "le fitting des scalers sur Train seul, les labels et la création des fenêtres "
               "doivent être audités indépendamment.")
    st.caption("Aucun entraînement ni backtest ne sont exécutés ici.")

with sections[4]:
    st.subheader("Traçabilité des expériences")
    st.dataframe(catalogue, hide_index=True, width="stretch")
    report = report_path(EXPERIMENT_ROOT, model, horizon)
    st.markdown("**Empreinte du fichier de prédictions sélectionné**")
    st.code(file_sha256(selected_file), language=None)
    st.caption(f"{selected_file.relative_to(ROOT).as_posix()}")
    if report.is_file():
        st.markdown("**Rapport JSON original (consultation)**")
        try:
            metadata = read_report(report)
            st.caption(f"SHA-256 rapport : {file_sha256(report)}")
            with st.expander("Afficher les métadonnées originales"):
                st.json(metadata)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            st.warning(f"Le rapport JSON ne peut pas être interprété : {exc}")
    else:
        st.warning("Pas de report.json associé à cette expérience.")
    st.caption("Les empreintes confirment l'état local des fichiers, "
               "pas une certification externe ou un ancrage sur une blockchain publique.")

with sections[5]:
    st.subheader("Exports d'audit — sans modification des artefacts")
    if not comparison.empty:
        st.download_button(
            "⬇ Télécharger la comparaison de l'horizon (CSV)",
            data=comparison.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"aurum_ai_model_lab_comparison_{horizon}m_validation.csv",
            mime="text/csv", key="aurum_lab_compare_export",
        )
    st.download_button(
        "⬇ Télécharger les plus grandes erreurs (CSV)",
        data=top_residuals(df, 100).to_csv(index=False).encode("utf-8-sig"),
        file_name=f"aurum_{model.lower()}_{horizon}m_largest_errors_validation.csv",
        mime="text/csv", key="aurum_lab_error_export",
    )
    st.download_button(
        "⬇ Télécharger le rapport d'audit (JSON)",
        data=summary_export(model, horizon, metrics, checks, selected_file),
        file_name=f"aurum_{model.lower()}_{horizon}m_validation_audit.json",
        mime="application/json", key="aurum_lab_audit_export",
    )
    st.caption("Les exports sont générés en mémoire depuis les fichiers existants. "
               "Aucun fichier source du projet n'est modifié.")
