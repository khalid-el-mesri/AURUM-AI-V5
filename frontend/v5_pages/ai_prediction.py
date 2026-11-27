"""AI Prediction — historical validation only, reusable within AURUM-AI V5."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.v5_pages._common import demand_session
from frontend.services.ai_prediction_engine import (
    MODELS, HORIZONS, PredictionArtifactError, available_experiments,
    experiment_file, read_validation, evaluation_metrics, temporal_audit, metric_table,
)

demand_session()
EXPERIMENT_DIR = ROOT / "artifacts" / "experiments" / "nyse_v3"

st.markdown("""
<style>
.aurum-pred-hero {
  background:linear-gradient(115deg,#07283f,#081827);border:1px solid #1c5b77;
  border-radius:18px;padding:28px;margin:12px 0 16px;
}
.aurum-pred-eye {color:#43d5ff;letter-spacing:2.3px;font-size:11px;font-weight:800;}
.aurum-pred-title {color:#f1f8ff;font-weight:900;font-size:36px;margin:11px 0 6px;}
.aurum-pred-sub {color:#b6d2e6;}
</style>
<div class="aurum-pred-hero">
  <div class="aurum-pred-eye">◇ AURUM-AI V5 · XOM QUANTITATIVE INTELLIGENCE</div>
  <div class="aurum-pred-title">🎯 AI Prediction</div>
  <div class="aurum-pred-sub">LSTM · GRU · CNN1D — expériences de validation historique NYSE V3</div>
</div>
""", unsafe_allow_html=True)

st.info("**Historique uniquement :** les prédictions sont chargées depuis les fichiers Validation. "
        "Il n'y a ni cotations en direct, ni inférence instantanée, ni transactions réelles. "
        "Le jeu Test reste inutilisé.")

available = available_experiments(EXPERIMENT_DIR)
if not available:
    st.error("Aucun fichier validation_predictions.npz trouvé dans artifacts/experiments/nyse_v3.")
    st.stop()

model_col, horizon_col = st.columns(2)
with model_col:
    models = [m for m in MODELS if any(a == m for a, h in available)]
    model = st.selectbox("Modèle", models, index=0, key="aurum_pred_model")
with horizon_col:
    horizon_choices = [h for h in HORIZONS if (model, h) in available]
    horizon = st.selectbox("Horizon (minutes)", horizon_choices, index=len(horizon_choices)-1, key="aurum_pred_horizon")

path = experiment_file(EXPERIMENT_DIR, model, horizon)

@st.cache_data(show_spinner=False)
def cached_validation(file_path: str, modified_ns: int, model_id: str, minutes: int) -> pd.DataFrame:
    return read_validation(Path(file_path), model=model_id, horizon=minutes)

try:
    df = cached_validation(str(path), path.stat().st_mtime_ns, model, horizon)
    metrics = evaluation_metrics(df)
except (PredictionArtifactError, ValueError, OSError, FileNotFoundError) as exc:
    st.error(f"Erreur de lecture/validation des artefacts : {exc}")
    st.stop()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Observations Validation", f"{metrics['n']:,}".replace(",", " "))
k2.metric("MAE · log-rendement", f"{metrics['mae']:.8f}")
k3.metric("RMSE · log-rendement", f"{metrics['rmse']:.8f}")
k4.metric("MAE baseline zéro", f"{metrics['zero_mae']:.8f}")
accuracy_col, explanation_col = st.columns([1, 3])

accuracy_col.metric(
    "Directional Accuracy (%) - Validation",
    f"{metrics['direction_accuracy_pct']:.2f}%"
)

explanation_col.caption(
    "Pourcentage de directions correctement pr\u00e9dites. "
    "Validation historique uniquement : aucune garantie "
    "sur Test ni sur la rentabilit\u00e9."
)


st.caption(f"Source : `{path.relative_to(ROOT)}` · "
           "Les valeurs prédites sont des **rendements logarithmiques**, pas des cours en USD. "
           "Le prix futur estimé est une conversion dérivée : close_actuel × exp(y_pred).")

tab_overview, tab_price, tab_metrics, tab_time, tab_export = st.tabs([
    "📈 Réel vs prédit", "💵 Prix reconstruits", "📊 Métriques & baseline",
    "🛡️ Intégrité temporelle", "📥 Export Validation",
])

with tab_overview:
    st.subheader("Rendements logarithmiques historiques — Validation")
    points = st.select_slider("Observations récentes à afficher", options=[100, 250, 500, 1000, 2000], value=250)
    subset = df.tail(points).copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=subset["target_timestamp_utc"], y=subset["observed_log_return_pct"],
                             mode="lines", name="Rendement réalisé (%)", line=dict(color="#24e3f0")))
    fig.add_trace(go.Scatter(x=subset["target_timestamp_utc"], y=subset["predicted_log_return_pct"],
                             mode="lines", name="Rendement prédit (%)", line=dict(color="#ffbf49")))
    fig.update_layout(template="plotly_dark", height=425, xaxis_title="Horodatage cible (UTC)",
                      yaxis_title="Log-rendement (%)", paper_bgcolor="#081525", plot_bgcolor="#081525",
                      legend=dict(orientation="h", y=1.08), hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    st.caption("Les points sont placés à l'horodatage cible, et non présentés comme des signaux en temps réel. "
               "Les éventuelles périodes sans observations ne sont pas imputées.")
    st.dataframe(subset[["signal_available_at_utc", "target_available_at_utc",
                         "observed_log_return_pct", "predicted_log_return_pct",
                         "absolute_error_log_return"]].tail(30).iloc[::-1],
                 width="stretch", hide_index=True)

with tab_price:
    st.subheader("Cours observé et cours dérivé de la prévision")
    subset = df.tail(250).copy()
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=subset["target_timestamp_utc"], y=subset["next_close"],
                                   name="Close réellement observé", mode="lines",
                                   line=dict(color="#22e6a1")))
    fig_price.add_trace(go.Scatter(x=subset["target_timestamp_utc"], y=subset["predicted_close"],
                                   name="Close calculé à partir du log-rendement prédit", mode="lines",
                                   line=dict(color="#ffbf49")))
    fig_price.update_layout(template="plotly_dark", height=420, xaxis_title="Horodatage cible (UTC)",
                            yaxis_title="Prix bid XOM (USD)", paper_bgcolor="#081525",
                            plot_bgcolor="#081525", hovermode="x unified")
    st.plotly_chart(fig_price, width="stretch")
    st.caption("Reconstruction mathématique, pas un nouveau modèle : close prédit = close actuel × exp(log-rendement prédit).")

with tab_metrics:
    st.subheader("Comparaison avec Zero Return")
    imp = metrics["mae_improvement_pct"]
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Amélioration MAE vs zéro", f"{imp:+.3f}%" if imp is not None else "Indisponible")
    col_b.metric("RMSE baseline zéro", f"{metrics['zero_rmse']:.8f}")
    col_c.metric("Exactitude directionnelle brute", f"{metrics['direction_accuracy_pct']:.2f}%")
    st.caption("Exactitude directionnelle descriptive : signe du log-rendement prédit vs réalisé, sans seuil de neutralité. "
               "Aucune rentabilité ni généralisation Test n'en découle.")
    st.subheader("LSTM / GRU / CNN1D pour le même horizon")
    compare = metric_table(EXPERIMENT_DIR, horizon)
    st.dataframe(compare, width="stretch", hide_index=True)
    if imp is not None and imp <= 0:
        st.warning("Sur ces observations Validation, ce modèle ne surpasse pas la baseline Zero Return en MAE.")
    else:
        st.info("Une MAE légèrement meilleure que Zero Return n'implique pas une stratégie rentable.")

with tab_time:
    st.subheader("Contrôle des horodatages et disponibilités")
    audit = temporal_audit(df)
    st.dataframe(pd.DataFrame([{"Contrôle": k, "Anomalies": n} for k, n in audit.items()]),
                 width="stretch", hide_index=True)
    if any(audit.values()):
        st.warning("Des incohérences temporelles nécessitent une investigation avant toute interprétation.")
    else:
        st.success("Les métadonnées temporelles vérifiées sont cohérentes pour ce fichier.")
    st.warning("Ce contrôle ne constitue **pas** une preuve d'absence totale de data leakage. "
               "Les scalers, labels, fenêtres et découpages Train/Validation/Test doivent aussi être audités.")
    st.caption("Un signal ne devient exploitable qu'à signal_available_at_ms. "
               "La cible n'est connue qu'à target_available_at_ms, même lorsque target_timestamp_ms coïncide avec le début de la bougie cible.")

with tab_export:
    st.subheader("Export en lecture seule")
    st.download_button("📥 Télécharger les prédictions Validation (CSV)",
                       data=df[["model", "horizon_minutes", "signal_available_at_utc",
                                "target_available_at_utc", "current_close", "next_close",
                                "y_true", "y_pred", "predicted_close"]].to_csv(index=False).encode("utf-8"),
                       file_name=f"aurum_{model.lower()}_{horizon}m_validation_predictions.csv",
                       mime="text/csv")
    st.caption("Aucun CSV RAW, dataset Parquet, checkpoint ou modèle entraîné n'est modifié.")
