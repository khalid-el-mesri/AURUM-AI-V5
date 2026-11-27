"""Read-only comparison of the 12 existing Validation-model reports."""
from __future__ import annotations
import streamlit as st
import plotly.express as px
from frontend.v5_pages._common import demand_session
from frontend.services.model_comparison import load_model_comparison

demand_session()
st.markdown("# 🧠 AI Model Lab")
st.caption("AURUM-AI V5 · XOM historical validation · LSTM / GRU / CNN1D")
st.warning("Les métriques affichées proviennent uniquement des artefacts **Validation**. Elles ne démontrent ni généralisation Test ni rentabilité future.")
try:
    comparison = load_model_comparison()
except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
    st.error("Les 12 rapports de validation ne sont pas disponibles ou ne passent pas les contrôles de cohérence.")
    st.caption(f"Détail : {exc}")
    st.stop()

horizons = sorted(comparison["Horizon (min)"].unique().tolist())
selected_horizon = st.selectbox("Horizon de prédiction (minutes)",horizons,index=len(horizons)-1)
shown = comparison.loc[comparison["Horizon (min)"] == selected_horizon].copy()
c1, c2, c3 = st.columns(3)
c1.metric("Modèles comparés",len(shown))
c2.metric("Meilleure MAE",f"{shown['MAE'].min():.8f}")
c3.metric("Meilleure RMSE",f"{shown['RMSE'].min():.8f}")
st.subheader("Résultats Validation")
st.dataframe(shown,hide_index=True,use_container_width=True)
metric = st.radio("Métrique",["MAE","RMSE"],horizontal=True)
fig = px.bar(shown,x="Model",y=metric,title=f"{metric} par modèle · {selected_horizon} min")
fig.update_layout(template="plotly_dark",paper_bgcolor="#081525",plot_bgcolor="#081525")
fig.update_yaxes(rangemode="tozero")
st.plotly_chart(fig,use_container_width=True)
st.caption("La comparaison est descriptive, limitée aux rapports de validation et au baseline Zero Return. Aucun modèle n'est déclaré gagnant sur un Test indépendant.")
