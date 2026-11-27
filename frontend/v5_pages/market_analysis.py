"""AURUM-AI V5 | Historical XOM market analysis (read only)."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from frontend.v5_pages._common import demand_session
from frontend.services.market_analysis_engine import (
    read_raw, subset_utc, aggregate_ohlc, enrich_market, gaps_table, month_coverage,
)

# This page cannot be invoked outside an authenticated V5 session.
demand_session()
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "01_raw" / "xomususd-m5-bid-2017-01-26-2026-10-01.csv"

st.markdown("""
<style>
.ma-hero{background:linear-gradient(115deg,#092e46,#081728);border:1px solid #24618b;
border-radius:16px;padding:28px 30px;margin-bottom:16px}
.ma-kicker{color:#39d5f9;font-size:11px;letter-spacing:2.4px;font-weight:850}
.ma-title{font-size:36px;font-weight:800;color:#f3faff;margin-top:6px}
.ma-sub{color:#acc9dc;margin-top:8px;font-size:14px}
</style>
<div class="ma-hero">
<div class="ma-kicker">◇ AURUM-AI V5 · QUANTITATIVE MARKET INTELLIGENCE</div>
<div class="ma-title">📈 Market Analysis · XOMUSD</div>
<div class="ma-sub">Historique Dukascopy · Cours bid OHLC · Analyse reproductible en UTC</div>
</div>
""", unsafe_allow_html=True)
st.info("**Lecture seule · marché historique.** Aucune transaction réelle, aucun cours live, aucun volume dans ce dataset. Les interruptions ne sont pas comblées.")

if not RAW.is_file():
    st.error(f"Dataset XOM original absent : {RAW}")
    st.stop()

@st.cache_data(show_spinner="Chargement des bougies historiques XOM…")
def source_cache(path: str, mtime_ns: int, file_size: int):
    return read_raw(path)

try:
    stat = RAW.stat()
    source = source_cache(str(RAW), stat.st_mtime_ns, stat.st_size)
except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as exc:
    st.error("Impossible de charger les données historiques. Aucun résultat simulé ne sera affiché.")
    st.exception(exc)
    st.stop()

first = source.index.min().date()
last = source.index.max().date()
default_start = max(first, last - timedelta(days=180))

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    start = st.date_input("Début (UTC)", value=default_start, min_value=first, max_value=last, key="ma_start")
with col2:
    end = st.date_input("Fin (UTC)", value=last, min_value=first, max_value=last, key="ma_end")
with col3:
    timeframe = st.selectbox("Granularité", ["1 jour", "1 heure", "5 min"], key="ma_timeframe")

if start > end:
    st.error("La date de début ne peut pas dépasser la date de fin.")
    st.stop()

raw = subset_utc(source, start, end)
if raw.empty:
    st.warning("Aucune bougie observée sur cette période. Les données ne seront ni inventées ni interpolées.")
    st.stop()

bars = aggregate_ohlc(raw, timeframe)
market = enrich_market(bars, timeframe)
segments = int(market["segment_id"].nunique())
monthly = month_coverage(raw)
long_gaps = gaps_table(raw)
start_close, last_close = float(market["close"].iloc[0]), float(market["close"].iloc[-1])
net_change = 100 * (last_close / start_close - 1)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Dernier close observé", f"{last_close:.2f} USD")
k2.metric("Variation bornes (%)", f"{net_change:+.2f}%")
k3.metric("Bougies observées (RAW)", f"{len(raw):,}".replace(",", " "))
k4.metric("Bougies agrégées", f"{len(market):,}".replace(",", " "))
k5.metric("Segments continus", segments)
st.caption(f"Source : `{RAW.relative_to(ROOT)}` · {market.index[0]} → {market.index[-1]} · Valeurs historiques bid, pas de volume. La variation compare les deux dernières/ premières observations, même si des interruptions les séparent.")

price_tab, indicators_tab, returns_tab, coverage_tab = st.tabs([
    "🕯️ Prix & Chandeliers", "📐 Indicateurs techniques", "📉 Rendements & Volatilité", "🕒 Couverture & Export"
])

PALETTE = {"cyan": "#28c9f4", "green": "#19dea1", "red": "#ff657b", "amber": "#ffc247", "muted": "#9eb7cc"}
LAYOUT = dict(template="plotly_dark", paper_bgcolor="#081525", plot_bgcolor="#081525", font=dict(color="#eaf5ff"))

def plot_candles(view: pd.DataFrame, overlays: bool = False):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=view.index, open=view["open"], high=view["high"],
        low=view["low"], close=view["close"], name="XOMUSD bid OHLC",
        increasing_line_color=PALETTE["green"], decreasing_line_color=PALETTE["red"],
    ))
    if overlays:
        for col, color, name in [("sma_20", PALETTE["cyan"], "SMA 20"),
                                 ("sma_50", PALETTE["amber"], "SMA 50"),
                                 ("ema_20", "#d38dfc", "EMA 20")]:
            # Prevent Plotly from connecting across NaN caused by gaps.
            fig.add_trace(go.Scatter(x=view.index, y=view[col], mode="lines", name=name,
                                     connectgaps=False, line=dict(color=color, width=1.4)))
    fig.update_layout(**LAYOUT, height=510, xaxis_rangeslider_visible=False,
                      xaxis_title="Date / heure UTC", yaxis_title="Prix bid (USD)",
                      margin=dict(l=15, r=10, t=32, b=10))
    return fig

with price_tab:
    st.subheader("Chandeliers historiques XOM")
    st.caption("Agrégation OHLC depuis les bougies RAW réellement présentes : aucune interpolation, aucune cotation de marché inventée.")
    MAX_CANDLES = 2500
    view = market.iloc[-MAX_CANDLES:] if len(market) > MAX_CANDLES else market
    if len(view) < len(market):
        st.warning(f"Graphique limité aux {MAX_CANDLES} dernières bougies sur {len(market):,} pour préserver la fluidité. L'analyse et les statistiques utilisent toute la période choisie.".replace(",", " "))
    st.plotly_chart(plot_candles(view), width="stretch")
    st.markdown("#### Extrait des bougies")
    st.dataframe(market[["open", "high", "low", "close"]].tail(25).reset_index(), hide_index=True, width="stretch")

with indicators_tab:
    st.subheader("SMA · EMA · RSI · ATR")
    st.warning("Les indicateurs intrajournaliers redémarrent après toute discontinuité de leur grille temporelle. En journalier, un écart de plus de 7 jours déclenche un nouveau segment (heuristique, pas calendrier NYSE). Les premières valeurs d'un segment peuvent donc être indisponibles.")
    st.plotly_chart(plot_candles(view, overlays=True), width="stretch")
    sub = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
                        subplot_titles=("RSI (14, Wilder)", "ATR (14, Wilder, USD)"))
    sub.add_trace(go.Scatter(x=view.index, y=view["rsi_14"], connectgaps=False,
                             name="RSI 14", line=dict(color=PALETTE["cyan"])), row=1, col=1)
    sub.add_trace(go.Scatter(x=view.index, y=view["atr_14"], connectgaps=False,
                             name="ATR 14", line=dict(color=PALETTE["amber"])), row=2, col=1)
    sub.update_layout(**LAYOUT, height=500, xaxis2_title="Date UTC", margin=dict(l=12, r=12, t=52, b=12))
    sub.update_yaxes(range=[0, 100], row=1, col=1)
    st.plotly_chart(sub, width="stretch")

with returns_tab:
    st.subheader("Variations calculées entre observations contiguës")
    st.caption("Les rendements traversant une interruption temporelle sont exclus (NaN). La volatilité ci-dessous est l'écart-type mobile des log-rendements, non annualisé. Elle n'est pas une prévision de risque.")
    c1, c2, c3 = st.columns(3)
    valid_returns = market["return_pct"].dropna()
    c1.metric("Rendements contigus", len(valid_returns))
    c2.metric("Variation moyenne / barre", f"{valid_returns.mean():.4f}%" if not valid_returns.empty else "N/A")
    c3.metric("Écart-type / barre", f"{valid_returns.std():.4f}%" if len(valid_returns) > 1 else "N/A")
    ret_fig = go.Figure()
    ret_fig.add_trace(go.Scattergl(x=view.index, y=view["return_pct"], mode="markers",
                                   marker=dict(color=PALETTE["cyan"], size=3), name="Rendement (%)"))
    ret_fig.update_layout(**LAYOUT, height=330, xaxis_title="Date UTC", yaxis_title="% par barre")
    st.plotly_chart(ret_fig, width="stretch")
    vol_fig = go.Figure()
    vol_fig.add_trace(go.Scatter(x=view.index, y=view["volatility_20_pct"], mode="lines",
                                 connectgaps=False, line=dict(color=PALETTE["amber"]), name="σ20 (%)"))
    vol_fig.update_layout(**LAYOUT, height=310, xaxis_title="Date UTC", yaxis_title="Écart-type des log-rendements (%)")
    st.plotly_chart(vol_fig, width="stretch")

with coverage_tab:
    st.subheader("Couverture temporelle observée")
    st.warning("Un écart temporel ne signifie pas automatiquement une donnée manquante. Les événements de plus de 7 jours sont signalés pour examen. Aucune comparaison à un calendrier boursier officiel n'est faite.")
    coverage_fig = go.Figure(go.Bar(x=monthly["Mois UTC"], y=monthly["Bougies observées"], marker_color=PALETTE["cyan"]))
    coverage_fig.update_layout(**LAYOUT, height=350, xaxis_title="Mois UTC", yaxis_title="Bougies M5 RAW observées")
    st.plotly_chart(coverage_fig, width="stretch")
    st.metric("Interruptions > 7 jours dans la sélection", len(long_gaps))
    st.dataframe(long_gaps, hide_index=True, width="stretch")
    st.markdown("#### Exports dérivés, sans écriture sur le RAW")
    export = market.reset_index()
    st.download_button("📥 Télécharger les bougies et indicateurs (CSV)",
                       export.to_csv(index=False).encode("utf-8-sig"),
                       file_name="aurum_xom_market_analysis.csv", mime="text/csv")
    st.download_button("📥 Télécharger la couverture mensuelle (CSV)",
                       monthly.to_csv(index=False).encode("utf-8-sig"),
                       file_name="aurum_xom_monthly_coverage.csv", mime="text/csv")
    st.caption("Exports de visualisation et d'étude, non validés comme dataset d'entraînement. Le fichier RAW d'origine n'est jamais modifié.")
