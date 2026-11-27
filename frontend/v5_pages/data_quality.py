"""AURUM-AI V5 — real Data Quality dashboard for historical XOM M5 raw CSV.

La page est en lecture seule : aucun fichier raw ou clean n'est modifié.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.v5_pages._common import demand_session
from frontend.services.data_quality_audit import audit_xom_frame, sha256_file

# Authenticate before opening or inspecting any files.
demand_session()

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "01_raw" / "xomususd-m5-bid-2017-01-26-2026-10-01.csv"

st.markdown("""
<style>
.dq-hero {background:linear-gradient(110deg,#0c2f48,#071522);border:1px solid #22628c;
 border-radius:17px;padding:26px 30px;margin-bottom:20px}
.dq-kicker{color:#29d5fa;letter-spacing:2.5px;font-size:12px;font-weight:800}
.dq-title{color:#f4f8ff;font-size:36px;font-weight:850;margin-top:8px}
.dq-sub{color:#b1cbdd;font-size:14px;margin-top:6px}
</style>
<div class="dq-hero">
<div class="dq-kicker">◇ AURUM-AI V5 · XOM DATA INTELLIGENCE</div>
<div class="dq-title">🔍 Data Quality · Raw Dataset Audit</div>
<div class="dq-sub">Audit reproductible des données OHLC M5 historiques · Source originale préservée</div>
</div>
""", unsafe_allow_html=True)

st.info("**Mode lecture seule.** Cette page ne modifie, ne supprime et ne nettoie aucune ligne du dataset original.")
if not RAW.is_file():
    st.error(f"Dataset RAW introuvable : {RAW}")
    st.caption("Vérifie le chemin data/01_raw et le nom exact du CSV. Aucune donnée fictive ne sera affichée.")
    st.stop()

@st.cache_data(show_spinner="Audit du dataset XOM en cours…")
def get_audit(path: str, mtime_ns: int, file_bytes: int):
    # mtime and size are explicit cache keys: regenerating the file invalidates cache.
    source = pd.read_csv(path, low_memory=False)
    return source, audit_xom_frame(source)

try:
    stat = RAW.stat()
    data, audit = get_audit(str(RAW), stat.st_mtime_ns, stat.st_size)
except (OSError, UnicodeError, pd.errors.ParserError, ValueError, OverflowError) as exc:
    st.error("Impossible d'analyser le CSV RAW sans ambiguïté.")
    st.exception(exc)
    st.stop()

s = audit["summary"]
num_fmt = lambda n: f"{n:,}".replace(",", " ")

st.markdown("### Dataset Overview")
st.caption(f"Fichier analysé : `{RAW.relative_to(ROOT)}` · {stat.st_size / (1024**2):.2f} Mio sur disque")
cols = st.columns(4)
cols[0].metric("Lignes", num_fmt(s["n_rows"]))
cols[1].metric("Colonnes", s["n_columns"])
cols[2].metric("Empreinte mémoire", f"{s['memory_mib']:.2f} Mio")
cols[3].metric("Valeurs nulles", num_fmt(s["null_cells"]))

c1, c2 = st.columns(2)
c1.write(f"**Premier timestamp UTC :** {s['first_utc'] or 'non disponible'}")
c2.write(f"**Dernier timestamp UTC :** {s['last_utc'] or 'non disponible'}")

basic, completeness_tab, temporal, ohlc_tab, export_tab = st.tabs([
    "📊 Overview", "🧩 Missing & Duplicates", "🕒 Time Gaps", "📈 OHLC Integrity", "📄 Export Report"
])

with basic:
    st.markdown("#### Répartition annuelle (nombre de lignes horodatées valides)")
    year = audit["yearly"]
    if not year.empty:
        fig = go.Figure(go.Bar(x=year["Année"].astype(str), y=year["Lignes"], marker_color="#24c9f4"))
        fig.update_layout(template="plotly_dark", paper_bgcolor="#081525", plot_bgcolor="#081525",
                          xaxis_title="Année (UTC)", yaxis_title="Nombre de lignes", height=330)
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("Aucun horodatage interprétable.")

    st.markdown("#### Aperçu de l'historique de clôture")
    st.caption("Pour éviter un graphique lourd, un échantillon déterministe de points est affiché. Le rapport d'audit porte sur **toutes** les lignes.")
    valid = pd.DataFrame({"Date UTC": audit["parsed_time"], "Close": audit["numeric_close"]}).dropna()
    if not valid.empty:
        valid = valid.sort_values("Date UTC").reset_index(drop=True)
        stride = max(1, len(valid) // 2500)
        sampled = valid.iloc[::stride]
        chart = go.Figure(go.Scattergl(x=sampled["Date UTC"], y=sampled["Close"], mode="lines",
                                       name="Close", line=dict(color="#28c9f4", width=1.5)))
        chart.update_layout(template="plotly_dark", paper_bgcolor="#081525", plot_bgcolor="#081525",
                            xaxis_title="Horodatage UTC", yaxis_title="Prix close (source)", height=400)
        st.plotly_chart(chart, width="stretch")
    st.markdown("#### Extrait du fichier source")
    st.dataframe(data.head(15), hide_index=True, width="stretch")

with completeness_tab:
    c1, c2, c3 = st.columns(3)
    c1.metric("Lignes identiques répétées", num_fmt(s["identical_duplicate_rows"]))
    c2.metric("Timestamps répétés", num_fmt(s["repeated_timestamp_rows"]))
    c3.metric("Horodatages invalides", num_fmt(s["invalid_timestamps"]))
    st.markdown("#### Valeurs manquantes, types et taux")
    st.dataframe(audit["completeness"], hide_index=True, width="stretch")
    st.caption("Les doublons de timestamp sont comptés au-delà de la première occurrence. Les lignes identiques ne sont pas effacées.")

with temporal:
    c1, c2, c3 = st.columns(3)
    c1.metric("Intervalles > 5 min", num_fmt(s["gap_count"]))
    c2.metric("Timestamps hors grille M5", num_fmt(s["off_grid_timestamps"]))
    c3.metric("Chronologie inversée", num_fmt(s["reverse_chronology_rows"]))
    st.markdown("#### Analyse des écarts temporels")
    st.warning("Un intervalle supérieur à 5 minutes **n'est pas automatiquement une erreur** : clôtures, nuits, week-ends, jours fériés ou horaires du fournisseur peuvent l'expliquer. Les catégories ci-dessous reposent sur le changement de date **UTC**, pas sur un calendrier officiel NYSE.")
    a, b, c = st.columns(3)
    a.metric("Écarts même date UTC", num_fmt(s["same_utc_date_gaps"]))
    b.metric("Écarts sur changement de date UTC", num_fmt(s["cross_utc_date_gaps"]))
    c.metric("Intervalles hors cadence 5min", num_fmt(s["non_m5_intervals"]))
    st.caption("Les créneaux M5 théoriques ignorent volontairement les calendriers de marché : **ne pas les utiliser comme nombre de valeurs réellement manquantes**.")
    gaps = audit["gaps"]
    if gaps.empty:
        st.success("Aucun intervalle > 5 minutes observé entre timestamps uniques valides.")
    else:
        option = st.selectbox("Filtrer les écarts", ["Tous", "Même date UTC — à examiner", "Changement de date UTC — calendrier nécessaire"])
        shown = gaps if option == "Tous" else gaps[gaps["Catégorie indicative"] == option]
        st.dataframe(shown.head(300), hide_index=True, width="stretch")
        st.caption(f"{len(shown)} écarts dans cette catégorie · affichage des 300 durées les plus longues au maximum.")
        st.download_button("Télécharger tous les intervalles observés (CSV)",
                           data=gaps.to_csv(index=False).encode("utf-8-sig"), file_name="aurum_xom_time_gaps.csv", mime="text/csv")

with ohlc_tab:
    a, b, c = st.columns(3)
    a.metric("Lignes OHLC incohérentes", num_fmt(s["ohlc_inconsistent_rows"]))
    b.metric("Prix non numériques / non finis", num_fmt(s["invalid_price_rows"]))
    c.metric("Prix nuls / négatifs", num_fmt(s["nonpositive_price_rows"]))
    st.markdown("#### Règles vérifiées")
    st.markdown("Les prix doivent être numériques, finis et strictement positifs. `high ≥ max(open, close, low)` et `low ≤ min(open, close, high)`. Les bougies où les quatre prix sont égaux sont **valides**.")
    st.markdown("#### Exemples d'anomalies détectées")
    st.dataframe(audit["evidence"], hide_index=True, width="stretch")
    st.caption("Jusqu'à 15 exemples par règle. Une ligne peut violer plusieurs règles ; les comptes ne doivent pas être additionnés.")

with export_tab:
    st.markdown("#### Rapport d'audit, reproductible et traçable")
    if st.button("Calculer l'empreinte SHA-256 du fichier RAW", key="dq_hash"):
        st.session_state["dq_file_sha256"] = sha256_file(RAW)
    fingerprint = st.session_state.get("dq_file_sha256")
    if fingerprint:
        st.code(fingerprint, language=None)
    report = {
        "project": "AURUM-AI V5",
        "asset": "XOMUSD (source Dukascopy, bid OHLC, M5)",
        "input_file": RAW.name,
        "size_bytes": stat.st_size,
        "file_sha256": fingerprint,
        "method": "Lecture seule; chronologie UTC, sans calendrier NYSE; aucune suppression/correction",
        "statistics": s,
        "notes": [
            "Les intervalles >5min ne représentent pas nécessairement des données manquantes.",
            "Les horaires de marché (fériés, DST, jours non ouvrés) requièrent un calendrier indépendant.",
            "Les contrôles constatent des propriétés du CSV et non des performances de trading.",
        ],
    }
    st.json(report, expanded=False)
    st.download_button("📥 Télécharger le rapport JSON", json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
                       file_name="aurum_xom_data_quality_report.json", mime="application/json", type="primary")
    st.download_button("📥 Télécharger les exemples d'anomalies (CSV)", audit["evidence"].to_csv(index=False).encode("utf-8-sig"),
                       file_name="aurum_xom_quality_examples.csv", mime="text/csv")
    st.caption("Ce rapport se base sur le fichier présent localement lors de l'analyse ; l'empreinte SHA-256 est facultative et peut être calculée pour identifier sa version exacte.")
