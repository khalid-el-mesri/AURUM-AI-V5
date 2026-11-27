"""AURUM-AI V5 | Feature Engineering Explorer, built on existing Parquet/NPZ.
No recomputation of training assets. No writes to data/ or model artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.v5_pages._common import demand_session
from frontend.services.feature_engineering_audit import (
    audit_feature_frame, datetime_axis, feature_missing_table,
    find_feature_sets, find_window_sets, inspect_window_file,
    inspect_window_preview, read_features, split_boundary_check,
)

demand_session()
ROOT = Path(__file__).resolve().parents[2]
COLOR = {"cyan": "#24d4f5", "green": "#17dfa4", "gold": "#fac657"}
BG = "#081525"

st.markdown("""
<style>
.fe-hero {background:linear-gradient(120deg,#0b314a,#071522);border:1px solid #245e85;
 border-radius:17px;padding:27px 30px;margin-bottom:15px}
.fe-tag {font-size:11px;font-weight:800;letter-spacing:2.4px;color:#35d6f9}
.fe-title {font-size:35px;font-weight:900;color:#f1f8ff;margin-top:9px}
.fe-sub {font-size:14px;color:#b7cfe0;margin-top:6px}
</style>
<div class="fe-hero">
  <div class="fe-tag">◇ AURUM-AI V5 · DEEP LEARNING DATA INTELLIGENCE</div>
  <div class="fe-title">🧮 Feature Engineering · XOM</div>
  <div class="fe-sub">Analyse en lecture seule des features Parquet et des fenêtres LSTM · GRU · CNN1D</div>
</div>
""", unsafe_allow_html=True)
st.info("**Lecture seule.** Les fichiers RAW, CLEAN, FEATURES, WINDOWS et les modèles entraînés ne sont pas modifiés. Les indicateurs présentés sont ceux déjà stockés, pas de nouveaux entraînements.")
feature_files = find_feature_sets(ROOT)
window_files = find_window_sets(ROOT)
if not feature_files and not window_files:
    st.error("Aucun fichier Parquet dans data/03_features ni NPZ dans data/04_windows.")
    st.stop()

@st.cache_data(show_spinner="Ouverture du dataset de features…")
def cached_features(path: str, mtime_ns: int, file_size: int):
    return read_features(Path(path))

@st.cache_data(show_spinner="Inspection du fichier de fenêtres…")
def cached_window_audit(path: str, mtime_ns: int, file_size: int):
    return inspect_window_file(Path(path))

frame = None
report = None
chosen_feature = None
if feature_files:
    chosen_feature = st.selectbox("Dataset de features existant", feature_files,
                                  format_func=lambda p: p.name,
                                  index=next((i for i, p in enumerate(feature_files) if 'nyse_v3' in p.name), 0))
    try:
        meta = chosen_feature.stat()
        frame = cached_features(str(chosen_feature), meta.st_mtime_ns, meta.st_size)
        report = audit_feature_frame(frame)
    except (OSError, ImportError, ValueError, TypeError) as exc:
        st.error("Impossible de lire le fichier Parquet. Vérifie que `pyarrow` est installé dans ton environnement.")
        st.exception(exc)
        st.stop()
else:
    st.warning("Aucun dataset de features Parquet trouvé. Les fenêtres peuvent toujours être explorées.")

if report is not None:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Lignes", f"{report['rows']:,}".replace(',', ' '))
    m2.metric("Colonnes", report['columns'])
    m3.metric("Segments observés", report['n_segments'] if report['n_segments'] is not None else "N/A")
    m4.metric("Valeurs nulles", f"{report['null_cells']:,}".replace(',', ' '))
    st.caption(f"Source : `data/03_features/{chosen_feature.name}` · {meta.st_size / (1024 ** 2):.2f} Mio · Début : {report['first_utc']} · Fin : {report['last_utc']}")

(tab_overview, tab_stats, tab_visual, tab_correlation,
 tab_windows, tab_leakage, tab_export) = st.tabs([
    "📚 Feature Overview", "📐 Technical Indicators", "📉 Feature Visualization",
    "🔗 Correlations", "🪟 Window Explorer", "🔐 Leakage & Gaps", "📄 Export Audit",
])

with tab_overview:
    st.markdown("#### Inventaire des versions de features")
    if feature_files:
        st.dataframe(pd.DataFrame([{
            'Fichier': p.name, 'Taille (Mio)': round(p.stat().st_size / (1024**2), 2)
        } for p in feature_files]), hide_index=True, width='stretch')
    if frame is not None:
        st.markdown('#### Colonnes et valeurs manquantes')
        st.dataframe(feature_missing_table(frame), hide_index=True, width='stretch')
        st.markdown('#### Aperçu des observations stockées')
        st.dataframe(frame.head(15), hide_index=True, width='stretch')
        st.caption("Les NaN en début de segment (p. ex. retour ou SMA) peuvent être des périodes de chauffe normales. Leur nombre n'est pas corrigé automatiquement.")

with tab_stats:
    if frame is None:
        st.warning('Aucun dataset de features sélectionné.')
    else:
        known = [v for v in ('return_1', 'log_return_1', 'range_pct', 'body_pct',
                             'sma_5', 'sma_5_distance', 'ema_10', 'volatility_10') if v in frame]
        st.markdown('#### Variables quantitatives déjà calculées')
        if known:
            stats = frame[known].describe(percentiles=[.05, .5, .95]).T
            st.dataframe(stats, width='stretch')
        else:
            st.warning('Aucune des variables usuelles n’est présente dans cette version.')
        st.caption("SMA/EMA et volatilité proviennent du fichier existant. RSI/ATR ne sont pas présents dans ces Parquet ; cette page ne prétend pas les y trouver.")
        st.markdown('#### Disponibilité des features')
        if 'feature_available_at_ms' in frame and 'timestamp' in frame:
            delay = (pd.to_numeric(frame['feature_available_at_ms'], errors='coerce')
                     - pd.to_numeric(frame['timestamp'], errors='coerce')) / 60000.0
            c1, c2, c3 = st.columns(3)
            c1.metric('Délai médian (min)', f'{delay.median():.1f}')
            c2.metric('Disponibilité avant la bougie', report['available_before_timestamp'])
            c3.metric('Disponibilité au timestamp', report['available_at_timestamp'])
        else:
            st.warning("Cette version n'a pas de champ `feature_available_at_ms` ; impossible de certifier l'heure de disponibilité de chaque feature.")

with tab_visual:
    if frame is None:
        st.warning('Aucun dataset de features sélectionné.')
    else:
        numeric_names = [c for c in frame.select_dtypes(include=[np.number]).columns
                         if c not in ('timestamp', 'feature_available_at_ms', 'segment_id', 'bar_in_segment')]
        if not numeric_names:
            st.warning('Aucune feature numérique à afficher.')
        else:
            chosen = st.selectbox('Variable à visualiser', numeric_names,
                                  index=numeric_names.index('close') if 'close' in numeric_names else 0)
            sample_size = st.slider('Points maximum du graphique', min_value=200, max_value=3000,
                                    value=1300, step=100)
            stride = max(1, int(np.ceil(len(frame) / sample_size)))
            dates = datetime_axis(frame)
            y = pd.to_numeric(frame[chosen], errors='coerce')
            df_plot = pd.DataFrame({'Horodatage UTC': dates, chosen: y}).iloc[::stride].dropna()
            if df_plot.empty:
                st.warning('Aucune valeur exploitable pour cette sélection.')
            else:
                fig = go.Figure(go.Scattergl(x=df_plot['Horodatage UTC'], y=df_plot[chosen],
                                             mode='markers', marker={'size': 4, 'color': COLOR['cyan']},
                                             name=chosen))
                fig.update_layout(template='plotly_dark', paper_bgcolor=BG, plot_bgcolor=BG,
                                  height=400, xaxis_title='UTC', yaxis_title=chosen)
                st.plotly_chart(fig, width='stretch')
                st.caption('Échantillonnage déterministe uniquement pour la visualisation. Un nuage de points ne relie pas les périodes interrompues.')
            st.markdown('#### Distribution sur les valeurs disponibles')
            vals = y.dropna()
            if not vals.empty:
                fig2 = go.Figure(go.Histogram(x=vals.iloc[::max(1, len(vals)//30000)],
                                              nbinsx=70, marker_color=COLOR['cyan']))
                fig2.update_layout(template='plotly_dark', paper_bgcolor=BG, plot_bgcolor=BG,
                                   height=300, xaxis_title=chosen, yaxis_title='Observations échantillonnées')
                st.plotly_chart(fig2, width='stretch')

with tab_correlation:
    if frame is None:
        st.warning('Aucun dataset de features sélectionné.')
    else:
        available = [c for c in frame.select_dtypes(include=[np.number]).columns
                     if c not in ('timestamp', 'feature_available_at_ms', 'segment_id')]
        if len(available) < 2:
            st.warning('Deux variables numériques au minimum sont nécessaires.')
        else:
            defaults = [c for c in ('return_1','log_return_1','range_pct','body_pct',
                                    'sma_5_distance','volatility_10','close') if c in available]
            selection = st.multiselect('Variables à corréler', available, default=defaults[:12])
            if len(selection) >= 2:
                matrix = frame[selection].corr()
                fig = go.Figure(go.Heatmap(z=matrix.to_numpy(), x=matrix.columns,
                                           y=matrix.index, colorscale='RdBu', zmin=-1,zmax=1,
                                           reversescale=True))
                fig.update_layout(template='plotly_dark', paper_bgcolor=BG, plot_bgcolor=BG,
                                  height=max(420, 45*len(selection)), margin=dict(l=50, r=25, t=35, b=80))
                st.plotly_chart(fig, width='stretch')
                st.caption('Corrélation de Pearson descriptive, sans relation de causalité ni validation prédictive hors échantillon.')
            else:
                st.info('Choisis au moins deux colonnes.')

with tab_windows:
    st.markdown('#### Fenêtres déjà générées (lecture seule)')
    if not window_files:
        st.warning('Aucun .npz trouvé dans data/04_windows.')
    else:
        st.caption(f'{len(window_files)} archives NPZ existantes détectées ; aucune nouvelle fenêtre n’est créée.')
        chosen_window = st.selectbox('Fichier de fenêtres', window_files,
                                     format_func=lambda p: str(p.relative_to(ROOT)))
        try:
            wstat = chosen_window.stat()
            win = cached_window_audit(str(chosen_window), wstat.st_mtime_ns, wstat.st_size)
        except (OSError, ValueError, KeyError) as exc:
            st.error('Lecture sécurisée du fichier NPZ impossible (pickle désactivé).')
            st.exception(exc)
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric('Samples', win['samples'] if win['samples'] is not None else 'N/A')
            dims = win['fields'].get('X', {}).get('shape', [])
            c2.metric('Longueur séquence', dims[1] if len(dims) == 3 else 'N/A')
            c3.metric('Features / pas', dims[2] if len(dims) == 3 else 'N/A')
            st.caption('Variables : ' + (', '.join(win['feature_names']) if win['feature_names'] else 'non renseignées'))
            st.dataframe(pd.DataFrame([{'Champ': k, 'Shape': str(v['shape']), 'Dtype': v['dtype']}
                                      for k, v in win['fields'].items()]), hide_index=True, width='stretch')
            if win['errors']:
                st.error('Contrôles structurels à examiner : ' + ' · '.join(win['errors']))
            st.markdown('#### Exemple d’une fenêtre réelle')
            if win['samples']:
                idx = st.number_input('Index de l’échantillon', min_value=0, max_value=win['samples'] - 1, value=0)
                st.dataframe(inspect_window_preview(chosen_window, idx), hide_index=True, width='stretch')
                st.caption('Les valeurs de X peuvent être normalisées ; elles ne correspondent pas obligatoirement aux prix USD.')

with tab_leakage:
    st.markdown('#### Contrôles temporels et limites')
    if report is not None:
        indicators = [
            ('Timestamps répétés', report['duplicate_timestamps']),
            ('Chronologie décroissante', report['reverse_chronology']),
            ('Gaps M5 à l’intérieur d’un segment', report['gaps_within_segment']),
            ('Feature annoncée disponible avant son timestamp', report['available_before_timestamp']),
            ('Timestamps non valides', report['invalid_timestamps']),
            ('Dates datetime_utc non concordantes', report['datetime_mismatch']),
        ]
        st.dataframe(pd.DataFrame([{'Contrôle': k, 'Nombre': v if v is not None else 'Non disponible'}
                                   for k, v in indicators]), hide_index=True, width='stretch')
    st.warning('Un test de métadonnées **ne prouve pas** l’absence de data leakage : il faut aussi auditer le découpage chronologique, les définitions des labels, les normalisations fit sur Train uniquement, ainsi que le code générateur des fenêtres.')
    if window_files:
        st.markdown('#### Audits Train → Validation, par horizon')
        candidates = [
            ('Base', 'train_windows.npz', 'validation_windows.npz'),
            ('NYSE V3 15m', 'nyse_v3/train_15m_windows.npz', 'nyse_v3/validation_15m_windows.npz'),
            ('NYSE V3 30m', 'nyse_v3/train_30m_windows.npz', 'nyse_v3/validation_30m_windows.npz'),
            ('NYSE V3 60m', 'nyse_v3/train_60m_windows.npz', 'nyse_v3/validation_60m_windows.npz'),
        ]
        if st.button('Vérifier les frontières Train / Validation', key='fe_check_splits'):
            comparisons = []
            for label, a, b in candidates:
                train = ROOT / 'data' / '04_windows' / a
                val = ROOT / 'data' / '04_windows' / b
                if train.is_file() and val.is_file():
                    try:
                        status = split_boundary_check(train, val)
                        comparisons.append({'Ensemble':label, **status})
                    except (OSError, ValueError) as exc:
                        comparisons.append({'Ensemble':label, 'status':f'Erreur lecture : {exc}'})
            if comparisons:
                st.dataframe(pd.DataFrame(comparisons), hide_index=True, width='stretch')
            else:
                st.warning('Aucune paire Train / Validation disponible.')
        st.caption('Un chevauchement ne permet pas à lui seul d’identifier sa cause ; il signale une frontière à examiner.')

with tab_export:
    st.markdown('#### Export d’audit reproductible (sans écrire sur disque)')
    payload = {
        'project': 'AURUM-AI V5', 'type': 'Feature Engineering inventory and data audit',
        'read_only': True, 'source': str(chosen_feature.relative_to(ROOT)) if chosen_feature else None,
        'feature_audit': report,
        'feature_files': [str(p.relative_to(ROOT)) for p in feature_files],
        'window_files': [str(p.relative_to(ROOT)) for p in window_files],
        'important_limitations': [
            'Aucun réentraînement, aucune modification de Parquet ou NPZ.',
            'Les contrôles temporels ne démontrent pas à eux seuls une absence de fuite des labels ou normalisation.',
            'Les NaN de démarrage d’un segment peuvent être attendus.',
            'Les écarts de couverture historique ne sont pas interpolés.',
        ],
    }
    st.download_button('📥 Télécharger l’audit JSON',
                       json.dumps(payload, ensure_ascii=False, indent=2).encode('utf-8'),
                       file_name='aurum_xom_feature_audit.json', mime='application/json',
                       width='stretch')
    if frame is not None:
        st.download_button('📥 Télécharger un aperçu CSV (5 000 lignes maximum)',
                           frame.head(5000).to_csv(index=False).encode('utf-8'),
                           file_name=f'{chosen_feature.stem}_preview.csv', mime='text/csv',
                           width='stretch')
    st.caption('Les exports sont produits en mémoire pour téléchargement. Aucune donnée d’entraînement n’est remplacée.')
