"""AURUM-AI V5 / Phase 9: Portfolio Simulator — evidence-based read-only view.

Intentionally never calls PortfolioAgent or ExecutionAgent and never creates orders.
Run with: python -m streamlit run frontend/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.services.portfolio_simulator_engine import (
    BASE, RUNS, audit_run, join_events, load_run, recorded_equity,
    report_json, run_metrics, trade_tape,
)


@st.cache_data(show_spinner=False)
def _load(root: str, horizon: int, decision_mtime: int, trade_mtime: int):
    return load_run(Path(root), horizon)


def _load_horizon(horizon: int):
    dname, tname = RUNS[horizon]
    dp, tp = ROOT / BASE / dname, ROOT / BASE / tname
    if not dp.is_file() or not tp.is_file():
        raise FileNotFoundError(f"Manque {dname if not dp.is_file() else tname}")
    return _load(str(ROOT), horizon, dp.stat().st_mtime_ns, tp.stat().st_mtime_ns)


def _usd(value) -> str:
    if value is None or pd.isna(value):
        return 'Non établi'
    return f'{float(value):,.2f} USD'


def _equity_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for n, (_, seg) in enumerate(frame.groupby('segment', sort=False)):
        fig.add_trace(go.Scatter(
            x=seg['datetime_utc'], y=seg['equity_after_usd'], mode='lines+markers'
            if len(seg) < 30 else 'lines',
            name='Equity consignée' if n == 0 else 'Segment historique',
            showlegend=n == 0, line={'color': '#30c9e9', 'width': 2},
            marker={'size': 5},
            hovertemplate='%{x}<br>%{y:.2f} USD<extra></extra>',
        ))
    fig.update_layout(template='plotly_dark', height=390,
        paper_bgcolor='#061522', plot_bgcolor='#061522',
        margin=dict(l=18, r=18, t=15, b=45),
        xaxis_title='Date de marché (UTC)', yaxis_title='Capital enregistré (USD)',
        legend={'orientation': 'h', 'y': 1.08})
    return fig


def _position_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame['datetime_utc'], y=frame['holdings_after'],
        mode='lines+markers', name='Titres après transaction',
        line={'color': '#20d5a6', 'shape': 'hv'}, marker={'size': 5}))
    fig.update_layout(template='plotly_dark', height=310,
        paper_bgcolor='#061522', plot_bgcolor='#061522',
        margin=dict(l=18, r=18, t=20, b=40),
        xaxis_title='Horodatage historique (UTC)', yaxis_title='Quantité détenue')
    return fig


def _render() -> None:
    st.markdown('''<style>
    .ps-hero{padding:24px 28px;background:linear-gradient(108deg,#123951,#061522);
    border:1px solid #1d688c;border-radius:16px;margin:12px 0 10px}
    .ps-hero .eyebrow{font-size:11px;letter-spacing:2px;color:#50d8f7;font-weight:700}
    .ps-hero h1{font-size:34px;color:#f4f8ff;margin:7px 0 6px}
    .ps-hero p{color:#bcd7e9;margin:0}
    </style>''', unsafe_allow_html=True)
    st.markdown('''<div class="ps-hero"><div class="eyebrow">◇ AURUM-AI V5 · PORTFOLIO INTELLIGENCE</div>
    <h1>💼 Portfolio Simulator</h1>
    <p>Historical Capital · Virtual Holdings · Trade Audit · Research Only</p></div>''',
    unsafe_allow_html=True)
    st.info('**Lecture seule — simulation historique.** Les positions et performances sont déjà consignées '
            'dans les CSV du backtesting. Aucun agent n’est relancé et aucune transaction n’est créée. '
            'Les horizons 30m et 60m sont deux expériences distinctes.')

    horizon = st.selectbox('Expérience historique', (60, 30),
                           format_func=lambda x: f'Consensus + Trend {x} minutes',
                           key='portfolio_simulator_horizon')
    try:
        decisions, trades = _load_horizon(horizon)
        audit = audit_run(decisions, trades)
        metrics = run_metrics(decisions, trades)
        events = join_events(decisions, trades)
        tape = trade_tape(trades)
        equity = recorded_equity(decisions)
    except (ValueError, TypeError, KeyError, OSError, IndexError) as err:
        st.error(f'Impossible de charger le portefeuille {horizon}m : {err}')
        return

    st.caption(f'Sources : `{BASE / RUNS[horizon][0]}` et `{BASE / RUNS[horizon][1]}`')
    a, b, c, d = st.columns(4)
    a.metric('Capital à la dernière décision', _usd(metrics['last_recorded_equity_usd']))
    b.metric('Cash après la dernière transaction', _usd(metrics['last_cash_usd']))
    c.metric('Positions XOM après la dernière transaction', f"{metrics['last_holdings']:.6f}")
    d.metric('Transactions virtuelles consignées', audit['transactions'])
    if all(audit[k] == 0 for k in (
        'unmatched_executed_decisions', 'unmatched_transactions',
        'cash_mismatch_gt_1_cent', 'cash_flow_mismatch_gt_1_cent',
        'trade_cash_flow_mismatch_gt_2_cents', 'trade_reverse_time')):
        st.success('Rapprochement historique cohérent selon les contrôles appliqués (horodatage, action, cash, frais).')
    else:
        st.warning('Des écarts ont été détectés : ouvrir « Audit » avant d’interpréter les résultats.')

    tabs = st.tabs(['📊 Vue d’ensemble', '💰 Cash & Positions', '📋 Transactions',
                    '🛡️ Risque & Frais', '⚖️ Comparaison 30m / 60m', '🔍 Audit', '📥 Export'])
    with tabs[0]:
        st.subheader(f'Courbe de capital consignée — {horizon} minutes')
        st.plotly_chart(_equity_chart(equity), width='stretch')
        st.caption('Uniquement les points de capital enregistrés après les décisions. Les interruptions '
                   'supérieures à sept jours ne sont pas reliées. Il ne s’agit pas d’une courbe de prix continue.')
        x, y, z = st.columns(3)
        x.metric('Référence de capital initial (si démontrable)', _usd(metrics['initial_cash_reference_usd']))
        y.metric('Rendement sur la référence', f"{metrics['return_vs_reference_pct']:+.3f} %"
                 if metrics['return_vs_reference_pct'] is not None else 'Non établi')
        z.metric('Drawdown sur décisions enregistrées',
                 f"{metrics['max_observed_decision_drawdown_pct']:.3f} %")
        st.caption('La référence initiale est reconstruite uniquement si la première transaction ouvre '
                   'une position BUY depuis zéro titre. Le drawdown ci-dessus ignore les mouvements '
                   'entre décisions : ce n’est pas un max drawdown intraday certifié.')
        st.dataframe(events.tail(30).iloc[::-1], width='stretch', hide_index=True)

    with tabs[1]:
        st.subheader('Positions historiques : état après chaque transaction')
        st.plotly_chart(_position_chart(tape), width='stretch')
        st.dataframe(tape[['datetime_utc', 'action', 'price', 'quantity',
                            'holdings_after', 'cash_after_usd',
                            'trade_time_portfolio_value_usd']].iloc[::-1],
                     width='stretch', hide_index=True)
        st.caption('La valeur instantanée des positions est approximée à partir du prix effectivement '
                   'enregistré pour chaque transaction. Ce n’est pas une valorisation continue du marché.')

    with tabs[2]:
        st.subheader('Registre des transactions simulées')
        a, b, c = st.columns(3)
        a.metric('BUY enregistrés', metrics['buy_count'])
        b.metric('SELL enregistrés', metrics['sell_count'])
        c.metric('P&L réalisé consigné', _usd(metrics['realized_pnl_recorded_usd']))
        action = st.selectbox('Action', ['Toutes', 'BUY', 'SELL'], key='portfolio_action_filter')
        filtered = tape if action == 'Toutes' else tape[tape['action'].str.upper() == action]
        st.dataframe(filtered.iloc[::-1], width='stretch', hide_index=True)
        st.caption('Les événements portent des statuts historiques EXECUTED, qui désignent uniquement '
                   'l’exécution VIRTUELLE dans ce backtest.')

    with tabs[3]:
        st.subheader('Frais et risque : mesures historiques')
        x, y, z = st.columns(3)
        x.metric('Frais de transaction consignés', _usd(metrics['fees_recorded_usd']))
        y.metric('P&L réalisé consigné', _usd(metrics['realized_pnl_recorded_usd']))
        z.metric('Drawdown observé aux décisions', f"{metrics['max_observed_decision_drawdown_pct']:.3f} %")
        st.warning('Les règles du Risk Agent ne sont PAS recalculées ici. Un statut APPROVED dans le '
                   'CSV est une décision historique, non une nouvelle autorisation d’acheter. '
                   'Les métriques calculées sur les décisions ne capturent pas tous les risques intrabar.')
        if 'risk_status' in decisions:
            st.bar_chart(decisions['risk_status'].fillna('NON RENSEIGNÉ').value_counts())
        if 'system_state' in decisions:
            st.dataframe(decisions[['timestamp_ms', 'strategy_action', 'risk_status',
                                    'system_state', 'equity_after_usd']].tail(50).iloc[::-1],
                         width='stretch', hide_index=True)

    with tabs[4]:
        st.subheader('Comparaison séparée — deux backtests historiques')
        st.info('Chaque indicateur est calculé à partir des fichiers de sa propre expérience. '
                'Aucun événement ni courbe de capital n’est joint entre 30m et 60m.')
        rows = []
        for h in (30, 60):
            try:
                dh, th = _load_horizon(h)
                mh, ah = run_metrics(dh, th), audit_run(dh, th)
                rows.append({'Horizon (min)': h, 'Décisions': ah['decisions'],
                             'Transactions': ah['transactions'],
                             'Dernier capital enregistré (USD)': mh['last_recorded_equity_usd'],
                             'Frais consignés (USD)': mh['fees_recorded_usd'],
                             'P&L réalisé consigné (USD)': mh['realized_pnl_recorded_usd'],
                             'Rendement / référence (%)': mh['return_vs_reference_pct'],
                             'Début historique': mh['market_start_utc'],
                             'Fin historique': mh['market_end_utc']})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                st.warning(f'Impossible de comparer {h}m : {exc}')
        if rows:
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
        st.caption('Attention : vérifier les périodes, les règles de sortie et les hypothèses de frais '
                   'avant toute comparaison de stratégie. Ne pas confondre rendement historique et prévision.')

    with tabs[5]:
        st.subheader('Audit des données de la même expérience')
        # Cast mixed values to string for Arrow / Streamlit compatibility.
        audit_frame = pd.DataFrame([{'Contrôle': name, 'Valeur': str(value)}
                                    for name, value in audit.items()])
        st.dataframe(audit_frame, width='stretch', hide_index=True)
        st.caption('Appariement par horodatage + action + rang d’occurrence pour EXECUTED seulement. '
                   'Un contrôle sans anomalie ne prouve ni la validité de toute la stratégie, '
                   'ni l’absence complète de data leakage.')

    with tabs[6]:
        st.subheader('Exports — résultats historiques en lecture seule')
        st.download_button('📥 Transactions CSV', data=trades.to_csv(index=False).encode('utf-8-sig'),
                           file_name=f'aurum_portfolio_{horizon}m_transactions.csv', mime='text/csv',
                           key=f'portfolio_export_trades_{horizon}')
        st.download_button('📥 Décisions et transactions rapprochées CSV',
                           data=events.to_csv(index=False).encode('utf-8-sig'),
                           file_name=f'aurum_portfolio_{horizon}m_decisions_reconciled.csv',
                           mime='text/csv', key=f'portfolio_export_reconciled_{horizon}')
        st.download_button('📥 Audit et métriques JSON',
                           data=report_json(horizon, metrics, audit),
                           file_name=f'aurum_portfolio_{horizon}m_audit.json',
                           mime='application/json', key=f'portfolio_export_json_{horizon}')
        st.caption('Téléchargements seulement : aucun fichier RAW n’est modifié.')


_render()
