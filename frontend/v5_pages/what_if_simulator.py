"""AURUM-AI V5 — What-If Simulator (Phase 12, historical research only).

Router: `python -m streamlit run frontend/app.py` then /what-if-simulator.
Does not modify any historical data or agent configuration.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.services.risk_management_audit import RUN_DIR, DECISIONS_FILE, TRADES_FILE, SUMMARY_FILE
from frontend.services.what_if_simulator_engine import (
    REFERENCE_NAME, WhatIfInputs, compare_scenarios, export_report,
    load_reference, simulate_fixed_trades, single_shock_mark_to_market,
)


@st.cache_data(show_spinner=False)
def _historical(root: str, *modifications: int):
    return load_reference(Path(root))


def _line(values: pd.DataFrame, col: str, label: str, color: str = '#3CDEED') -> go.Figure:
    f = go.Figure()
    f.add_scatter(x=values['trade_id'], y=values[col], mode='lines',
                  name=label, line={'width': 2.6, 'color':color},
                  hovertemplate='Trade %{x}<br>%{y:,.2f} USD<extra></extra>')
    f.update_layout(template='plotly_dark', paper_bgcolor='#061726', plot_bgcolor='#061726',
        height=380, margin={'l':12,'r':12,'t':22,'b':32}, showlegend=False,
        xaxis={'title':'Trade historique clôturé','gridcolor':'#173447'},
        yaxis={'title':'Capital après clôture (USD)','gridcolor':'#173447'})
    return f


def _hero():
    st.markdown('''<style>
    .aurum-whatif-hero{padding:24px 27px;background:linear-gradient(112deg,#11354c,#061426);
       border:1px solid #2384aa;border-radius:16px;margin:6px 0 12px}
    .aurum-whatif-hero h1{font-size:34px;color:#f1f7ff;margin:8px 0}
    .aurum-whatif-hero .eyebrow{font-size:11px;letter-spacing:2px;font-weight:800;color:#4be0f2}
    .aurum-whatif-hero p{color:#c1dce9;margin:0}
    </style>''', unsafe_allow_html=True)
    st.markdown('''<div class="aurum-whatif-hero">
      <div class="eyebrow">◇ AURUM-AI V5 · RESEARCH SCENARIO LAB</div>
      <h1>🎛️ What-If Simulator</h1>
      <p>Capital · Taille des opérations · Coûts supplémentaires · Comparaisons · Sensibilité au choc</p>
      </div>''', unsafe_allow_html=True)


def main():
    _hero()
    st.info('**Mode exploratoire en lecture seule — Validation historique.** Les hypothèses ajustent '
        'arithmétiquement les **90 opérations clôturées** de Consensus + Trend 60m. '
        'Les signaux, dates, prix et décisions sont figés : **aucun nouvel ordre, aucun réentraînement, '
        'aucune exécution réelle ou virtuelle**. Ce n’est pas un nouveau backtest.')
    paths = [ROOT / RUN_DIR / p for p in (DECISIONS_FILE, TRADES_FILE, SUMMARY_FILE)]
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.is_file()]
    if missing:
        st.error('Sources requises manquantes : ' + ', '.join(missing))
        st.stop()
    try:
        pairs, source, audit = _historical(str(ROOT), *(p.stat().st_mtime_ns for p in paths))
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        st.error(f'Chargement de référence refusé : {exc}')
        st.stop()

    reference_initial = float(source['initial_capital_usd'])
    st.caption('Source : `reports/backtesting/combined_holding_30m_60m/` · '
               f'**{REFERENCE_NAME}** · **Validation** · modèle fixe de sensibilité')
    a,b,c,d = st.columns(4)
    a.metric('Capital initial historique', f'{reference_initial:,.2f} USD')
    b.metric('Capital final historique', f'{float(source["final_equity_usd"]):,.2f} USD')
    c.metric('Rendement historique', f'{float(source["net_return_pct"]):+.3f} %')
    d.metric('Trades clôturés', str(len(pairs)))
    st.success(f'Audit de référence : {audit["matched"]}/{audit["transactions"]} transactions rapprochées. '
               'Les fichiers sources restent inchangés.')

    st.subheader('Configurer un scénario What-If')
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        capital = st.number_input('Capital initial hypothétique (USD)',min_value=1000.0,
            max_value=1000000.0,value=10000.0,step=1000.0,format='%.2f',key='wi_capital')
    with c2:
        size = st.slider('Intensité des tailles de trades', min_value=0.10,max_value=2.0,
             value=1.0,step=0.05,key='wi_size',help='1× = notionnels proportionnels au capital. '
             'Ne démontre pas que le RiskAgent approuverait ces ordres.')
    with c3:
        bps = st.slider('Surcoût (bps par côté)',min_value=0,max_value=100,value=0,
             step=5,key='wi_bps',help='S’ajoute aux frais historiques déjà inclus, et aux deux côtés des trades.')
    with c4:
        fixed = st.number_input('Frais fixes supplémentaires / trade (USD)',min_value=0.0,
            max_value=100.0,value=0.0,step=0.5,key='wi_fixed')

    inputs = WhatIfInputs(float(capital),float(size),float(bps),float(fixed))
    try:
        selected, metrics = simulate_fixed_trades(pairs,reference_initial,inputs)
        comparison, curves = compare_scenarios(pairs,reference_initial,{
            'Historique original':WhatIfInputs(reference_initial,1.,0.,0.),
            'Votre scénario':inputs,
            'Sans surcoûts':WhatIfInputs(float(capital),float(size),0.,0.),
            'Taille réduite 75 %':WhatIfInputs(float(capital),max(0.1,round(float(size)*.75,3)),float(bps),float(fixed)),
        })
    except (ValueError,TypeError,OverflowError) as exc:
        st.error('Paramètres incompatibles avec l’analyse : '+str(exc))
        st.stop()

    tabs = st.tabs(['🎛️ Résultat du scénario','📊 Comparaison','🧾 Contribution des trades',
                    '🛡️ Limites et prudence','🌪️ Choc indépendant','📥 Audit & Export'])

    with tabs[0]:
        st.subheader('Portefeuille hypothétique — opérations figées')
        p,q,r,s=st.columns(4)
        p.metric('Capital hypothétique final',f'{metrics["ending_capital_usd"]:,.2f} USD')
        q.metric('P&L hypothétique',f'{metrics["what_if_net_pnl_usd"]:+,.2f} USD')
        r.metric('Rendement hypothétique',f'{metrics["what_if_return_pct"]:+.3f} %')
        s.metric('Surcoûts ajoutés',f'{metrics["total_additional_cost_usd"]:,.2f} USD')
        st.plotly_chart(_line(selected,'equity_after_trade_usd','Capital What-If'),width='stretch')
        st.caption('La ligne représente uniquement le capital **après clôture** des trades historiques '
            '(pas les fluctuations entre trades ni le drawdown 5 minutes). Les frais du backtest de référence '
            'sont déjà compris dans le P&L source.')
        st.dataframe(selected[['trade_id','scaled_reference_pnl_usd','extra_variable_cost_usd',
          'extra_fixed_cost_usd','what_if_trade_pnl_usd','equity_after_trade_usd']].tail(15),
          hide_index=True,width='stretch',height=325)

    with tabs[1]:
        st.subheader('Comparer différentes hypothèses — mêmes opérations')
        st.dataframe(comparison,hide_index=True,width='stretch')
        fig=go.Figure()
        colors=['#3cdeed','#efbc68','#77dba9','#b9a1fb']
        for i,(name,frame) in enumerate(curves.items()):
            # normalized to % of each scenario's own starting capital
            base=float(comparison.loc[comparison['scenario']==name,'capital_initial_usd'].iloc[0])
            fig.add_scatter(x=frame['trade_id'],y=(frame['equity_after_trade_usd']/base-1)*100,
                 mode='lines',name=name,line={'color':colors[i % len(colors)],'width':2.4})
        fig.update_layout(template='plotly_dark',paper_bgcolor='#061726',plot_bgcolor='#061726',
           height=390,margin={'l':10,'r':15,'t':25,'b':35},
           xaxis_title='Numéro de trade clôturé',yaxis_title='Rendement cumulé aux clôtures (%)',
           legend={'orientation':'h','y':1.1})
        st.plotly_chart(fig,width='stretch')
        st.warning('Modifier le capital ou multiplier la taille des trades peut réduire ou amplifier les pertes '
                   'sans améliorer la qualité prédictive. Aucun RiskAgent n’a vérifié ces nouvelles tailles.')

    with tabs[2]:
        st.subheader('Décomposition des résultats contrefactuels')
        c1,c2,c3=st.columns(3)
        c1.metric('Frais historiques mis à l’échelle',f'{metrics["total_scaled_recorded_fees_usd"]:,.2f} USD')
        c2.metric('P&L historique mis à l’échelle',f'{metrics["base_scaled_net_pnl_usd"]:+,.2f} USD')
        c3.metric('Trades hypothétiquement perdants',str(metrics['losing_trades']))
        st.dataframe(selected[['trade_id','entry_time_utc','exit_time_utc',
             'scaled_buy_notional_usd','scaled_sell_notional_usd',
             'scaled_reference_pnl_usd','total_extra_cost_usd','what_if_trade_pnl_usd']],
             width='stretch',hide_index=True,height=390)
        st.caption('Tous les prix et horodatages de ces opérations sont historiques ; seuls notionnels, '
                   'P&L et surcoûts sont transformés par les hypothèses.')

    with tabs[3]:
        st.subheader('Contrôles indicatifs — pas une approbation RiskAgent')
        c1,c2,c3=st.columns(3)
        c1.metric('Max DD aux clôtures',f'{metrics["max_trade_close_drawdown_pct"]:.3f} %')
        c2.metric('Ordre BUY / capital précédent (proxy max)',f'{metrics["max_proxy_buy_fraction_pct"]:.3f} %')
        c3.metric('Proxys BUY au-dessus de 5 %',str(metrics['proxy_trades_above_5pct']))
        if metrics['proxy_trades_above_5pct']:
            st.warning('Certains notionnels BUY dépassent 5 % du dernier capital observé **à la clôture précédente**. '
                      'C’est un signal exploratoire, pas une preuve de rejet réel par le Risk Agent.')
        else:
            st.info('Aucun dépassement du proxy 5 % identifié aux clôtures. Cela ne valide pas toutes '
                    'les règles d’exposition ou de liquidité du Risk Agent.')
        if metrics['trade_close_dd_above_10pct']:
            st.warning('Le drawdown du scénario aux clôtures atteint au moins 10 %. '
                       'Un nouveau backtest dynamique serait nécessaire pour connaître les décisions du Kill Switch.')
        st.error('**Non évaluables ici :** perte journalière à 2 %, exposition intratrade à 25 %, '
                 'validité des nouveaux ordres, stop-loss, état du Kill Switch et risque réel. '
                 'Le moteur n’a pas les snapshots nécessaires pour reconstituer ces décisions.')
        st.dataframe(selected[['trade_id','proxy_buy_fraction_pct','trade_close_drawdown_pct',
                               'equity_before_trade_usd','equity_after_trade_usd']].tail(20),
                     width='stretch',hide_index=True)

    with tabs[4]:
        st.subheader('Choc de marché sur une position XOM fictive — expérience distincte')
        a,b = st.columns(2)
        with a:
            shock = st.slider('Variation du prix XOM fictive (%)',min_value=-95,max_value=50,
                             value=-15,step=5,key='wi_shock')
        with b:
            exposure = st.slider('Exposition initiale hypothétique (%)',min_value=0,max_value=25,
                                 value=25,step=5,key='wi_exposure')
        shock_result=single_shock_mark_to_market(float(capital),float(exposure),float(shock))
        x,y,z=st.columns(3)
        x.metric('Capital après le choc fictif',f'{shock_result["hypothetical_equity_usd"]:,.2f} USD')
        y.metric('P&L mark-to-market fictif',f'{shock_result["hypothetical_pnl_usd"]:+,.2f} USD')
        z.metric('Perte depuis le départ',f'{shock_result["hypothetical_drawdown_pct"]:.2f} %')
        if shock_result['reaches_10pct_drawdown_threshold']:
            st.warning('Cette baisse synthétique atteint numériquement 10 % de drawdown ; '
                       'l’état réel du Kill Switch n’a PAS été simulé ici.')
        st.info('Cette expérience est **indépendante des 90 trades historiques**. '
                'Une seule variation synthétique de prix est appliquée à une position longue fixe, '
                'sans ordre, glissement, rebalancement, prédiction ni agent.')
        st.dataframe(pd.DataFrame([{'Paramètre':str(k),'Résultat':str(v)} for k,v in shock_result.items()]),
                     hide_index=True,width='stretch')

    with tabs[5]:
        st.subheader('Traçabilité et exports de recherche')
        st.dataframe(pd.DataFrame([{'Contrôle':str(k),'Anomalies':int(v)}
                                   for k,v in audit['checks'].items()]),
                     hide_index=True,width='stretch')
        st.caption('Sources : '+', '.join(str(p.relative_to(ROOT)) for p in paths))
        st.download_button('📥 Scénario modifié (CSV)',
                           selected.to_csv(index=False).encode('utf-8-sig'),
                           file_name='aurum_whatif_custom_trades_validation.csv',mime='text/csv')
        st.download_button('📥 Comparaison des scénarios (CSV)',
                           comparison.to_csv(index=False).encode('utf-8-sig'),
                           file_name='aurum_whatif_comparison_validation.csv',mime='text/csv')
        st.download_button('📥 Rapport What-If (JSON)',
                           export_report(inputs,metrics,comparison,source,shock_result).encode('utf-8'),
                           file_name='aurum_whatif_report_validation.json',mime='application/json')
        st.warning('Résultats exploratoires sur Validation : pas de séparation Test indépendante, '
                   'pas de nouveaux signaux ni validation dynamique du Risk Agent. Le drawdown '
                   'historique du résumé est calculé sur 5 minutes et **ne doit pas** être confondu '
                   'avec le drawdown sur les seules clôtures affichées ici.')

main()
