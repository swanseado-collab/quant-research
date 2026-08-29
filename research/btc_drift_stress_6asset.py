#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research import btc_return_stress_6asset as s
from research import btc_return_stress_6asset_v2 as fast
from research import multi_asset_6way_kospi_trend_compare as kt
from research import multi_asset_5way_allocation as m

OUT=Path('results/btc_drift_stress_6asset'); OUT.mkdir(parents=True,exist_ok=True)
ALPHAS=s.ALPHAS; KEY=s.KEY; GOALS=s.GOALS; K_RULE=s.K_RULE
s.port_matrix_krw_fast=fast.port_matrix_krw_fast


def calibrate_edge(cache):
    vals=[]
    for x in cache:
        if x['segment'] not in ['TRAIN','VALID']: continue
        years=(x['end']-x['start']).days/365.2425
        btc=float(x['curves'][-1,2]); tb=float(x['tbill'][-1])
        vals.append(np.log(max(btc,1e-15)/max(tb,1e-15))/years)
    return float(np.median(vals)),np.asarray(vals,float)


def drift_stress_curve(btc_curve,alpha,annual_log_excess_edge):
    """Subtract a constant annual drift while preserving historical daily shocks and drawdowns."""
    b=np.asarray(btc_curve,float); t=np.arange(len(b),dtype=float)/365.2425
    drag=(1.0-alpha)*annual_log_excess_edge
    return b*np.exp(-drag*t)


def eval_all(cache,W,edge):
    rows=[]; btcrows=[]
    for item in cache:
      basecur=item['curves']; sdt=item['start']; edt=item['end']
      for alpha in ALPHAS:
        curves=basecur.copy(); curves[:,2]=drift_stress_curve(basecur[:,2],alpha,edge)
        fu,mu,fk,mk,to=fast.port_matrix_krw_fast(curves,W,item['dates'],item['fx'])
        b=curves[:,2]
        btcrows.append({'alpha':alpha,'horizon':item['horizon'],'start':sdt,'end':edt,'start_year':item['start_year'],'segment':item['segment'],'cagr':m.cagr(float(b[-1]),sdt,edt),'mdd':m.mdd(b)})
        years=(edt-sdt).days/365.2425
        cagr_usd=np.power(np.maximum(fu,1e-15),1/years)-1; cagr_krw=np.power(np.maximum(fk,1e-15),1/years)-1
        for j,w in enumerate(W):
          rec={'alpha':alpha,'horizon':item['horizon'],'start':sdt,'end':edt,'start_year':item['start_year'],'segment':item['segment'],'cagr_usd':cagr_usd[j],'mdd_usd':mu[j],'cagr_krw':cagr_krw[j],'mdd_krw':mk[j],'turnover':to[j]}
          rec.update({KEY[i]:w[i] for i in range(6)}); rows.append(rec)
    return pd.DataFrame(rows),pd.DataFrame(btcrows)


def edge_effect(B):
    return B.groupby(['alpha','segment','horizon']).agg(cohorts=('cagr','size'),median_cagr=('cagr','median'),p10_cagr=('cagr',lambda x:x.quantile(.1)),worst_cagr=('cagr','min'),median_mdd=('mdd','median'),worst_mdd=('mdd','min')).reset_index()


def main():
    W=s.candidate_weights(); pd.DataFrame(W,columns=['SPY','QQQ','BTC','ETH','KOSPI200','TBILL']).to_csv(OUT/'candidate_weights.csv',index=False)
    d,sr,qr=kt.prepare(K_RULE); err=s.validate_fast(d,W); cache=s.make_cache(d); edge,edge_samples=calibrate_edge(cache)
    R,B=eval_all(cache,W,edge); R.to_csv(OUT/'cohorts.csv',index=False); B.to_csv(OUT/'btc_sleeve_cohorts.csv',index=False)
    A,SEL=s.select_profiles(R); A.to_csv(OUT/'trainvalid_summary.csv',index=False); SEL.to_csv(OUT/'selected_trainvalid.csv',index=False)
    O=s.oos_selected(R,SEL); O.to_csv(OUT/'selected_oos.csv',index=False)
    F=s.fixed_profiles(R); F.to_csv(OUT/'fixed_profile_stress.csv',index=False)
    E=edge_effect(B); E.to_csv(OUT/'btc_stress_effect.csv',index=False)
    meta={'stress':'Calibrated drift haircut: retain all historical BTC trend-sleeve shocks; subtract (1-alpha) times TRAIN+VALID median annual log excess return over T-bill.','annual_log_excess_edge':edge,'equivalent_excess_geometric_rate':float(np.exp(edge)-1),'alphas':ALPHAS,'goals':GOALS,'k_rule':K_RULE,'spy_rule':sr,'qqq_rule':qr,'btc_rule':'MA150_C3','eth_rule':'MA200_C1 + 40/5x12','fee':s.FEE,'candidate_count':len(W),'fast_engine_validation_max_abs_error':err,'cohorts':len(cache),'data_end':str(d.date.max().date()),'tv_edge_sample_n':len(edge_samples)}
    (OUT/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    report='# BTC calibrated drift-stress study\n\n```json\n'+json.dumps(meta,ensure_ascii=False,indent=2)+'\n```\n\n## Train+Validation selections\n'+SEL.to_markdown(index=False)+'\n\n## OOS\n'+O.to_markdown(index=False)+'\n\n## Fixed profiles\n'+F.to_markdown(index=False)+'\n\n## BTC sleeve stress effect\n'+E.to_markdown(index=False)
    (OUT/'REPORT.md').write_text(report)
    print('META',json.dumps(meta,ensure_ascii=False)); print('\nSELECTED TV\n',SEL.to_string(index=False)); print('\nSELECTED OOS\n',O.to_string(index=False)); print('\nFIXED OOS\n',F[F.segment=='OOS'].to_string(index=False)); print('\nBTC EFFECT\n',E.to_string(index=False))

if __name__=='__main__': main()
