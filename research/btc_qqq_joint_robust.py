#!/usr/bin/env python3
from __future__ import annotations
import itertools, json
from pathlib import Path
import numpy as np
import pandas as pd
from research import multi_asset_5way_allocation as m
from research import multi_asset_6way_kospi_trend_compare as kt
from research import btc_return_stress_6asset_v2 as eng

OUT=Path('results/btc_qqq_joint_robust'); OUT.mkdir(parents=True,exist_ok=True)
KEY=['spy','qqq','btc','eth','kospi200','tbill']
K_RULE='MA100_C3'
BTC_ALPHAS=[1.0,0.5,0.25]
QQQ_ALPHAS=[1.0,0.5,0.0]
GOALS=[-0.30,-0.35,-0.40]
REG_W={'median_cagr':0.5,'p10_cagr':0.3,'worst_cagr':0.2}


def years(s,e): return (pd.Timestamp(e)-pd.Timestamp(s)).days/365.2425

def apply_drift(curve, benchmark, alpha, annual_log_edge, dates):
    c=np.asarray(curve,float); b=np.asarray(benchmark,float)
    t=(pd.to_datetime(dates)-pd.Timestamp(dates[0])).days.to_numpy(float)/365.2425
    # Preserve every historical shock; only reduce calibrated long-run log excess drift.
    return c*np.exp(-(1.0-alpha)*annual_log_edge*t)

def calibrate_edges(cache):
    btc=[]; qqq=[]
    for x in cache:
        if x['segment'] not in ('TRAIN','VALID'): continue
        y=years(x['start'],x['end']); c=x['curves']
        btc.append(np.log(c[-1,2]/x['tbill'][-1])/y)
        qqq.append(np.log(c[-1,1]/c[-1,0])/y)
    return float(np.median(btc)),float(np.median(qqq))

def summarize_arrays(cagr,mdd):
    return {
      'cohorts':cagr.shape[0],
      'median_cagr':np.median(cagr,axis=0),
      'p10_cagr':np.quantile(cagr,.1,axis=0),
      'worst_cagr':np.min(cagr,axis=0),
      'median_mdd':np.median(mdd,axis=0),
      'worst_mdd':np.min(mdd,axis=0),
    }

def evaluate(cache,W,be,qe):
    tv_rows=[]; oos_store={}; summaries=[]
    for ba,qa in itertools.product(BTC_ALPHAS,QQQ_ALPHAS):
        tv_c=[]; tv_m=[]; oo_c=[]; oo_m=[]
        for x in cache:
            c=x['curves'].copy()
            c[:,2]=apply_drift(c[:,2],x['tbill'],ba,be,x['dates'])
            c[:,1]=apply_drift(c[:,1],c[:,0],qa,qe,x['dates'])
            _,_,fk,mk,_=eng.port_matrix_krw_fast(c,W,x['dates'],x['fx'])
            y=years(x['start'],x['end']); cg=np.power(np.maximum(fk,1e-15),1/y)-1
            if x['segment'] in ('TRAIN','VALID'):
                tv_c.append(cg); tv_m.append(mk)
            elif x['segment']=='OOS':
                oo_c.append(cg); oo_m.append(mk)
        tv_c=np.asarray(tv_c); tv_m=np.asarray(tv_m); oo_c=np.asarray(oo_c); oo_m=np.asarray(oo_m)
        s=summarize_arrays(tv_c,tv_m)
        df=pd.DataFrame({k:W[:,i] for i,k in enumerate(KEY)})
        for k,v in s.items(): df[k]=v if np.ndim(v) else v
        df['btc_alpha']=ba; df['qqq_alpha']=qa; summaries.append(df)
        oos_store[(ba,qa)]=(oo_c,oo_m)
    return pd.concat(summaries,ignore_index=True),oos_store

def select_robust(S,W):
    scenarios=list(itertools.product(BTC_ALPHAS,QQQ_ALPHAS)); selections=[]; regret_rows=[]
    # map weight tuple -> row index; S is generated in W order for every scenario
    for goal in GOALS:
        # Candidate must respect the MDD budget in every model world.
        mats=[]; benchmarks={}
        feasible=np.ones(len(W),dtype=bool)
        for ba,qa in scenarios:
            z=S[(S.btc_alpha==ba)&(S.qqq_alpha==qa)].reset_index(drop=True)
            feasible &= z.worst_mdd.to_numpy()>=goal
            q=z[z.worst_mdd>=goal]
            benchmarks[(ba,qa)]={c:float(q[c].max()) for c in REG_W}
            mats.append(((ba,qa),z))
        ids=np.flatnonzero(feasible)
        if len(ids)==0: continue
        maxreg=np.full(len(ids),-np.inf); avgreg=np.zeros(len(ids)); maxmed=np.full(len(ids),-np.inf)
        per=[]
        for (ba,qa),z in mats:
            comp=np.zeros(len(ids)); medreg=benchmarks[(ba,qa)]['median_cagr']-z.median_cagr.to_numpy()[ids]
            for c,w in REG_W.items(): comp += w*(benchmarks[(ba,qa)][c]-z[c].to_numpy()[ids])
            maxreg=np.maximum(maxreg,comp); avgreg+=comp/len(scenarios); maxmed=np.maximum(maxmed,medreg)
        order=np.lexsort((avgreg,maxreg)); ix=ids[order[0]]; w=W[ix]
        rec={'goal_mdd':goal,'candidate_index':int(ix),'max_composite_regret':float(maxreg[order[0]]),'avg_composite_regret':float(avgreg[order[0]]),'max_median_cagr_regret':float(maxmed[order[0]])}
        rec.update({KEY[i]:float(w[i]) for i in range(6)}); selections.append(rec)
        for (ba,qa),z in mats:
            r=z.iloc[ix]
            rr={'goal_mdd':goal,'btc_alpha':ba,'qqq_alpha':qa,'candidate_index':int(ix)}
            rr.update({KEY[i]:float(w[i]) for i in range(6)})
            for c in ['median_cagr','p10_cagr','worst_cagr','worst_mdd']:
                rr['tv_'+c]=float(r[c]); rr['benchmark_'+c]=benchmarks[(ba,qa)].get(c,np.nan)
            regret_rows.append(rr)
    return pd.DataFrame(selections),pd.DataFrame(regret_rows)

def oos_selected(sel,oos_store,W):
    rows=[]
    for _,q in sel.iterrows():
        ix=int(q.candidate_index)
        for ba,qa in itertools.product(BTC_ALPHAS,QQQ_ALPHAS):
            c,md=oos_store[(ba,qa)]; cg=c[:,ix]; mm=md[:,ix]
            rec={'goal_mdd':q.goal_mdd,'btc_alpha':ba,'qqq_alpha':qa,'cohorts':len(cg),'median_cagr':float(np.median(cg)),'p10_cagr':float(np.quantile(cg,.1)),'worst_cagr':float(np.min(cg)),'median_mdd':float(np.median(mm)),'worst_mdd':float(np.min(mm))}
            for k in KEY: rec[k]=float(q[k])
            rows.append(rec)
    return pd.DataFrame(rows)

def scenario_optima(S):
    rows=[]
    for goal in GOALS:
      for ba,qa in itertools.product(BTC_ALPHAS,QQQ_ALPHAS):
        z=S[(S.btc_alpha==ba)&(S.qqq_alpha==qa)&(S.worst_mdd>=goal)].copy()
        if z.empty: continue
        # same tail-aware score as prior research
        for c in REG_W: z['r_'+c]=z[c].rank(ascending=False,pct=True,method='average')
        z['score']=sum(REG_W[c]*z['r_'+c] for c in REG_W)
        r=z.sort_values(['score','worst_mdd'],ascending=[True,False]).iloc[0]
        rec={'goal_mdd':goal,'btc_alpha':ba,'qqq_alpha':qa,'score':float(r.score)}
        for k in KEY: rec[k]=float(r[k])
        for c in ['median_cagr','p10_cagr','worst_cagr','worst_mdd']: rec[c]=float(r[c])
        rows.append(rec)
    return pd.DataFrame(rows)

def main():
    W=eng.candidate_weights(); d,sr,qr=kt.prepare(K_RULE); err=eng.validate_fast(d,W); cache=eng.make_cache(d)
    be,qe=calibrate_edges(cache)
    S,Ostore=evaluate(cache,W,be,qe); S.to_csv(OUT/'scenario_trainvalid_summary.csv',index=False)
    SEL,REG=select_robust(S,W); SEL.to_csv(OUT/'robust_selected.csv',index=False); REG.to_csv(OUT/'robust_trainvalid_by_scenario.csv',index=False)
    OO=oos_selected(SEL,Ostore,W); OO.to_csv(OUT/'robust_oos_by_scenario.csv',index=False)
    OPT=scenario_optima(S); OPT.to_csv(OUT/'scenario_specific_optima.csv',index=False)
    meta={'btc_alphas':BTC_ALPHAS,'qqq_alphas':QQQ_ALPHAS,'goals':GOALS,'regret_weights':REG_W,'btc_annual_log_excess_edge_vs_tbill':be,'qqq_annual_log_excess_edge_vs_spy':qe,'k_rule':K_RULE,'spy_rule':sr,'qqq_rule':qr,'btc_rule':'MA150_C3','candidate_count':len(W),'cohorts':len(cache),'data_end':str(d.date.max().date()),'engine_validation':err}
    (OUT/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    report='# Joint BTC + QQQ robust allocation\n\n```json\n'+json.dumps(meta,ensure_ascii=False,indent=2)+'\n```\n\n## Robust selections\n'+SEL.to_markdown(index=False)+'\n\n## OOS by scenario\n'+OO.to_markdown(index=False)+'\n\n## Scenario optima\n'+OPT.to_markdown(index=False)
    (OUT/'REPORT.md').write_text(report)
    print('META',json.dumps(meta,ensure_ascii=False)); print('\nROBUST\n',SEL.to_string(index=False)); print('\nOOS\n',OO.to_string(index=False)); print('\nOPTIMA\n',OPT.to_string(index=False))
if __name__=='__main__': main()
