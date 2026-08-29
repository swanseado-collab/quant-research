#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
from research import multi_asset_6way_kospi_trend_compare as kt
from research import btc_return_stress_6asset as stress
from research import btc_qqq_joint_robust as j
from research import btc_qqq_joint_robust_kospi_expand as ex

OUT=Path('results/fx_regime_robust_stage'); OUT.mkdir(parents=True,exist_ok=True)
KEY=j.KEY; GOAL=-.30; CURRENT=np.array([.10,.25,.30,.10,.20,.05]); REG_W=j.REG_W
REGIMES={'HIST':None,'KRW_STRONG':-0.02,'FLAT':0.0,'USD_STRONG':0.02}

def years(s,e): return (pd.Timestamp(e)-pd.Timestamp(s)).days/365.2425

def calibrate_fx(cache):
    z=[]
    for x in cache:
        if x['segment'] not in ('TRAIN','VALID'): continue
        y=years(x['start'],x['end']); z.append(np.log(x['fx'][-1]/x['fx'][0])/y)
    return float(np.median(z))

def fx_regime(fx,dates,hist_edge,target):
    f=np.asarray(fx,float)
    if target is None: return f.copy()
    t=(pd.to_datetime(dates)-pd.Timestamp(dates[0])).days.to_numpy(float)/365.2425
    return f*np.exp((target-hist_edge)*t)

def eval_regimes(cache,W,hist_edge):
    sums={}; oos={}
    for name,target in REGIMES.items():
        tvc=[]; tvm=[]; ooc=[]; oom=[]
        for x in cache:
            fx=fx_regime(x['fx'],x['dates'],hist_edge,target); _,_,fk,mk,_=stress.port_matrix_krw_fast(x['curves'],W,x['dates'],fx); y=years(x['start'],x['end']); cg=np.power(np.maximum(fk,1e-15),1/y)-1
            if x['segment'] in ('TRAIN','VALID'): tvc.append(cg); tvm.append(mk)
            else: ooc.append(cg); oom.append(mk)
        tvc=np.asarray(tvc); tvm=np.asarray(tvm); ooc=np.asarray(ooc); oom=np.asarray(oom)
        sums[name]={'median_cagr':np.median(tvc,0),'p10_cagr':np.quantile(tvc,.1,axis=0),'worst_cagr':np.min(tvc,0),'worst_mdd':np.min(tvm,0)}; oos[name]=(ooc,oom)
    return sums,oos

def select(S,W):
    feasible=np.ones(len(W),bool); bench={}
    for name,z in S.items():
        feasible &= z['worst_mdd']>=GOAL; ok=z['worst_mdd']>=GOAL; bench[name]={c:float(np.max(z[c][ok])) for c in REG_W}
    ids=np.flatnonzero(feasible); mx=np.full(len(ids),-np.inf); av=np.zeros(len(ids))
    for name,z in S.items():
        comp=np.zeros(len(ids))
        for c,w in REG_W.items(): comp+=w*(bench[name][c]-z[c][ids])
        mx=np.maximum(mx,comp); av+=comp/len(S)
    order=np.lexsort((av,mx)); ix=int(ids[order[0]]); rec={'candidate_index':ix,'max_composite_regret':float(mx[order[0]]),'avg_composite_regret':float(av[order[0]])}; rec.update({KEY[i]:float(W[ix,i]) for i in range(6)}); return rec

def find_index(W,w):
    z=np.where(np.all(np.isclose(W,w[None,:]),axis=1))[0]; return int(z[0]) if len(z) else None

def stats(c,m,ix,W,name,label):
    x=c[:,ix]; y=m[:,ix]; return {'profile':label,'fx_regime':name,'median_cagr':float(np.median(x)),'p10_cagr':float(np.quantile(x,.1)),'worst_cagr':float(np.min(x)),'median_mdd':float(np.median(y)),'worst_mdd':float(np.min(y)),**{KEY[i]:float(W[ix,i]) for i in range(6)}}

def main():
    W=ex.expanded_weights(); d,sr,qr=kt.prepare(j.K_RULE); err=stress.validate_fast(d,W); cache=stress.make_cache(d); edge=calibrate_fx(cache); S,O=eval_regimes(cache,W,edge); rec=select(S,W); ix=int(rec['candidate_index']); ci=find_index(W,CURRENT)
    rows=[]
    for name,(c,m) in O.items(): rows.append(stats(c,m,ix,W,name,'robust')); rows.append(stats(c,m,ci,W,name,'current_v1'))
    ODF=pd.DataFrame(rows); opts=[]
    for name,z in S.items():
        ok=z['worst_mdd']>=GOAL; ids=np.flatnonzero(ok); scores=np.zeros(len(ids))
        for c,w in REG_W.items():
            vals=z[c][ids]; order=np.argsort(np.argsort(-vals)); scores+=w*(order+1)/len(ids)
        q=ids[np.argmin(scores)]; opts.append({'fx_regime':name,**{KEY[i]:float(W[q,i]) for i in range(6)},**{c:float(z[c][q]) for c in ['median_cagr','p10_cagr','worst_cagr','worst_mdd']}})
    OPT=pd.DataFrame(opts)
    pd.DataFrame([rec]).to_csv(OUT/'robust_selected.csv',index=False); ODF.to_csv(OUT/'oos_compare.csv',index=False); OPT.to_csv(OUT/'regime_specific_optima.csv',index=False)
    meta={'regimes':REGIMES,'calibrated_tv_median_usdkrw_annual_log_drift':edge,'goal':GOAL,'candidate_count':len(W),'spy_rule':sr,'qqq_rule':qr,'k_rule':j.K_RULE,'engine_validation':err,'data_end':str(d.date.max().date())}; (OUT/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)); (OUT/'REPORT.md').write_text('# USDKRW regime robust stage\n\n```json\n'+json.dumps(meta,ensure_ascii=False,indent=2)+'\n```\n\n## Robust\n'+pd.DataFrame([rec]).to_markdown(index=False)+'\n\n## OOS\n'+ODF.to_markdown(index=False)+'\n\n## Regime optima\n'+OPT.to_markdown(index=False))
    print('META',json.dumps(meta,ensure_ascii=False)); print('\nROBUST\n',pd.DataFrame([rec]).to_string(index=False)); print('\nOOS\n',ODF.to_string(index=False)); print('\nOPTIMA\n',OPT.to_string(index=False))
if __name__=='__main__': main()
