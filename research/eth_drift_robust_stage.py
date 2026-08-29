#!/usr/bin/env python3
from pathlib import Path
import itertools, json
import numpy as np
import pandas as pd
from research import multi_asset_6way_kospi_trend_compare as kt
from research import btc_return_stress_6asset as stress
from research import btc_qqq_joint_robust as j
from research import btc_qqq_joint_robust_kospi_expand as ex

OUT=Path('results/eth_drift_robust_stage'); OUT.mkdir(parents=True,exist_ok=True)
KEY=j.KEY
ALPHAS=[1.0,0.5,0.25,0.0]
GOAL=-0.30
CURRENT=np.array([.10,.25,.30,.10,.20,.05])
REG_W=j.REG_W

def years(s,e): return (pd.Timestamp(e)-pd.Timestamp(s)).days/365.2425

def apply_drift(curve,alpha,edge,dates):
    c=np.asarray(curve,float)
    t=(pd.to_datetime(dates)-pd.Timestamp(dates[0])).days.to_numpy(float)/365.2425
    return c*np.exp(-(1-alpha)*edge*t)

def calibrate_eth(cache):
    z=[]
    for x in cache:
        if x['segment'] not in ('TRAIN','VALID'): continue
        y=years(x['start'],x['end'])
        z.append(np.log(x['curves'][-1,3]/x['tbill'][-1])/y)
    return float(np.median(z))

def eval_scenarios(cache,W,edge):
    sums=[]; oos={}
    for a in ALPHAS:
        tvc=[]; tvm=[]; ooc=[]; oom=[]
        for x in cache:
            c=x['curves'].copy(); c[:,3]=apply_drift(c[:,3],a,edge,x['dates'])
            _,_,fk,mk,_=stress.port_matrix_krw_fast(c,W,x['dates'],x['fx'])
            y=years(x['start'],x['end']); cg=np.power(np.maximum(fk,1e-15),1/y)-1
            if x['segment'] in ('TRAIN','VALID'): tvc.append(cg); tvm.append(mk)
            elif x['segment']=='OOS': ooc.append(cg); oom.append(mk)
        tvc=np.asarray(tvc); tvm=np.asarray(tvm); ooc=np.asarray(ooc); oom=np.asarray(oom)
        d=pd.DataFrame({k:W[:,i] for i,k in enumerate(KEY)})
        d['eth_alpha']=a; d['median_cagr']=np.median(tvc,0); d['p10_cagr']=np.quantile(tvc,.1,axis=0); d['worst_cagr']=np.min(tvc,0); d['worst_mdd']=np.min(tvm,0); sums.append(d)
        oos[a]=(ooc,oom)
    return pd.concat(sums,ignore_index=True),oos

def select(S,W):
    feasible=np.ones(len(W),bool); bench={}; zs={}
    for a in ALPHAS:
        z=S[np.isclose(S.eth_alpha,a)].reset_index(drop=True); zs[a]=z
        feasible &= z.worst_mdd.to_numpy()>=GOAL
        q=z[z.worst_mdd>=GOAL]; bench[a]={c:float(q[c].max()) for c in REG_W}
    ids=np.flatnonzero(feasible); maxreg=np.full(len(ids),-np.inf); avgreg=np.zeros(len(ids))
    for a,z in zs.items():
        comp=np.zeros(len(ids))
        for c,w in REG_W.items(): comp += w*(bench[a][c]-z[c].to_numpy()[ids])
        maxreg=np.maximum(maxreg,comp); avgreg+=comp/len(ALPHAS)
    order=np.lexsort((avgreg,maxreg)); ix=int(ids[order[0]])
    rec={'candidate_index':ix,'max_composite_regret':float(maxreg[order[0]]),'avg_composite_regret':float(avgreg[order[0]])}; rec.update({KEY[i]:float(W[ix,i]) for i in range(6)})
    return rec,bench

def summarize_oos(ix,W,oos):
    rows=[]
    for a,(c,m) in oos.items():
        cg=c[:,ix]; mm=m[:,ix]; rows.append({'eth_alpha':a,'median_cagr':float(np.median(cg)),'p10_cagr':float(np.quantile(cg,.1)),'worst_cagr':float(np.min(cg)),'median_mdd':float(np.median(mm)),'worst_mdd':float(np.min(mm)),**{KEY[i]:float(W[ix,i]) for i in range(6)}})
    return pd.DataFrame(rows)

def find_index(W,w):
    z=np.where(np.all(np.isclose(W,w[None,:]),axis=1))[0]
    return int(z[0]) if len(z) else None

def optima(S):
    rows=[]
    for a in ALPHAS:
        z=S[np.isclose(S.eth_alpha,a)&(S.worst_mdd>=GOAL)].copy()
        for c in REG_W: z['r_'+c]=z[c].rank(ascending=False,pct=True,method='average')
        z['score']=sum(REG_W[c]*z['r_'+c] for c in REG_W)
        r=z.sort_values(['score','worst_mdd'],ascending=[True,False]).iloc[0]
        rows.append({'eth_alpha':a,**{k:float(r[k]) for k in KEY},**{c:float(r[c]) for c in ['median_cagr','p10_cagr','worst_cagr','worst_mdd']}})
    return pd.DataFrame(rows)

def main():
    W=ex.expanded_weights(); d,sr,qr=kt.prepare(j.K_RULE); err=stress.validate_fast(d,W); cache=stress.make_cache(d); edge=calibrate_eth(cache)
    S,O=eval_scenarios(cache,W,edge); rec,bench=select(S,W); ix=int(rec['candidate_index']); OO=summarize_oos(ix,W,O); OPT=optima(S)
    ci=find_index(W,CURRENT); CUR=summarize_oos(ci,W,O) if ci is not None else pd.DataFrame()
    pd.DataFrame([rec]).to_csv(OUT/'robust_selected.csv',index=False); OO.to_csv(OUT/'robust_oos.csv',index=False); OPT.to_csv(OUT/'scenario_optima.csv',index=False); CUR.to_csv(OUT/'current_profile_oos.csv',index=False)
    meta={'eth_alphas':ALPHAS,'eth_annual_log_excess_edge_vs_tbill':edge,'goal':GOAL,'candidate_count':len(W),'spy_rule':sr,'qqq_rule':qr,'k_rule':j.K_RULE,'btc_rule':'MA150_C3','data_end':str(d.date.max().date()),'engine_validation':err}
    (OUT/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)); (OUT/'REPORT.md').write_text('# ETH drift robust stage\n\n```json\n'+json.dumps(meta,ensure_ascii=False,indent=2)+'\n```\n\n## Robust\n'+pd.DataFrame([rec]).to_markdown(index=False)+'\n\n## OOS\n'+OO.to_markdown(index=False)+'\n\n## Current V1 OOS\n'+CUR.to_markdown(index=False)+'\n\n## Scenario optima\n'+OPT.to_markdown(index=False))
    print('META',json.dumps(meta,ensure_ascii=False)); print('\nROBUST\n',pd.DataFrame([rec]).to_string(index=False)); print('\nROBUST OOS\n',OO.to_string(index=False)); print('\nCURRENT OOS\n',CUR.to_string(index=False)); print('\nOPTIMA\n',OPT.to_string(index=False))
if __name__=='__main__': main()
