#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
from research import multi_asset_5way_allocation as m
from research import multi_asset_6way_kospi_trend_compare as kt
from research import btc_return_stress_6asset as stress
from research import btc_qqq_joint_robust as j
from research import btc_qqq_joint_robust_kospi_expand as ex

OUT=Path('results/spy_rule_robust_stage'); OUT.mkdir(parents=True,exist_ok=True)
KEY=j.KEY; RULES=['MA150_C5','MA250_C5']; GOAL=-.30; CURRENT=np.array([.10,.25,.30,.10,.20,.05]); REG_W=j.REG_W

def override_spy(d,rule):
    x=d.copy(); tr=x[x.spy_trade_day.astype(bool)][['date','spy_close']].copy().reset_index(drop=True)
    a,b=rule.replace('MA','').split('_C'); w=int(a); c=int(b); ma=tr.spy_close.rolling(w,min_periods=w).mean(); tr['new_state']=m.state_from(tr.spy_close,ma,c)
    mp=dict(zip(tr.date,tr.new_state)); st=[]; cur=0
    for dt in x.date:
        if dt in mp: cur=int(mp[dt])
        st.append(cur)
    x['spy_state']=st
    return x

def years(s,e): return (pd.Timestamp(e)-pd.Timestamp(s)).days/365.2425

def evaluate(cache,W):
    tvc=[]; tvm=[]; ooc=[]; oom=[]
    for x in cache:
        _,_,fk,mk,_=stress.port_matrix_krw_fast(x['curves'],W,x['dates'],x['fx']); y=years(x['start'],x['end']); cg=np.power(np.maximum(fk,1e-15),1/y)-1
        if x['segment'] in ('TRAIN','VALID'): tvc.append(cg); tvm.append(mk)
        else: ooc.append(cg); oom.append(mk)
    return np.asarray(tvc),np.asarray(tvm),np.asarray(ooc),np.asarray(oom)

def select(S,W):
    feasible=np.ones(len(W),bool); bench={}
    for rule,z in S.items():
        feasible &= z['worst_mdd']>=GOAL; ok=z['worst_mdd']>=GOAL; bench[rule]={c:float(np.max(z[c][ok])) for c in REG_W}
    ids=np.flatnonzero(feasible); mx=np.full(len(ids),-np.inf); av=np.zeros(len(ids))
    for rule,z in S.items():
        comp=np.zeros(len(ids))
        for c,w in REG_W.items(): comp+=w*(bench[rule][c]-z[c][ids])
        mx=np.maximum(mx,comp); av+=comp/len(S)
    order=np.lexsort((av,mx)); ix=int(ids[order[0]]); rec={'candidate_index':ix,'max_composite_regret':float(mx[order[0]]),'avg_composite_regret':float(av[order[0]])}; rec.update({KEY[i]:float(W[ix,i]) for i in range(6)}); return rec

def stats(c,m,ix,W,rule,label):
    x=c[:,ix]; y=m[:,ix]; return {'profile':label,'spy_rule':rule,'median_cagr':float(np.median(x)),'p10_cagr':float(np.quantile(x,.1)),'worst_cagr':float(np.min(x)),'median_mdd':float(np.median(y)),'worst_mdd':float(np.min(y)),**{KEY[i]:float(W[ix,i]) for i in range(6)}}

def find_index(W,w):
    z=np.where(np.all(np.isclose(W,w[None,:]),axis=1))[0]; return int(z[0]) if len(z) else None

def main():
    W=ex.expanded_weights(); d0,sr_auto,qr=kt.prepare(j.K_RULE); allS={}; OO={}; validation={}
    for rule in RULES:
        d=override_spy(d0,rule); validation[rule]=stress.validate_fast(d,W); cache=stress.make_cache(d); tvc,tvm,ooc,oom=evaluate(cache,W)
        allS[rule]={'median_cagr':np.median(tvc,0),'p10_cagr':np.quantile(tvc,.1,axis=0),'worst_cagr':np.min(tvc,0),'worst_mdd':np.min(tvm,0)}; OO[rule]=(ooc,oom)
    rec=select(allS,W); ix=int(rec['candidate_index']); ci=find_index(W,CURRENT); rows=[]
    for rule,(c,mdd) in OO.items(): rows.append(stats(c,mdd,ix,W,rule,'robust')); rows.append(stats(c,mdd,ci,W,rule,'current_v1'))
    O=pd.DataFrame(rows)
    opt=[]
    for rule,z in allS.items():
        ok=z['worst_mdd']>=GOAL; ids=np.flatnonzero(ok); scores=np.zeros(len(ids))
        # percentile tail-aware score, lower is better
        for c,w in REG_W.items():
            vals=z[c][ids]; order=np.argsort(np.argsort(-vals)); ranks=(order+1)/len(ids); scores+=w*ranks
        q=ids[np.argmin(scores)]; opt.append({'spy_rule':rule,**{KEY[i]:float(W[q,i]) for i in range(6)},**{c:float(z[c][q]) for c in ['median_cagr','p10_cagr','worst_cagr','worst_mdd']}})
    OPT=pd.DataFrame(opt)
    pd.DataFrame([rec]).to_csv(OUT/'robust_selected.csv',index=False); O.to_csv(OUT/'oos_compare.csv',index=False); OPT.to_csv(OUT/'rule_specific_optima.csv',index=False)
    meta={'rules':RULES,'goal':GOAL,'candidate_count':len(W),'auto_selected_spy_rule':sr_auto,'qqq_rule':qr,'k_rule':j.K_RULE,'engine_validation':validation,'data_end':str(d0.date.max().date())}; (OUT/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)); (OUT/'REPORT.md').write_text('# SPY rule robustness stage\n\n```json\n'+json.dumps(meta,ensure_ascii=False,indent=2)+'\n```\n\n## Robust\n'+pd.DataFrame([rec]).to_markdown(index=False)+'\n\n## OOS\n'+O.to_markdown(index=False)+'\n\n## Rule-specific optima\n'+OPT.to_markdown(index=False))
    print('META',json.dumps(meta,ensure_ascii=False)); print('\nROBUST\n',pd.DataFrame([rec]).to_string(index=False)); print('\nOOS\n',O.to_string(index=False)); print('\nOPTIMA\n',OPT.to_string(index=False))
if __name__=='__main__': main()
