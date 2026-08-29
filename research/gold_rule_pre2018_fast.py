#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
from research import multi_asset_5way_allocation as m
from research import gold_rule_pre2018 as slow

OUT=Path('results/gold_rule_pre2018_fast'); OUT.mkdir(parents=True,exist_ok=True)
RULES=slow.RULES

def strategy_growth(d,st):
    n=len(d); g=np.ones(n,float); desired=np.zeros(n,int); desired[1:]=st[:-1]
    y=d.yk.to_numpy(float); op=d.open.to_numpy(float); cl=d.close.to_numpy(float); dates=d.date.to_numpy()
    for i in range(1,n):
        days=(pd.Timestamp(dates[i])-pd.Timestamp(dates[i-1])).days; rf=(1+max(y[i],0)/100.)**(days/365.2425)
        prev=desired[i-1]; cur=desired[i]
        if prev==0 and cur==0: g[i]=rf
        elif prev==0 and cur==1: g[i]=rf*cl[i]/(op[i]*(1+m.FEE))
        elif prev==1 and cur==1: g[i]=cl[i]/cl[i-1]
        else: g[i]=op[i]*(1-m.FEE)/cl[i-1]
    return g,desired

def cohort(d,g,desired,s,e):
    first=cl0=float(d.loc[s,'close'])/(float(d.loc[s,'open'])*(1+m.FEE)) if desired[s] else 1.0
    vals=np.empty(e-s+1,float); vals[0]=first
    if e>s: vals[1:]=first*np.cumprod(g[s+1:e+1])
    transitions=np.count_nonzero(desired[s+1:e+1]!=desired[s:e]); trades=int(bool(desired[s]))+int(transitions)
    return vals,trades

def summarize(g):
    return pd.Series({'cohorts':len(g),'median_cagr':g.cagr.median(),'p10_cagr':g.cagr.quantile(.1),'worst_cagr':g.cagr.min(),'median_mdd':g.mdd.median(),'worst_mdd':g.mdd.min(),'median_trades':g.trades.median()})

def main():
    d=m.add_rf(m.eqdata('GLD'),m.fred()); d=d[d.date<=pd.Timestamp('2017-12-31')].reset_index(drop=True)
    starts=d.groupby(d.date.dt.to_period('M')).head(1).index.tolist(); rows=[]
    for r in RULES:
      st=slow.rule_state(d,r); g,des=strategy_growth(d,st)
      for h in [3,5]:
       for s in starts:
        dt=d.loc[s,'date']
        if dt<pd.Timestamp('2006-01-01') or s<260: continue
        e=slow.end_idx(d,s,h)
        if e is None: continue
        sy=dt.year; seg='TRAIN' if sy<=2008 else ('VALID' if sy<=2011 else 'PRE_OOS')
        vals,tr=cohort(d,g,des,s,e); rows.append({'rule':r,'horizon':h,'start':dt,'end':d.loc[e,'date'],'start_year':sy,'segment':seg,'cagr':m.cagr(vals[-1],dt,d.loc[e,'date']),'mdd':m.mdd(vals),'trades':tr})
    R=pd.DataFrame(rows); R.to_csv(OUT/'cohorts_monthly.csv',index=False)
    tv=R[R.segment.isin(['TRAIN','VALID'])]; A=tv.groupby('rule').apply(summarize,include_groups=False).reset_index()
    for c in ['median_cagr','p10_cagr','worst_cagr','median_mdd','worst_mdd']: A['r_'+c]=A[c].rank(ascending=False,pct=True,method='average')
    A['score']=A[[c for c in A if c.startswith('r_')]].mean(1); A=A.sort_values(['score','median_trades']).reset_index(drop=True); A.to_csv(OUT/'rule_rank_trainvalid.csv',index=False)
    sel=str(A.iloc[0].rule); P=R[(R.segment=='PRE_OOS')&(R.rule==sel)]; ps=summarize(P).to_dict()
    # Validate fast engine against original sim on a sample of starts/rules.
    errs=[]
    for r in ['BH','MA100_C3','MA200_C3','M12']:
      st=slow.rule_state(d,r); g,des=strategy_growth(d,st)
      for s in starts[20:80:20]:
       e=slow.end_idx(d,s,3)
       if e is None: continue
       a,_=m.sim_trend(d,st,s,e); b,_=cohort(d,g,des,s,e); errs.append(float(np.max(np.abs(a-b))))
    err=max(errs) if errs else 0.0
    meta={'selected_rule':sel,'data_start':str(d.date.min().date()),'data_end':str(d.date.max().date()),'fast_vs_original_max_abs_error':err,'pre_oos_selected':ps,'selection':'TRAIN<=2008 + VALID 2009-2011 only'}
    (OUT/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)); (OUT/'REPORT.md').write_text('# Fast gold rule pre-2018\n\n```json\n'+json.dumps(meta,ensure_ascii=False,indent=2)+'\n```\n\n'+A.to_markdown(index=False))
    print('META',json.dumps(meta,ensure_ascii=False)); print('\nRANK\n',A.to_string(index=False)); print('\nPRE_OOS\n',P.groupby('horizon').apply(summarize,include_groups=False).reset_index().to_string(index=False))
if __name__=='__main__': main()
