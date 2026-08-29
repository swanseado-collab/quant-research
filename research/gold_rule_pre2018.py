#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
from research import multi_asset_5way_allocation as m

OUT=Path('results/gold_rule_pre2018'); OUT.mkdir(parents=True,exist_ok=True)
RULES=['BH']+[f'MA{w}_C{c}' for w in [100,150,200,250] for c in [1,3,5]]+['M10','M12']

def monthly_state(d,months):
    x=d[['date','close']].copy(); x['p']=x.date.dt.to_period('M')
    last=x.groupby('p',sort=True).tail(1).copy(); last['ma']=last.close.rolling(months,min_periods=months).mean(); last['sig']=(last.close>last.ma).astype(int)
    mp=dict(zip(last.date,last.sig)); st=np.zeros(len(d),int); cur=0
    for i,dt in enumerate(d.date):
        if dt in mp: cur=int(mp[dt])
        st[i]=cur
    return st

def rule_state(d,rule):
    if rule=='BH': return np.ones(len(d),int)
    if rule.startswith('MA'):
        a,b=rule.replace('MA','').split('_C'); w=int(a); c=int(b); ma=d.close.rolling(w,min_periods=w).mean(); return m.state_from(d.close,ma,c)
    return monthly_state(d,int(rule[1:]))

def end_idx(d,s,h):
    target=d.loc[s,'date']+pd.DateOffset(years=h); e=int(np.searchsorted(d.date.values,target.to_datetime64(),'right')-1)
    if e<=s or e>=len(d) or (target-d.loc[e,'date']).days>5 or d.loc[e,'date']>pd.Timestamp('2017-12-31'): return None
    return e

def summarize(g):
    return pd.Series({'cohorts':len(g),'median_cagr':g.cagr.median(),'p10_cagr':g.cagr.quantile(.1),'worst_cagr':g.cagr.min(),'median_mdd':g.mdd.median(),'worst_mdd':g.mdd.min(),'median_trades':g.trades.median()})

def main():
    d=m.add_rf(m.eqdata('GLD'),m.fred()); d=d[d.date<=pd.Timestamp('2017-12-31')].reset_index(drop=True)
    starts=d.groupby(d.date.dt.to_period('M')).head(1).index.tolist(); rows=[]
    states={r:rule_state(d,r) for r in RULES}
    for r,st in states.items():
      for h in [3,5]:
       for s in starts:
        dt=d.loc[s,'date']
        if dt<pd.Timestamp('2006-01-01') or s<260: continue
        e=end_idx(d,s,h)
        if e is None: continue
        sy=dt.year
        seg='TRAIN' if sy<=2008 else ('VALID' if sy<=2011 else 'PRE_OOS')
        eq,tr=m.sim_trend(d,st,s,e)
        rows.append({'rule':r,'horizon':h,'start':dt,'end':d.loc[e,'date'],'start_year':sy,'segment':seg,'cagr':m.cagr(eq[-1],dt,d.loc[e,'date']),'mdd':m.mdd(eq),'trades':tr})
    R=pd.DataFrame(rows); R.to_csv(OUT/'cohorts_monthly.csv',index=False)
    S=R.groupby(['rule','segment']).apply(summarize,include_groups=False).reset_index(); S.to_csv(OUT/'summary_by_segment.csv',index=False)
    # Select on TRAIN + VALID only. PRE_OOS is untouched until after selection.
    tv=R[R.segment.isin(['TRAIN','VALID'])]
    A=tv.groupby('rule').apply(summarize,include_groups=False).reset_index()
    for c in ['median_cagr','p10_cagr','worst_cagr','median_mdd','worst_mdd']:
        A['r_'+c]=A[c].rank(ascending=False,pct=True,method='average')
    A['score']=A[[c for c in A if c.startswith('r_')]].mean(1)
    A=A.sort_values(['score','median_trades'],ascending=[True,True]).reset_index(drop=True); A.to_csv(OUT/'rule_rank_trainvalid.csv',index=False)
    selected=str(A.iloc[0].rule)
    po=R[(R.segment=='PRE_OOS') & (R.rule==selected)]
    pre=summarize(po).to_dict() if len(po) else {}
    # Annual-start sensitivity, not used for selection.
    annual=R[R.start.dt.month.eq(1)].copy(); annual.to_csv(OUT/'annual_start_cohorts.csv',index=False)
    meta={'data_start':str(d.date.min().date()),'data_end':str(d.date.max().date()),'selected_rule':selected,'rules':RULES,'fee':m.FEE,'selection':'TRAIN start<=2008 + VALID 2009-2011; PRE_OOS starts>=2012 untouched','pre_oos_selected':pre}
    (OUT/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    (OUT/'REPORT.md').write_text('# Gold rule pre-2018 study\n\n```json\n'+json.dumps(meta,ensure_ascii=False,indent=2)+'\n```\n\n## TV rank\n'+A.to_markdown(index=False)+'\n\n## Segment summary\n'+S.to_markdown(index=False))
    print('META',json.dumps(meta,ensure_ascii=False)); print('\nRANK\n',A.to_string(index=False)); print('\nPRE_OOS SELECTED\n',po.groupby('horizon').apply(summarize,include_groups=False).reset_index().to_string(index=False) if len(po) else 'NONE')
if __name__=='__main__': main()
