#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.tqqq_panic_buy_study_v2 import adj_ohlc,synth_3x,prep,sim,buyhold,trend_tqqq,FEE
from research.tqqq_panic_buy_study_v3 import fastsim,arrays,bounds

OUT=Path('results/tqqq_panic_buy_v4_dotcom'); OUT.mkdir(parents=True,exist_ok=True)

def select(R,mdd_limit=-.30,regime=None):
    x=R[(R.TRAIN_trades>=5)&(R.VALID_trades>=2)&(R.TRAIN_mdd>=mdd_limit)&(R.VALID_mdd>=mdd_limit)].copy()
    if regime is not None:x=x[x.regime==regime]
    x['tv_floor']=x[['TRAIN_cagr','VALID_cagr']].min(axis=1); x['tv_avg']=x[['TRAIN_cagr','VALID_cagr']].mean(axis=1); x['tv_worst_mdd']=x[['TRAIN_mdd','VALID_mdd']].min(axis=1)
    x['score']=x.tv_floor+.25*x.tv_avg+.10*x.tv_worst_mdd
    return x.sort_values(['score','tv_floor'],ascending=False)

def exact_summary(p,r,periods,prefix):
    out=[]
    for name,(s,e) in periods.items():
        z=sim(p,int(r.lookback),float(r.trigger),float(r.step),int(r.ntr),float(r.tp),bool(r.regime),s,e)
        td=z['trade_df'].copy()
        if len(td):
            td['entry']=pd.to_datetime(td.entry); td['exit']=pd.to_datetime(td.exit); td['hold_days']=(td['exit']-td['entry']).dt.days
            medhold=float(td.hold_days.median()); maxhold=int(td.hold_days.max()); exposure=float(td.hold_days.sum()/max((pd.Timestamp(e) if e else p.index[-1])-pd.Timestamp(s),pd.Timedelta(days=1)).days)
        else:medhold=np.nan; maxhold=0; exposure=0.0
        out.append({'candidate':prefix,'period':name,**{k:z[k] for k in ['cagr','mdd','trades','win','worst_trade','avg_trade']},'median_hold_days':medhold,'max_hold_days':maxhold,'rough_time_in_market':exposure})
        td.to_csv(OUT/f'{prefix}_trades_{name}.csv',index=False)
    return out

def main():
    q=adj_ohlc('QQQ','1999-03-10'); syn=synth_3x(q); p=prep(syn,q)
    actual=adj_ohlc('TQQQ','2010-02-11'); pact=prep(actual,q)
    periods={'TRAIN':('2000-03-01','2015-12-31'),'VALID':('2016-01-01','2020-12-31'),'OOS':('2021-01-01',None)}
    pa={lb:arrays(p,lb) for lb in [60,120,252]}; pb={name:bounds(p.index,*se) for name,se in periods.items()}
    a=pa[60]; b=pb['TRAIN']; fastsim(*a,*b[:2],.15,.05,2,.2,True,FEE,b[2])
    rows=[]
    for regime in [True,False]:
      for lb in [60,120,252]:
       ar=pa[lb]
       for trig in [.10,.15,.20,.25,.30]:
        for step in [.05,.10]:
         for ntr in [1,2,3,4]:
          for tp in [.10,.15,.20,.30,.40]:
            rec={'regime':regime,'lookback':lb,'trigger':trig,'step':step,'ntr':ntr,'tp':tp}
            for name in periods:
                si,ei,yrs=pb[name]; z=fastsim(*ar,si,ei,trig,step,ntr,tp,regime,FEE,yrs)
                for k,v in zip(['cagr','mdd','trades','win','worst_trade','avg_trade'],z):rec[f'{name}_{k}']=v
            rows.append(rec)
    R=pd.DataFrame(rows); R.to_csv(OUT/'all_candidates.csv',index=False)
    selections={}
    for label,lim,rg in [('unrestricted',-.60,None),('mdd30',-.30,None),('regime_mdd30',-.30,True),('regime_mdd40',-.40,True),('regime_mdd50',-.50,True)]:
        x=select(R,lim,rg); selections[label]=x.iloc[0]; x.head(50).to_csv(OUT/f'top50_{label}.csv',index=False)
    summaries=[]; actual_summ=[]
    for label,r in selections.items():
        summaries += exact_summary(p,r,periods,label)
        z=sim(pact,int(r.lookback),float(r.trigger),float(r.step),int(r.ntr),float(r.tp),bool(r.regime),'2010-03-01',None)
        td=z['trade_df'].copy(); td['entry']=pd.to_datetime(td.entry); td['exit']=pd.to_datetime(td.exit); td['hold_days']=(td['exit']-td['entry']).dt.days if len(td) else []
        actual_summ.append({'candidate':label,**{k:z[k] for k in ['cagr','mdd','trades','win','worst_trade','avg_trade']},'median_hold_days':float(td.hold_days.median()) if len(td) else np.nan,'max_hold_days':int(td.hold_days.max()) if len(td) else 0})
    # Baselines and crisis slices
    base=[]
    for name,(s,e) in periods.items():
        for z in [buyhold(q,s,e,'QQQ_BH'),buyhold(syn,s,e,'SYN_TQQQ_BH'),trend_tqqq(p,s,e)]:base.append({'period':name,**z})
    crisis={'DOTCOM':('2000-03-01','2003-03-31'),'GFC':('2007-10-01','2009-06-30'),'COVID':('2020-01-01','2020-12-31'),'BEAR2022':('2022-01-01','2022-12-31')}
    crisis_rows=[]
    for label,r in selections.items():
        for cname,(s,e) in crisis.items():
            z=sim(p,int(r.lookback),float(r.trigger),float(r.step),int(r.ntr),float(r.tp),bool(r.regime),s,e)
            crisis_rows.append({'candidate':label,'crisis':cname,**{k:z[k] for k in ['cagr','mdd','trades','win','worst_trade']}})
    # current actual drawdown for each candidate
    cur=[]; ma=q.close.rolling(250).mean(); rg=bool((q.close>ma).tail(3).all())
    for label,r in selections.items():
        lb=int(r.lookback); dd=float(actual.close.iloc[-1]/actual.close.rolling(lb).max().iloc[-1]-1); levels=[-(float(r.trigger)+k*float(r.step)) for k in range(int(r.ntr))]
        cur.append({'candidate':label,'qqq_regime_on':rg,'actual_tqqq_dd':dd,'entry_levels':str(levels),'would_enter':bool((rg or not bool(r.regime)) and any(dd<=x for x in levels))})
    pd.DataFrame(summaries).to_csv(OUT/'selected_periods.csv',index=False); pd.DataFrame(actual_summ).to_csv(OUT/'actual_validation.csv',index=False); pd.DataFrame(base).to_csv(OUT/'baselines.csv',index=False); pd.DataFrame(crisis_rows).to_csv(OUT/'crisis_slices.csv',index=False); pd.DataFrame(cur).to_csv(OUT/'current_state.csv',index=False)
    seljson={k:{c:(bool(v) if isinstance(v,(bool,np.bool_)) else int(v) if isinstance(v,(int,np.integer)) else float(v) if isinstance(v,(float,np.floating)) else v) for c,v in r[['regime','lookback','trigger','step','ntr','tp','TRAIN_cagr','TRAIN_mdd','VALID_cagr','VALID_mdd','OOS_cagr','OOS_mdd']].items()} for k,r in selections.items()}
    meta={'data_start':str(q.index[0].date()),'data_end':str(q.index[-1].date()),'candidate_count':len(R),'selections':seljson,'current':cur,'fee_each_side':FEE}
    (OUT/'meta.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    print('META',json.dumps(meta,ensure_ascii=False)); print('\nSELECTED\n',pd.DataFrame(summaries).to_string(index=False)); print('\nACTUAL\n',pd.DataFrame(actual_summ).to_string(index=False)); print('\nCRISIS\n',pd.DataFrame(crisis_rows).to_string(index=False)); print('\nBASE\n',pd.DataFrame(base).to_string(index=False)); print('\nCURRENT\n',pd.DataFrame(cur).to_string(index=False))

if __name__=='__main__':main()
