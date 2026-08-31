#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.tqqq_panic_buy_study_v2 import adj_ohlc,synth_3x,prep,sim,FEE
from research.tqqq_panic_buy_study_v3 import fastsim,arrays,bounds

OUT=Path('results/tqqq_panic_buy_v5_actual'); OUT.mkdir(parents=True,exist_ok=True)
KEYS=['regime','lookback','trigger','step','ntr','tp']

def score_df(x):
    x=x.copy(); x['tv_floor']=x[['A_TRAIN_cagr','A_VALID_cagr']].min(axis=1); x['tv_avg']=x[['A_TRAIN_cagr','A_VALID_cagr']].mean(axis=1); x['tv_worst_mdd']=x[['A_TRAIN_mdd','A_VALID_mdd']].min(axis=1)
    x['score']=x.tv_floor+.25*x.tv_avg+.10*x.tv_worst_mdd
    return x.sort_values(['score','tv_floor'],ascending=False)

def exact_actual(p,r,label,periods):
    rows=[]
    for name,(s,e) in periods.items():
        z=sim(p,int(r.lookback),float(r.trigger),float(r.step),int(r.ntr),float(r.tp),bool(r.regime),s,e); td=z['trade_df'].copy()
        if len(td):
            td['entry']=pd.to_datetime(td.entry);td['exit']=pd.to_datetime(td.exit);td['hold_days']=(td['exit']-td['entry']).dt.days
            mh=float(td.hold_days.median());mx=int(td.hold_days.max()); exposure=float(td.hold_days.sum()/max(((pd.Timestamp(e) if e else p.index[-1])-pd.Timestamp(s)).days,1))
        else:mh=np.nan;mx=0;exposure=0
        rows.append({'candidate':label,'period':name,**{k:z[k] for k in ['cagr','mdd','trades','win','worst_trade','avg_trade']},'median_hold_days':mh,'max_hold_days':mx,'rough_time_in_market':exposure})
        td.to_csv(OUT/f'{label}_{name}_trades.csv',index=False)
    return rows

def main():
    q=adj_ohlc('QQQ','1999-03-10'); syn=synth_3x(q); ps=prep(syn,q)
    actual=adj_ohlc('TQQQ','2010-02-11'); pa=prep(actual,q)
    aper={'TRAIN':('2010-03-01','2015-12-31'),'VALID':('2016-01-01','2020-12-31'),'OOS':('2021-01-01',None)}
    stress={'PRE2010':('2000-03-01','2009-12-31'),'DOTCOM':('2000-03-01','2003-03-31'),'GFC':('2007-10-01','2009-06-30')}
    aa={lb:arrays(pa,lb) for lb in [60,120,252]}; ab={n:bounds(pa.index,*se) for n,se in aper.items()}
    sa={lb:arrays(ps,lb) for lb in [60,120,252]}; sb={n:bounds(ps.index,*se) for n,se in stress.items()}
    a=aa[60];b=ab['TRAIN'];fastsim(*a,*b[:2],.2,.1,2,.15,True,FEE,b[2])
    rows=[]
    for regime in [True,False]:
      for lb in [60,120,252]:
       for trig in [.10,.15,.20,.25,.30]:
        for step in [.05,.10]:
         for ntr in [1,2,3,4]:
          for tp in [.10,.15,.20,.30,.40]:
            rec={'regime':regime,'lookback':lb,'trigger':trig,'step':step,'ntr':ntr,'tp':tp}
            for name in aper:
                si,ei,yrs=ab[name]; z=fastsim(*aa[lb],si,ei,trig,step,ntr,tp,regime,FEE,yrs)
                for k,v in zip(['cagr','mdd','trades','win','worst_trade','avg_trade'],z):rec[f'A_{name}_{k}']=v
            for name in stress:
                si,ei,yrs=sb[name]; z=fastsim(*sa[lb],si,ei,trig,step,ntr,tp,regime,FEE,yrs)
                for k,v in zip(['cagr','mdd','trades','win','worst_trade','avg_trade'],z):rec[f'S_{name}_{k}']=v
            rows.append(rec)
    R=pd.DataFrame(rows);R.to_csv(OUT/'all_candidates.csv',index=False)
    base=(R.A_TRAIN_trades>=3)&(R.A_VALID_trades>=2)
    specs={
      'actual_unrestricted': base & (R.A_TRAIN_mdd>=-.60)&(R.A_VALID_mdd>=-.60),
      'robust30': base & (R.A_TRAIN_mdd>=-.30)&(R.A_VALID_mdd>=-.30)&(R.S_PRE2010_mdd>=-.30),
      'robust40': base & (R.A_TRAIN_mdd>=-.40)&(R.A_VALID_mdd>=-.40)&(R.S_PRE2010_mdd>=-.40),
      'regime_robust30': base & R.regime & (R.A_TRAIN_mdd>=-.30)&(R.A_VALID_mdd>=-.30)&(R.S_PRE2010_mdd>=-.30),
      'regime_robust40': base & R.regime & (R.A_TRAIN_mdd>=-.40)&(R.A_VALID_mdd>=-.40)&(R.S_PRE2010_mdd>=-.40),
    }
    selections={};tops=[]
    for label,mask in specs.items():
        x=score_df(R[mask]); selections[label]=x.iloc[0]; y=x.head(50).copy();y['selection']=label;tops.append(y)
    pd.concat(tops).to_csv(OUT/'top50_by_selection.csv',index=False)
    details=[]
    for label,r in selections.items():details+=exact_actual(pa,r,label,aper)
    pd.DataFrame(details).to_csv(OUT/'selected_actual_periods.csv',index=False)
    # exact synthetic stress and current state
    stresses=[];cur=[]; qma=q.close.rolling(250).mean(); qreg=bool((q.close>qma).tail(3).all())
    for label,r in selections.items():
        for sn,(s,e) in stress.items():
            z=sim(ps,int(r.lookback),float(r.trigger),float(r.step),int(r.ntr),float(r.tp),bool(r.regime),s,e)
            stresses.append({'candidate':label,'stress':sn,**{k:z[k] for k in ['cagr','mdd','trades','win','worst_trade']}})
        lb=int(r.lookback); dd=float(actual.close.iloc[-1]/actual.close.rolling(lb).max().iloc[-1]-1);levels=[-(float(r.trigger)+k*float(r.step)) for k in range(int(r.ntr))]
        cur.append({'candidate':label,'qqq_regime_on':qreg,'actual_tqqq_dd':dd,'entry_levels':str(levels),'would_enter':bool((qreg or not bool(r.regime)) and any(dd<=x for x in levels))})
    pd.DataFrame(stresses).to_csv(OUT/'selected_synthetic_stress.csv',index=False);pd.DataFrame(cur).to_csv(OUT/'current_state.csv',index=False)
    def cv(v):
        if isinstance(v,(np.bool_,bool)):return bool(v)
        if isinstance(v,(np.integer,int)):return int(v)
        if isinstance(v,(np.floating,float)):return float(v)
        return v
    meta={'actual_start':str(actual.index[0].date()),'actual_end':str(actual.index[-1].date()),'candidate_count':len(R),'selections':{lab:{k:cv(r[k]) for k in KEYS+['A_TRAIN_cagr','A_TRAIN_mdd','A_VALID_cagr','A_VALID_mdd','A_OOS_cagr','A_OOS_mdd','S_PRE2010_cagr','S_PRE2010_mdd']} for lab,r in selections.items()},'current':cur,'fee_each_side':FEE}
    (OUT/'meta.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    print('META',json.dumps(meta,ensure_ascii=False));print('\nACTUAL DETAILS\n',pd.DataFrame(details).to_string(index=False));print('\nSYNTH STRESS\n',pd.DataFrame(stresses).to_string(index=False));print('\nCURRENT\n',pd.DataFrame(cur).to_string(index=False))

if __name__=='__main__':main()
