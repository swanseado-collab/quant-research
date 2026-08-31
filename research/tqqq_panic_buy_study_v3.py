#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit
from research.tqqq_panic_buy_study_v2 import adj_ohlc,synth_3x,prep,sim,buyhold,trend_tqqq,FEE,START

OUT=Path('results/tqqq_panic_buy_v3'); OUT.mkdir(parents=True,exist_ok=True)

@njit
def fastsim(op,hi,cl,reg,dd,sidx,eidx,trigger,step,ntr,tp,regime_filter,fee,years):
    cash=1.0; shares=0.0; cost=0.0; stage=0; armed=True; cyc=0.0
    peak=1.0; mdd=0.0; trades=0; wins=0; worst=1e9; tsum=0.0
    for i in range(sidx,eidx+1):
        o=op[i]; h=hi[i]; c=cl[i]; rg=reg[i]; pdv=dd[i]
        if shares>0 and regime_filter and not rg:
            proceeds=shares*o*(1-fee); tr=proceeds/cost-1.0; cash+=proceeds
            trades+=1; wins+=1 if tr>0 else 0; worst=min(worst,tr); tsum+=tr
            shares=0.; cost=0.; stage=0; armed=False; cyc=0.
        if shares>0:
            avg=cost/shares; tgt=avg*(1+tp)
            if h>=tgt:
                fill=o if o>tgt else tgt; proceeds=shares*fill*(1-fee); tr=proceeds/cost-1.0; cash+=proceeds
                trades+=1; wins+=1 if tr>0 else 0; worst=min(worst,tr); tsum+=tr
                shares=0.; cost=0.; stage=0; armed=False; cyc=0.
        if shares==0 and not np.isnan(pdv) and pdv>-0.05: armed=True
        if not np.isnan(pdv) and (rg or not regime_filter):
            target=0
            for k in range(ntr):
                if pdv<=-(trigger+k*step): target=k+1
            if shares==0 and not armed: target=0
            while stage<target and cash>1e-12:
                if stage==0:cyc=cash/ntr
                invest=cyc if cash>=cyc else cash
                shares+=invest*(1-fee)/o; cash-=invest; cost+=invest; stage+=1
        eq=cash+shares*c
        if eq>peak:peak=eq
        dr=eq/peak-1.0
        if dr<mdd:mdd=dr
    if shares>0:
        proceeds=shares*cl[eidx]*(1-fee); tr=proceeds/cost-1.0; cash+=proceeds
        trades+=1; wins+=1 if tr>0 else 0; worst=min(worst,tr); tsum+=tr
    cagr=cash**(1/years)-1.0
    return cagr,mdd,trades,(wins/trades if trades else np.nan),(worst if trades else np.nan),(tsum/trades if trades else np.nan)

def arrays(p,lb):
    rh=p.close.rolling(lb,min_periods=lb).max(); dd=(p.close/rh-1).shift(1).to_numpy(dtype=np.float64)
    return p.open.to_numpy(float),p.high.to_numpy(float),p.close.to_numpy(float),p.regime_open.to_numpy(np.bool_),dd

def bounds(idx,s,e):
    si=int(np.searchsorted(idx.values,np.datetime64(s),'left')); ei=len(idx)-1 if e is None else int(np.searchsorted(idx.values,np.datetime64(e),'right'))-1
    yrs=(idx[ei]-idx[si]).days/365.25
    return si,ei,yrs

def main():
    q=adj_ohlc('QQQ',START); syn=synth_3x(q); p=prep(syn,q)
    actual=adj_ohlc('TQQQ','2010-02-11'); pact=prep(actual,q)
    periods={'TRAIN':('2004-01-01','2015-12-31'),'VALID':('2016-01-01','2020-12-31'),'OOS':('2021-01-01',None)}
    pa={lb:arrays(p,lb) for lb in [60,120,252]}; pb={name:bounds(p.index,*se) for name,se in periods.items()}
    # compile
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
    R=pd.DataFrame(rows)
    E=R[(R.TRAIN_trades>=5)&(R.VALID_trades>=2)&(R.TRAIN_mdd>=-.60)&(R.VALID_mdd>=-.60)].copy()
    E['tv_floor']=E[['TRAIN_cagr','VALID_cagr']].min(axis=1); E['tv_avg']=E[['TRAIN_cagr','VALID_cagr']].mean(axis=1); E['tv_worst_mdd']=E[['TRAIN_mdd','VALID_mdd']].min(axis=1)
    E['score']=E.tv_floor+.25*E.tv_avg+.10*E.tv_worst_mdd; E=E.sort_values(['score','tv_floor'],ascending=False); top=E.head(100); w=top.iloc[0]
    plateau=E[(E.score>=w.score-.02)&(E.tv_floor>0)].copy(); keys=['regime','lookback','trigger','step','ntr','tp']
    wr={k:w[k] for k in keys}
    # exact event-engine validation for winner
    exact=[]
    for name,(s,e) in periods.items():
        z=sim(p,int(w.lookback),float(w.trigger),float(w.step),int(w.ntr),float(w.tp),bool(w.regime),s,e)
        exact.append({'period':name,**{k:z[k] for k in ['cagr','mdd','final','trades','win','worst_trade','avg_trade']}}); z['trade_df'].to_csv(OUT/f'winner_trades_{name}.csv',index=False)
    # actual TQQQ validation top 25 with exact event engine
    act=[]
    for _,r in top.head(25).iterrows():
        z=sim(pact,int(r.lookback),float(r.trigger),float(r.step),int(r.ntr),float(r.tp),bool(r.regime),'2010-03-01',None)
        act.append({**{k:r[k] for k in keys},'ACTUAL_cagr':z['cagr'],'ACTUAL_mdd':z['mdd'],'ACTUAL_trades':z['trades'],'ACTUAL_win':z['win'],'ACTUAL_worst_trade':z['worst_trade']})
    A=pd.DataFrame(act)
    base=[]
    for name,(s,e) in periods.items():
        for z in [buyhold(q,s,e,'QQQ_BH'),buyhold(syn,s,e,'SYN_TQQQ_BH'),trend_tqqq(p,s,e)]:base.append({'period':name,**z})
    stab=[]
    for k in keys:
        g=plateau.groupby(k).agg(n=('score','size'),score_med=('score','median'),oos_cagr_med=('OOS_cagr','median'),oos_mdd_med=('OOS_mdd','median')).reset_index(); g['parameter']=k; g=g.rename(columns={k:'value'}); stab.append(g)
    stab=pd.concat(stab,ignore_index=True)
    lb=int(w.lookback); actdd=float(actual.close.iloc[-1]/actual.close.rolling(lb).max().iloc[-1]-1); syndd=float(syn.close.iloc[-1]/syn.close.rolling(lb).max().iloc[-1]-1)
    ma=q.close.rolling(250).mean(); rg=bool((q.close>ma).tail(3).all()); levels=[-(float(w.trigger)+k*float(w.step)) for k in range(int(w.ntr))]
    current={'qqq_end':str(q.index[-1].date()),'tqqq_end':str(actual.index[-1].date()),'qqq_regime_on':rg,'actual_tqqq_dd':actdd,'synthetic_tqqq_dd':syndd,'entry_levels':levels,'would_enter_next_open':bool((rg or not bool(w.regime)) and any(actdd<=x for x in levels))}
    # exact-vs-fast error
    D=pd.DataFrame(exact).set_index('period'); fastw=top.iloc[0]; err=max(abs(float(fastw[f'{n}_cagr'])-D.loc[n,'cagr']) for n in periods)
    def cv(v):
        if isinstance(v,(np.bool_,bool)):return bool(v)
        if isinstance(v,(np.integer,int)):return int(v)
        if isinstance(v,(np.floating,float)):return float(v)
        return v
    meta={'winner':{k:cv(v) for k,v in wr.items()},'winner_score':float(w.score),'candidate_count':len(R),'eligible_count':len(E),'plateau_count':len(plateau),'fast_exact_max_cagr_error':float(err),'current':current,'fee_each_side':FEE}
    R.to_csv(OUT/'all_candidates.csv',index=False); top.to_csv(OUT/'top100.csv',index=False); plateau.to_csv(OUT/'plateau.csv',index=False); A.to_csv(OUT/'actual_tqqq_validation_top25.csv',index=False); pd.DataFrame(exact).to_csv(OUT/'winner_periods.csv',index=False); pd.DataFrame(base).to_csv(OUT/'baselines.csv',index=False); stab.to_csv(OUT/'plateau_parameter_stability.csv',index=False); (OUT/'meta.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    print('META',json.dumps(meta,ensure_ascii=False)); print('\nWINNER\n',pd.DataFrame(exact).to_string(index=False)); print('\nTOP10\n',top.head(10).to_string(index=False)); print('\nACTUAL\n',A.head(10).to_string(index=False)); print('\nBASE\n',pd.DataFrame(base).to_string(index=False)); print('\nSTABILITY\n',stab.to_string(index=False))

if __name__=='__main__':main()
