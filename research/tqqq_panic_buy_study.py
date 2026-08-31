#!/usr/bin/env python3
from __future__ import annotations
import math, json
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

OUT=Path('results/tqqq_panic_buy'); OUT.mkdir(parents=True,exist_ok=True)
FEE=0.0010  # 10bp each buy/sell, conservative economic-edge screen
START='2003-01-01'

def adj_ohlc(ticker,start=START):
    x=yf.download(ticker,start=start,auto_adjust=False,progress=False)
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    x=x.dropna().copy()
    ratio=(x['Adj Close']/x['Close']).replace([np.inf,-np.inf],np.nan).ffill().bfill()
    d=pd.DataFrame(index=x.index)
    for c in ['Open','High','Low','Close']:
        d[c.lower()]=x[c]*ratio
    return d.dropna()

def synth_3x(q):
    out=pd.DataFrame(index=q.index,columns=['open','high','low','close'],dtype=float)
    out.iloc[0]=100.0
    prev=100.0
    for i in range(1,len(q)):
        pc=q.close.iloc[i-1]
        vals={}
        for c in ['open','high','low','close']:
            r=q[c].iloc[i]/pc-1
            vals[c]=max(prev*(1+3*r),0.01)
        lo=min(vals['low'],vals['open'],vals['close']); hi=max(vals['high'],vals['open'],vals['close'])
        vals['low']=lo; vals['high']=hi
        out.iloc[i]=[vals['open'],vals['high'],vals['low'],vals['close']]
        prev=vals['close']
    return out

def prep(tprice,q):
    idx=tprice.index.intersection(q.index)
    p=tprice.loc[idx].copy(); qq=q.loc[idx].copy()
    ma=qq.close.rolling(250,min_periods=250).mean()
    above=(qq.close>ma)
    c3=above.rolling(3,min_periods=3).sum().eq(3)
    p['regime_open']=c3.shift(1).fillna(False).astype(bool)
    return p

def sim(p,lookback,trigger,step,ntr,tp,regime_filter=True,start=None,end=None):
    d=p if start is None else p[p.index>=pd.Timestamp(start)]
    if end is not None: d=d[d.index<=pd.Timestamp(end)]
    if len(d)<300:return None
    # rolling high known at prior close
    rh=d.close.rolling(lookback,min_periods=max(20,lookback//3)).max()
    dd=(d.close/rh-1).shift(1)
    cash=1.0; shares=0.0; cost=0.0; stage=0; armed=True
    equity=[]; trades=[]; entry_date=None; cycle_buy=0.0
    tranche=1.0/ntr
    for i,(dt,row) in enumerate(d.iterrows()):
        o,h,l,c=float(row.open),float(row.high),float(row.low),float(row.close)
        reg=bool(row.regime_open)
        prevdd=float(dd.loc[dt]) if pd.notna(dd.loc[dt]) else np.nan
        # regime-off exit at open, using only prior closes
        if shares>0 and regime_filter and not reg:
            gross=shares*o; proceeds=gross*(1-FEE); ret=proceeds/cost-1 if cost>0 else np.nan
            cash+=proceeds; trades.append((entry_date,dt,ret,stage,'REGIME'))
            shares=0; cost=0; stage=0; entry_date=None; cycle_buy=0; armed=False
        # profit-taking limit during day
        if shares>0:
            avg=cost/shares
            tgt=avg*(1+tp)
            if h>=tgt:
                fill=max(o,tgt); gross=shares*fill; proceeds=gross*(1-FEE); ret=proceeds/cost-1
                cash+=proceeds; trades.append((entry_date,dt,ret,stage,'TP'))
                shares=0; cost=0; stage=0; entry_date=None; cycle_buy=0; armed=False
        # re-arm only after meaningful recovery from drawdown
        if shares==0 and pd.notna(prevdd) and prevdd>-0.05: armed=True
        # staged panic buys at today's open from prior-close drawdown
        if pd.notna(prevdd) and (reg or not regime_filter):
            target_stage=0
            for k in range(ntr):
                if prevdd<=-(trigger+k*step): target_stage=k+1
            if shares==0 and not armed: target_stage=0
            while stage<target_stage and cash>1e-12:
                invest=min(cash,tranche) if stage==0 else min(cash,tranche)
                if invest<=1e-12: break
                sh=invest*(1-FEE)/o
                cash-=invest; shares+=sh; cost+=invest; stage+=1
                if entry_date is None: entry_date=dt
        eq=cash+shares*c
        equity.append((dt,eq))
    # liquidate at final close
    if shares>0:
        dt=d.index[-1]; c=float(d.close.iloc[-1]); proceeds=shares*c*(1-FEE); ret=proceeds/cost-1
        cash+=proceeds; trades.append((entry_date,dt,ret,stage,'END')); shares=0
        equity[-1]=(dt,cash)
    e=pd.Series(dict(equity)).sort_index()
    yrs=(e.index[-1]-e.index[0]).days/365.25
    cagr=e.iloc[-1]**(1/yrs)-1 if yrs>0 else np.nan
    mdd=(e/e.cummax()-1).min()
    td=pd.DataFrame(trades,columns=['entry','exit','trade_ret','stages','reason']) if trades else pd.DataFrame(columns=['entry','exit','trade_ret','stages','reason'])
    return {'cagr':cagr,'mdd':mdd,'final':e.iloc[-1],'trades':len(td),'win':float((td.trade_ret>0).mean()) if len(td) else np.nan,
            'worst_trade':float(td.trade_ret.min()) if len(td) else np.nan,'avg_trade':float(td.trade_ret.mean()) if len(td) else np.nan,
            'equity':e,'trade_df':td}

def metrics_buyhold(price,start,end=None,levname='asset'):
    d=price[price.index>=pd.Timestamp(start)]
    if end: d=d[d.index<=pd.Timestamp(end)]
    r=d.close.pct_change().fillna(0); e=(1+r).cumprod(); yrs=(e.index[-1]-e.index[0]).days/365.25
    return {'name':levname,'cagr':e.iloc[-1]**(1/yrs)-1,'mdd':(e/e.cummax()-1).min(),'final':e.iloc[-1]}

def main():
    q=adj_ohlc('QQQ',START)
    syn=synth_3x(q)
    p=prep(syn,q)
    actual=adj_ohlc('TQQQ','2010-02-11')
    pact=prep(actual,q)
    periods={'TRAIN':('2004-01-01','2015-12-31'),'VALID':('2016-01-01','2020-12-31'),'OOS':('2021-01-01',None)}
    rows=[]
    for regime in [True,False]:
      for lb in [60,120,252]:
       for trig in [.10,.15,.20,.25,.30]:
        for step in [.05,.10]:
         for ntr in [1,2,3,4]:
          for tp in [.10,.15,.20,.30,.40]:
            rec={'regime':regime,'lookback':lb,'trigger':trig,'step':step,'ntr':ntr,'tp':tp}
            ok=True
            for name,(s,e) in periods.items():
                z=sim(p,lb,trig,step,ntr,tp,regime,s,e)
                if z is None: ok=False; break
                for k in ['cagr','mdd','trades','win','worst_trade','avg_trade']: rec[f'{name}_{k}']=z[k]
            if ok: rows.append(rec)
    R=pd.DataFrame(rows)
    # Selection uses TRAIN+VALID only. Require enough trades and avoid catastrophic tactical sleeves.
    eligible=R[(R.TRAIN_trades>=5)&(R.VALID_trades>=2)&(R.TRAIN_mdd>=-.60)&(R.VALID_mdd>=-.60)].copy()
    eligible['tv_floor']=eligible[['TRAIN_cagr','VALID_cagr']].min(axis=1)
    eligible['tv_avg']=(eligible.TRAIN_cagr+eligible.VALID_cagr)/2
    eligible['tv_worst_mdd']=eligible[['TRAIN_mdd','VALID_mdd']].min(axis=1)
    eligible['score']=eligible.tv_floor + .25*eligible.tv_avg + .10*eligible.tv_worst_mdd
    eligible=eligible.sort_values(['score','tv_floor'],ascending=False)
    top=eligible.head(100).copy(); winner=top.iloc[0]
    # Plateau: within 2pp score of winner and positive floor
    plateau=eligible[(eligible.score>=winner.score-.02)&(eligible.tv_floor>0)].copy()
    # actual TQQQ post-launch consistency for top 25, same params
    vals=[]
    for _,r in top.head(25).iterrows():
        z=sim(pact,int(r.lookback),float(r.trigger),float(r.step),int(r.ntr),float(r.tp),bool(r.regime),'2010-03-01',None)
        vals.append({**{k:r[k] for k in ['regime','lookback','trigger','step','ntr','tp']},'ACTUAL_cagr':z['cagr'],'ACTUAL_mdd':z['mdd'],'ACTUAL_trades':z['trades'],'ACTUAL_win':z['win']})
    A=pd.DataFrame(vals)
    # Winner detailed periods and current state
    wr={k:winner[k] for k in ['regime','lookback','trigger','step','ntr','tp']}
    details=[]
    for name,(s,e) in periods.items():
        z=sim(p,int(wr['lookback']),float(wr['trigger']),float(wr['step']),int(wr['ntr']),float(wr['tp']),bool(wr['regime']),s,e)
        details.append({'period':name,**{k:z[k] for k in ['cagr','mdd','final','trades','win','worst_trade','avg_trade']}})
        z['trade_df'].to_csv(OUT/f'winner_trades_{name}.csv',index=False)
    # baselines per period
    base=[]
    for name,(s,e) in periods.items():
        base.append({'period':name,**metrics_buyhold(q,s,e,'QQQ_BH')})
        base.append({'period':name,**metrics_buyhold(syn,s,e,'SYN_TQQQ_BH')})
    # current drawdown / signal for winner
    rh=syn.close.rolling(int(wr['lookback'])).max(); curdd=float(syn.close.iloc[-1]/rh.iloc[-1]-1)
    ma=q.close.rolling(250).mean(); reg=bool((q.close>ma).tail(3).all())
    levels=[-(float(wr['trigger'])+k*float(wr['step'])) for k in range(int(wr['ntr']))]
    current={'data_end':str(q.index[-1].date()),'qqq_regime_on':reg,'synthetic_tqqq_drawdown_from_rolling_high':curdd,'entry_levels':levels,
             'would_enter_next_open':bool((reg or not bool(wr['regime'])) and any(curdd<=x for x in levels))}
    R.to_csv(OUT/'all_candidates.csv',index=False); top.to_csv(OUT/'top100.csv',index=False); plateau.to_csv(OUT/'plateau.csv',index=False); A.to_csv(OUT/'actual_tqqq_validation_top25.csv',index=False)
    pd.DataFrame(details).to_csv(OUT/'winner_periods.csv',index=False); pd.DataFrame(base).to_csv(OUT/'baselines.csv',index=False)
    meta={'winner':{k:(bool(v) if isinstance(v,(np.bool_,bool)) else float(v) if isinstance(v,(np.floating,float)) else int(v) if isinstance(v,(np.integer,int)) else v) for k,v in wr.items()},
          'candidate_count':len(R),'eligible_count':len(eligible),'plateau_count':len(plateau),'current':current,'fee_each_side':FEE}
    (OUT/'meta.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    print('WINNER',json.dumps(meta,ensure_ascii=False)); print(pd.DataFrame(details).to_string(index=False)); print('\nTOP10'); print(top.head(10).to_string(index=False)); print('\nACTUAL'); print(A.head(10).to_string(index=False)); print('\nBASE'); print(pd.DataFrame(base).to_string(index=False))

if __name__=='__main__': main()
