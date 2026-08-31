#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

OUT=Path('results/tqqq_panic_buy_v2'); OUT.mkdir(parents=True,exist_ok=True)
FEE=0.0010  # 10 bp each side; economic-edge screen before Korean tax/FX
START='2003-01-01'

def adj_ohlc(ticker,start=START):
    x=yf.download(ticker,start=start,auto_adjust=False,progress=False)
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    x=x.dropna().copy()
    ratio=(x['Adj Close']/x['Close']).replace([np.inf,-np.inf],np.nan).ffill().bfill()
    d=pd.DataFrame(index=pd.to_datetime(x.index).tz_localize(None) if getattr(x.index,'tz',None) else pd.to_datetime(x.index))
    for c in ['Open','High','Low','Close']: d[c.lower()]=np.asarray(x[c]*ratio,dtype=float)
    return d.dropna().sort_index()

def synth_3x(q):
    out=pd.DataFrame(index=q.index,columns=['open','high','low','close'],dtype=float)
    out.iloc[0]=100.0; prev=100.0
    for i in range(1,len(q)):
        pc=float(q.close.iloc[i-1]); vals={}
        for c in ['open','high','low','close']:
            r=float(q[c].iloc[i])/pc-1
            vals[c]=max(prev*(1+3*r),1e-8)
        vals['low']=min(vals['low'],vals['open'],vals['close']); vals['high']=max(vals['high'],vals['open'],vals['close'])
        out.iloc[i]=[vals['open'],vals['high'],vals['low'],vals['close']]; prev=vals['close']
    return out

def prep(tprice,q):
    idx=tprice.index.intersection(q.index); p=tprice.loc[idx].copy(); qq=q.loc[idx]
    ma=qq.close.rolling(250,min_periods=250).mean(); above=qq.close>ma
    c3=above.rolling(3,min_periods=3).sum().eq(3)
    p['regime_open']=c3.shift(1).fillna(False).astype(bool)
    return p

def sim(p,lookback,trigger,step,ntr,tp,regime_filter=True,start=None,end=None):
    # Compute drawdown on full history first, then slice, so each evaluation period has proper warmup.
    rh=p.close.rolling(lookback,min_periods=lookback).max(); dd_prev=(p.close/rh-1).shift(1)
    mask=pd.Series(True,index=p.index)
    if start is not None: mask &= p.index>=pd.Timestamp(start)
    if end is not None: mask &= p.index<=pd.Timestamp(end)
    d=p.loc[mask]; dd=dd_prev.loc[mask]
    if len(d)<250: return None
    cash=1.0; shares=0.0; cost=0.0; stage=0; armed=True; cycle_tranche=None
    equity=[]; trades=[]; entry_date=None
    for dt,row in d.iterrows():
        o,h,c=float(row.open),float(row.high),float(row.close); reg=bool(row.regime_open)
        prevdd=float(dd.loc[dt]) if pd.notna(dd.loc[dt]) else np.nan
        # Forced regime exit at current open based only on completed prior closes.
        if shares>0 and regime_filter and not reg:
            proceeds=shares*o*(1-FEE); ret=proceeds/cost-1 if cost>0 else np.nan
            cash+=proceeds; trades.append((entry_date,dt,ret,stage,'REGIME'))
            shares=0.0; cost=0.0; stage=0; entry_date=None; armed=False; cycle_tranche=None
        # Standing take-profit limit. If gap opens through it, assume fill at open; otherwise target.
        if shares>0:
            avg=cost/shares; tgt=avg*(1+tp)
            if h>=tgt:
                fill=max(o,tgt); proceeds=shares*fill*(1-FEE); ret=proceeds/cost-1
                cash+=proceeds; trades.append((entry_date,dt,ret,stage,'TP'))
                shares=0.0; cost=0.0; stage=0; entry_date=None; armed=False; cycle_tranche=None
        # A completed cycle only re-arms after TQQQ recovers to within 5% of its rolling high.
        if shares==0 and pd.notna(prevdd) and prevdd>-0.05: armed=True
        # Prior-close drawdown determines today's open purchase stage.
        if pd.notna(prevdd) and (reg or not regime_filter):
            target_stage=0
            for k in range(ntr):
                if prevdd<=-(trigger+k*step): target_stage=k+1
            if shares==0 and not armed: target_stage=0
            while stage<target_stage and cash>1e-12:
                if stage==0:
                    # Equal tranches of CURRENT sleeve NAV at the beginning of each new cycle.
                    cycle_tranche=cash/ntr
                invest=min(cash,cycle_tranche)
                if invest<=1e-12: break
                sh=invest*(1-FEE)/o; cash-=invest; shares+=sh; cost+=invest; stage+=1
                if entry_date is None: entry_date=dt
        equity.append((dt,cash+shares*c))
    if shares>0:
        dt=d.index[-1]; c=float(d.close.iloc[-1]); proceeds=shares*c*(1-FEE); ret=proceeds/cost-1
        cash+=proceeds; trades.append((entry_date,dt,ret,stage,'END')); equity[-1]=(dt,cash)
    e=pd.Series(dict(equity)).sort_index(); yrs=(e.index[-1]-e.index[0]).days/365.25
    cagr=e.iloc[-1]**(1/yrs)-1; mdd=float((e/e.cummax()-1).min())
    td=pd.DataFrame(trades,columns=['entry','exit','trade_ret','stages','reason']) if trades else pd.DataFrame(columns=['entry','exit','trade_ret','stages','reason'])
    return {'cagr':float(cagr),'mdd':mdd,'final':float(e.iloc[-1]),'trades':len(td),'win':float((td.trade_ret>0).mean()) if len(td) else np.nan,
            'worst_trade':float(td.trade_ret.min()) if len(td) else np.nan,'avg_trade':float(td.trade_ret.mean()) if len(td) else np.nan,'equity':e,'trade_df':td}

def buyhold(price,start,end=None,name='asset'):
    d=price[price.index>=pd.Timestamp(start)]
    if end is not None:d=d[d.index<=pd.Timestamp(end)]
    e=d.close/d.close.iloc[0]; yrs=(e.index[-1]-e.index[0]).days/365.25
    return {'period_name':name,'cagr':float(e.iloc[-1]**(1/yrs)-1),'mdd':float((e/e.cummax()-1).min()),'final':float(e.iloc[-1])}

def trend_tqqq(p,start,end=None):
    d=p[p.index>=pd.Timestamp(start)]
    if end is not None:d=d[d.index<=pd.Timestamp(end)]
    # Close-to-close TQQQ exposure using prior-close-confirmed regime; intentionally simple baseline.
    r=d.close.pct_change().fillna(0); strat=np.where(d.regime_open,r,0.0); e=pd.Series((1+strat).cumprod(),index=d.index)
    yrs=(e.index[-1]-e.index[0]).days/365.25
    return {'period_name':'TQQQ_MA250C3','cagr':float(e.iloc[-1]**(1/yrs)-1),'mdd':float((e/e.cummax()-1).min()),'final':float(e.iloc[-1])}

def main():
    q=adj_ohlc('QQQ',START); syn=synth_3x(q); p=prep(syn,q)
    actual=adj_ohlc('TQQQ','2010-02-11'); pact=prep(actual,q)
    periods={'TRAIN':('2004-01-01','2015-12-31'),'VALID':('2016-01-01','2020-12-31'),'OOS':('2021-01-01',None)}
    rows=[]
    for regime in [True,False]:
      for lb in [60,120,252]:
       for trig in [.10,.15,.20,.25,.30]:
        for step in [.05,.10]:
         for ntr in [1,2,3,4]:
          for tp in [.10,.15,.20,.30,.40]:
            rec={'regime':regime,'lookback':lb,'trigger':trig,'step':step,'ntr':ntr,'tp':tp}; ok=True
            for name,(s,e) in periods.items():
                z=sim(p,lb,trig,step,ntr,tp,regime,s,e)
                if z is None: ok=False; break
                for k in ['cagr','mdd','trades','win','worst_trade','avg_trade']:rec[f'{name}_{k}']=z[k]
            if ok:rows.append(rec)
    R=pd.DataFrame(rows)
    eligible=R[(R.TRAIN_trades>=5)&(R.VALID_trades>=2)&(R.TRAIN_mdd>=-.60)&(R.VALID_mdd>=-.60)].copy()
    eligible['tv_floor']=eligible[['TRAIN_cagr','VALID_cagr']].min(axis=1)
    eligible['tv_avg']=eligible[['TRAIN_cagr','VALID_cagr']].mean(axis=1)
    eligible['tv_worst_mdd']=eligible[['TRAIN_mdd','VALID_mdd']].min(axis=1)
    eligible['score']=eligible.tv_floor+.25*eligible.tv_avg+.10*eligible.tv_worst_mdd
    eligible=eligible.sort_values(['score','tv_floor'],ascending=False); top=eligible.head(100).copy(); winner=top.iloc[0]
    plateau=eligible[(eligible.score>=winner.score-.02)&(eligible.tv_floor>0)].copy()
    keys=['regime','lookback','trigger','step','ntr','tp']; wr={k:winner[k] for k in keys}
    actual_rows=[]
    for _,r in top.head(25).iterrows():
        z=sim(pact,int(r.lookback),float(r.trigger),float(r.step),int(r.ntr),float(r.tp),bool(r.regime),'2010-03-01',None)
        actual_rows.append({**{k:r[k] for k in keys},'ACTUAL_cagr':z['cagr'],'ACTUAL_mdd':z['mdd'],'ACTUAL_trades':z['trades'],'ACTUAL_win':z['win'],'ACTUAL_worst_trade':z['worst_trade']})
    A=pd.DataFrame(actual_rows)
    details=[]
    for name,(s,e) in periods.items():
        z=sim(p,int(wr['lookback']),float(wr['trigger']),float(wr['step']),int(wr['ntr']),float(wr['tp']),bool(wr['regime']),s,e)
        details.append({'period':name,**{k:z[k] for k in ['cagr','mdd','final','trades','win','worst_trade','avg_trade']}})
        z['trade_df'].to_csv(OUT/f'winner_trades_{name}.csv',index=False)
    base=[]
    for name,(s,e) in periods.items():
        for z in [buyhold(q,s,e,'QQQ_BH'),buyhold(syn,s,e,'SYN_TQQQ_BH'),trend_tqqq(p,s,e)]:base.append({'period':name,**z})
    # Parameter robustness in plateau
    stability=[]
    for k in keys:
        g=plateau.groupby(k).agg(n=('score','size'),score_med=('score','median'),oos_cagr_med=('OOS_cagr','median'),oos_mdd_med=('OOS_mdd','median')).reset_index(); g['parameter']=k; stability.append(g.rename(columns={k:'value'}))
    stab=pd.concat(stability,ignore_index=True)
    # Current state, both synthetic and actual TQQQ
    lb=int(wr['lookback']); syn_dd=float(syn.close.iloc[-1]/syn.close.rolling(lb).max().iloc[-1]-1)
    act_dd=float(actual.close.iloc[-1]/actual.close.rolling(lb).max().iloc[-1]-1)
    qma=q.close.rolling(250).mean(); reg=bool((q.close>qma).tail(3).all())
    levels=[-(float(wr['trigger'])+k*float(wr['step'])) for k in range(int(wr['ntr']))]
    current={'data_end_qqq':str(q.index[-1].date()),'data_end_tqqq':str(actual.index[-1].date()),'qqq_regime_on':reg,
             'synthetic_tqqq_dd':syn_dd,'actual_tqqq_dd':act_dd,'entry_levels':levels,
             'would_enter_actual_next_open':bool((reg or not bool(wr['regime'])) and any(act_dd<=x for x in levels))}
    R.to_csv(OUT/'all_candidates.csv',index=False); top.to_csv(OUT/'top100.csv',index=False); plateau.to_csv(OUT/'plateau.csv',index=False)
    A.to_csv(OUT/'actual_tqqq_validation_top25.csv',index=False); pd.DataFrame(details).to_csv(OUT/'winner_periods.csv',index=False)
    pd.DataFrame(base).to_csv(OUT/'baselines.csv',index=False); stab.to_csv(OUT/'plateau_parameter_stability.csv',index=False)
    def cv(v):
        if isinstance(v,(np.bool_,bool)):return bool(v)
        if isinstance(v,(np.integer,int)):return int(v)
        if isinstance(v,(np.floating,float)):return float(v)
        return v
    meta={'winner':{k:cv(v) for k,v in wr.items()},'winner_score':float(winner.score),'candidate_count':len(R),'eligible_count':len(eligible),'plateau_count':len(plateau),'current':current,'fee_each_side':FEE}
    (OUT/'meta.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    print('META',json.dumps(meta,ensure_ascii=False)); print('\nWINNER PERIODS\n',pd.DataFrame(details).to_string(index=False)); print('\nTOP10\n',top.head(10).to_string(index=False)); print('\nACTUAL TOP10\n',A.head(10).to_string(index=False)); print('\nBASELINES\n',pd.DataFrame(base).to_string(index=False)); print('\nPLATEAU STABILITY\n',stab.to_string(index=False))

if __name__=='__main__':main()
