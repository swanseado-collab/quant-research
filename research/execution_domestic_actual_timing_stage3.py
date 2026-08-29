#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from research import execution_taxable_stage1_v3 as patched
from research import execution_taxable_stage1 as v

OUT=Path('results/execution_domestic_actual_timing_stage3'); OUT.mkdir(parents=True,exist_ok=True)
NOTIONAL=300_000_000.


def raw_kr(ticker,start='2021-01-01'):
    x=yf.Ticker(ticker).history(start=start,auto_adjust=False,actions=True)
    if x.empty: raise RuntimeError(ticker)
    x=x.reset_index(); x.columns=[str(c).lower() for c in x.columns]; dc='date' if 'date' in x.columns else x.columns[0]
    x=x.rename(columns={dc:'date'}); x['date']=pd.to_datetime(x.date).dt.tz_localize(None).dt.normalize()
    if 'dividends' not in x:x['dividends']=0.
    return x[['date','open','close','dividends']].dropna(subset=['open','close']).sort_values('date')


def prepare_actual():
    d,_,_=v.prepare_data()
    sp=raw_kr('379800.KS').rename(columns={'open':'spy_actual_open','close':'spy_actual_close','dividends':'spy_actual_div'})
    nq=raw_kr('379810.KS').rename(columns={'open':'qqq_actual_open','close':'qqq_actual_close','dividends':'qqq_actual_div'})
    d=d.merge(sp,on='date',how='left').merge(nq,on='date',how='left')
    for a in ('spy','qqq'):
        trade=d[f'{a}_actual_close'].notna().astype(int)
        d[f'{a}_kr_trade_day']=trade
        d[f'{a}_actual_close']=d[f'{a}_actual_close'].ffill(); d[f'{a}_actual_open']=d[f'{a}_actual_open'].fillna(d[f'{a}_actual_close']); d[f'{a}_actual_div']=d[f'{a}_actual_div'].fillna(0.)
        # US signal already in d is same-date U.S. close. Korean open on date D can only use signal through D-1.
        d[f'{a}_kr_state']=d[f'{a}_state'].shift(1).ffill().fillna(0).astype(int)
    return d.dropna(subset=['spy_actual_close','qqq_actual_close']).reset_index(drop=True)


def sim_actual(d,s,e,notional,rebal,crypto_tax):
    # Temporarily replace U.S. sleeve raw KRW execution paths/trading calendar with actual Korean-listed ETFs.
    backup={}
    for a in ('spy','qqq'):
        for c in ('raw_open','raw_close','div','state','trade_day'):
            key=f'{a}_{c}'; backup[key]=d[key].copy() if key in d else None
        d[f'{a}_raw_open']=d[f'{a}_actual_open']; d[f'{a}_raw_close']=d[f'{a}_actual_close']; d[f'{a}_div']=d[f'{a}_actual_div']; d[f'{a}_state']=d[f'{a}_kr_state']; d[f'{a}_trade_day']=d[f'{a}_kr_trade_day']
    old_proxy=v.PROXY_ER.copy(); old_target=v.TARGET_ER['DOMESTIC_ETF'].copy()
    # Actual prices already contain actual product costs; prevent fee-difference scaling.
    v.PROXY_ER['spy']=v.TARGET_ER['DOMESTIC_ETF']['spy']; v.PROXY_ER['qqq']=v.TARGET_ER['DOMESTIC_ETF']['qqq']
    try: return v.sim(d,s,e,'DOMESTIC_ETF',notional,rebal,crypto_tax,v.COST_BASE['DOMESTIC_ETF'])
    finally:
        v.PROXY_ER.update(old_proxy); v.TARGET_ER['DOMESTIC_ETF'].update(old_target)
        for a in ('spy','qqq'):
            for c in ('raw_open','raw_close','div','state','trade_day'):
                key=f'{a}_{c}'
                if backup[key] is not None:d[key]=backup[key]


def main():
    d=prepare_actual(); starts=d.groupby(d.date.dt.to_period('M')).head(1).index.tolist(); rows=[]
    for s in starts:
        if d.date.iloc[s]<pd.Timestamp('2021-05-01') or s<260:continue
        e=v.m.end_idx(d,s,3)
        if e is None:continue
        for rb in v.REBAL_MODES:
          for ct in v.CRYPTO_TAX:
            za=sim_actual(d,s,e,NOTIONAL,rb,ct)
            zs=v.sim(d,s,e,'DOMESTIC_ETF',NOTIONAL,rb,ct,v.COST_BASE['DOMESTIC_ETF'])
            for typ,z in [('ACTUAL_KR_TIMING',za),('SYNTHETIC_WRAPPER',zs)]: rows.append({'type':typ,'rebalance':rb,'crypto_tax':ct,'start':d.date.iloc[s],'end':d.date.iloc[e],**z})
    R=pd.DataFrame(rows); R.to_csv(OUT/'cohorts_3y.csv',index=False)
    S=R.groupby(['type','rebalance','crypto_tax']).apply(v.summarize,include_groups=False).reset_index(); S.to_csv(OUT/'summary.csv',index=False)
    # Paired differences by exact start/rebalance/tax scenario.
    p=R.pivot(index=['start','end','rebalance','crypto_tax'],columns='type',values=['cagr_liquidated','mdd','tax_paid','trade_cost']).reset_index(); p.columns=['_'.join([str(x) for x in c if str(x)]) if isinstance(c,tuple) else c for c in p.columns]; p.to_csv(OUT/'paired.csv',index=False)
    meta={'products':{'SP500':'379800.KS','NASDAQ100':'379810.KS'},'signal_timing':'U.S. signal through calendar D-1 only; execute Korean-listed ETF at date D Korean open','notional':NOTIONAL,'horizon':'3Y only because product inception 2021','tax_model':'same DOMESTIC_ETF tax model as execution stage1','note':'Actual KR-listed OHLC/dividends; no synthetic fee scaling for SPY/QQQ sleeves.'}; (OUT/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    print('META',json.dumps(meta,ensure_ascii=False)); print('\nSUMMARY\n',S.to_string(index=False));
if __name__=='__main__':main()
