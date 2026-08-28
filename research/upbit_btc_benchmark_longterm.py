#!/usr/bin/env python3
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
OUT=Path('results/upbit_btc_benchmark_longterm'); OUT.mkdir(parents=True,exist_ok=True)
FEE=.0005

def fetch():
    url='https://api.upbit.com/v1/candles/days'; rows=[]; to=None
    s=requests.Session(); s.headers.update({'User-Agent':'Mozilla/5.0'})
    for _ in range(40):
        p={'market':'KRW-BTC','count':200}
        if to: p['to']=to
        r=s.get(url,params=p,timeout=30); r.raise_for_status(); x=r.json()
        if not x: break
        rows += x
        oldest=pd.Timestamp(x[-1]['candle_date_time_utc'])
        if oldest<=pd.Timestamp('2017-08-01'): break
        to=(oldest-pd.Timedelta(seconds=1)).isoformat()+'Z'; time.sleep(.12)
    d=pd.DataFrame(rows); d['date']=pd.to_datetime(d.candle_date_time_utc).dt.tz_localize(None)
    d=d.rename(columns={'opening_price':'open','high_price':'high','low_price':'low','trade_price':'close'})
    return d[['date','open','high','low','close']].sort_values('date').drop_duplicates('date').reset_index(drop=True)

def md(v):
    a=np.asarray(v,float); return float(np.min(a/np.maximum.accumulate(a)-1))
def cg(fin,ini,s,e):
    y=(e-s).days/365.2425; return float((fin/ini)**(1/y)-1)
def endy(d,s,y):
    t=d.loc[s,'date']+pd.DateOffset(years=y); i=int(np.searchsorted(d.date.values,t.to_datetime64(),'right')-1)
    return i if i>s and (t-d.loc[i,'date']).days<=2 else None
def sim(d,s,e):
    cap=10000.; pr=cap/(1+FEE); q=pr/float(d.loc[s,'open']); eq=q*d.loc[s:e,'close'].to_numpy(); fin=float(eq[-1])
    return {'start':d.loc[s,'date'],'end':d.loc[e,'date'],'return':fin/cap-1,'cagr':cg(fin,cap,d.loc[s,'date'],d.loc[e,'date']),'mdd':md(eq)}
def main():
    d=fetch(); d.to_csv(OUT/'upbit_btckrw_daily.csv',index=False)
    first=d.groupby(d.date.dt.to_period('M')).head(1).index.tolist(); starts=[i for i in first if d.loc[i,'date']>=pd.Timestamp('2018-10-01')]
    rows=[]
    for h in [3,5]:
        for s in starts:
            e=endy(d,s,h)
            if e is None: continue
            r=sim(d,s,e); r.update({'horizon':f'{h}Y','start_year':d.loc[s,'date'].year}); rows.append(r)
    e=len(d)-1
    for s in starts:
        if (d.loc[e,'date']-d.loc[s,'date']).days<365*3: continue
        r=sim(d,s,e); r.update({'horizon':'MAX','start_year':d.loc[s,'date'].year}); rows.append(r)
    z=pd.DataFrame(rows); z.to_csv(OUT/'btc_krw_cohorts.csv',index=False)
    g=z.groupby('horizon').agg(cohorts=('cagr','size'),median_cagr=('cagr','median'),p10_cagr=('cagr',lambda x:x.quantile(.1)),worst_cagr=('cagr','min'),median_mdd=('mdd','median'),worst_mdd=('mdd','min')).reset_index(); g.to_csv(OUT/'btc_krw_summary.csv',index=False); print(g.to_string(index=False))
if __name__=='__main__':main()
