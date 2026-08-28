#!/usr/bin/env python3
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

OUT=Path('results/btc_benchmark_longterm'); OUT.mkdir(parents=True,exist_ok=True)
FEE=0.0005

def fetch(symbol):
    start=int(pd.Timestamp('2017-08-17',tz='UTC').timestamp()*1000)
    rows=[]; cur=start
    for base in ['https://data-api.binance.vision/api/v3/klines','https://api.binance.com/api/v3/klines']:
        try:
            rows=[]; cur=start
            while True:
                r=requests.get(base,params={'symbol':symbol,'interval':'1d','startTime':cur,'limit':1000},timeout=30)
                r.raise_for_status(); x=r.json()
                if not x: break
                rows += x
                nxt=int(x[-1][0])+86400000
                if len(x)<1000 or nxt<=cur: break
                cur=nxt; time.sleep(.03)
            if len(rows)>2500: break
        except Exception:
            rows=[]
    if not rows: raise RuntimeError('fetch failed')
    d=pd.DataFrame(rows,columns=['ts','open','high','low','close','vol','ct','qv','n','tb','tq','ign'])
    d['date']=pd.to_datetime(d.ts,unit='ms',utc=True).dt.tz_localize(None)
    for c in ['open','high','low','close']: d[c]=pd.to_numeric(d[c],errors='coerce')
    return d[['date','open','high','low','close']].dropna().drop_duplicates('date').sort_values('date').reset_index(drop=True)

def mdd(v):
    a=np.asarray(v,float); p=np.maximum.accumulate(a); return float(np.min(a/p-1))

def cagr(fin,ini,s,e):
    y=(e-s).days/365.2425
    return float((fin/ini)**(1/y)-1)

def end_years(d,s,y):
    t=d.loc[s,'date']+pd.DateOffset(years=y)
    i=int(np.searchsorted(d.date.values,t.to_datetime64(),side='right')-1)
    if i<=s or (t-d.loc[i,'date']).days>2:return None
    return i

def sim(d,s,e):
    # 100% at start open, one-way 0.05% buy fee; mark to close thereafter
    capital=10000.0
    principal=capital/(1+FEE)
    q=principal/float(d.loc[s,'open'])
    eq=q*d.loc[s:e,'close'].to_numpy()
    fin=float(eq[-1])
    return {'start':d.loc[s,'date'],'end':d.loc[e,'date'],'return':fin/capital-1,'cagr':cagr(fin,capital,d.loc[s,'date'],d.loc[e,'date']),'mdd':mdd(eq)}

def main():
    d=fetch('BTCUSDT'); d.to_csv(OUT/'btcusdt_daily.csv',index=False)
    first=d.groupby(d.date.dt.to_period('M')).head(1).index.tolist()
    starts=[i for i in first if d.loc[i,'date']>=pd.Timestamp('2018-08-01')]
    rows=[]
    for h in [3,5]:
        for s in starts:
            e=end_years(d,s,h)
            if e is None: continue
            r=sim(d,s,e); r.update({'horizon':f'{h}Y','start_year':d.loc[s,'date'].year}); rows.append(r)
    e=len(d)-1
    for s in starts:
        if (d.loc[e,'date']-d.loc[s,'date']).days<365*3: continue
        r=sim(d,s,e); r.update({'horizon':'MAX','start_year':d.loc[s,'date'].year}); rows.append(r)
    z=pd.DataFrame(rows); z.to_csv(OUT/'btc_cohorts.csv',index=False)
    g=z.groupby('horizon').agg(cohorts=('cagr','size'),median_cagr=('cagr','median'),p10_cagr=('cagr',lambda x:x.quantile(.1)),worst_cagr=('cagr','min'),median_mdd=('mdd','median'),worst_mdd=('mdd','min')).reset_index()
    g.to_csv(OUT/'btc_summary.csv',index=False)
    print(g.to_string(index=False))
if __name__=='__main__':main()
