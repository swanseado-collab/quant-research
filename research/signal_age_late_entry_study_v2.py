#!/usr/bin/env python3
import time, requests
import pandas as pd
from research import signal_age_late_entry_study as s

def binance_fixed(sym,start):
    q=int(pd.Timestamp(start,tz='UTC').timestamp()*1000); rows=[]
    for base in ['https://data-api.binance.vision/api/v3/klines','https://api.binance.com/api/v3/klines']:
      try:
        rows=[]; cur=q
        while True:
            r=requests.get(base,params={'symbol':sym,'interval':'1d','startTime':cur,'limit':1000},timeout=30); r.raise_for_status(); z=r.json()
            if not z: break
            rows+=z; nxt=int(z[-1][0])+86400000
            if len(z)<1000 or nxt<=cur: break
            cur=nxt; time.sleep(.02)
        if len(rows)>1000: break
      except Exception: rows=[]
    if not rows: raise RuntimeError('BTC download failed')
    d=pd.DataFrame(rows,columns=['ts','open','high','low','close','v','ct','qv','n','tb','tq','ig'])
    d['date']=pd.to_datetime(d.ts,unit='ms',utc=True).dt.tz_localize(None).dt.normalize()
    for c in ['open','high','low','close']: d[c]=pd.to_numeric(d[c],errors='coerce')
    today=pd.Timestamp.now('UTC').tz_localize(None).normalize()
    return d[d['date']<today][['date','open','high','low','close']].dropna().drop_duplicates('date').sort_values('date').reset_index(drop=True)

s.binance=binance_fixed
if __name__=='__main__': s.main()
