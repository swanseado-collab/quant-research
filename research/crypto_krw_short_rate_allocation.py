#!/usr/bin/env python3
from __future__ import annotations
import io,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
OUT=Path('results/crypto_krw_short_rate_allocation');OUT.mkdir(parents=True,exist_ok=True)
FEE=.0005;BAND=.10
CW=np.round(np.arange(.10,1.001,.05),2);ES=[0.,.10,.20,.40,.60,1.0];GOALS=[-.20,-.30,-.40]

def fetch_upbit(market):
    url='https://api.upbit.com/v1/candles/days';rows=[];to=None;s=requests.Session();s.headers.update({'User-Agent':'Mozilla/5.0'})
    for _ in range(40):
        p={'market':market,'count':200}
        if to:p['to']=to
        r=s.get(url,params=p,timeout=30);r.raise_for_status();x=r.json()
        if not x:break
        rows+=x;old=pd.Timestamp(x[-1]['candle_date_time_utc'])
        if old<=pd.Timestamp('2017-08-01'):break
        to=(old-pd.Timedelta(seconds=1)).isoformat()+'Z';time.sleep(.12)
    d=pd.DataFrame(rows);d['date']=pd.to_datetime(d.candle_date_time_utc).dt.tz_localize(None)
    d=d.rename(columns={'opening_price':'open','high_price':'high','low_price':'low','trade_price':'close'})
    return d[['date','open','high','low','close']].sort_values('date').drop_duplicates('date').reset_index(drop=True)

def fred_kr():
    u='https://fred.stlouisfed.org/graph/fredgraph.csv?id=IR3TIB01KRM156N';r=requests.get(u,timeout=30);r.raise_for_status()
    x=pd.read_csv(io.StringIO(r.text));x.columns=['obs_date','yield_pct'];x['obs_date']=pd.to_datetime(x.obs_date);x['yield_pct']=pd.to_numeric(x.yield_pct,errors='coerce');x=x.dropna()
    # Monthly OECD observation is published later. +50d is deliberately conservative to avoid using the monthly average before release.
    x['available_date']=x.obs_date+pd.Timedelta(days=50)
    return x.sort_values('available_date').reset_index(drop=True)

def prep():
    b=fetch_upbit('KRW-BTC').rename(columns={c:f'btc_{c}' for c in ['open','high','low','close']});e=fetch_upbit('KRW-ETH').rename(columns={c:f'eth_{c}' for c in ['open','high','low','close']});r=fred_kr()
    d=b.merge(e,on='date');rates=r[['available_date','yield_pct']].rename(columns={'available_date':'date'});d=pd.merge_asof(d.sort_values('date'),rates.sort_values('date'),on='date',direction='backward')
    d['known']=d.yield_pct.shift(1).ffill();d['rf']=(1+d.known.clip(lower=0)/100)**(1/365.2425)
    d['bma']=d.btc_close.rolling(150,min_periods=150).mean();d['ema']=d.eth_close.rolling(200,min_periods=200).mean();ab=d.btc_close>d.bma;be=d.btc_close<d.bma
    st=[];state=0
    for i in range(len(d)):
        if i>=2 and bool(ab.iloc[i-2:i+1].all()):state=1
        elif i>=2 and bool(be.iloc[i-2:i+1].all()):state=0
        st.append(state)
    d['bst']=st;d['est']=(d.eth_close>d.ema).astype(int);return d.dropna(subset=['known']).reset_index(drop=True),r

def f(d,i,t):return float(d.loc[i,'rf']) if t else 1.
def md(x):
    x=np.asarray(x,float);return float(np.min(x/np.maximum.accumulate(x)-1))
def cg(fin,s,e):return float(fin**(1/((e-s).days/365.2425))-1)
def endi(d,s,h):
    t=d.loc[s,'date']+pd.DateOffset(years=h);i=int(np.searchsorted(d.date.values,t.to_datetime64(),'right')-1);return i if i>s and (t-d.loc[i,'date']).days<=2 else None

def btc(d,s,e,t):
    cash=1.;q=0.;held=0;v=[]
    for i in range(s,e+1):
        if i>s:cash*=f(d,i,t)
        des=int(d.loc[i-1,'bst']);op=float(d.loc[i,'btc_open'])
        if des!=held:
            if des and cash>0:pr=cash/(1+FEE);q=pr/op;cash=0.;held=1
            elif (not des) and q>0:cash=q*op*(1-FEE);q=0.;held=0
        v.append(cash+q*float(d.loc[i,'btc_close']))
    return np.array(v)

def eth(d,s,e,t):
    reserve=1.;rp=1.;active=0.;q=0.;held=0;v=[];dates=d.date.values;sc={}
    for k,p in [(0,.4)]+[(k,.05) for k in range(1,13)]:
        ix=int(np.searchsorted(dates,(pd.Timestamp(d.loc[s,'date'])+pd.DateOffset(months=k)).to_datetime64(),'left'))
        if ix<=e:sc[ix]=sc.get(ix,0)+p
    for i in range(s,e+1):
        if i>s:
            z=f(d,i,t);reserve*=z;active*=z
        des=int(d.loc[i-1,'est']);op=float(d.loc[i,'eth_open'])
        if des!=held:
            if not des and q>0:active+=q*op*(1-FEE);q=0.;held=0
            elif des:
                if active>0:pr=active/(1+FEE);q+=pr/op;active=0.
                held=1
        p=sc.get(i,0.)
        if p>0 and rp>1e-12:
            frac=min(1,p/rp);x=reserve*frac;reserve-=x;rp=max(0,rp-p);active+=x
            if des and active>0:pr=active/(1+FEE);q+=pr/op;active=0.;held=1
        v.append(reserve+active+q*float(d.loc[i,'eth_close']))
    return np.array(v)

def mix(b,e,es,dates):
    if es<=0:return b.copy()
    if es>=1:return e.copy()
    ub=1-es;ue=es;v=[];yr=pd.Timestamp(dates[0]).year
    for j in range(len(b)):
        cb=ub*b[j];ce=ue*e[j];tot=cb+ce;y=pd.Timestamp(dates[j]).year
        if j>0 and y!=yr:
            nb=tot*(1-es);ne=tot*es;tot-=(abs(nb-cb)+abs(ne-ce))*FEE;nb=tot*(1-es);ne=tot*es;ub=nb/b[j];ue=ne/e[j];tot=nb+ne;yr=y
        v.append(tot)
    return np.array(v)
def rfi(d,s,e,t):
    x=np.ones(e-s+1)
    for j,i in enumerate(range(s+1,e+1),1):x[j]=x[j-1]*f(d,i,t)
    return x
def port(cr,rf,w,dates):
    uc=w;ur=1-w;v=[];pend=False;reb=0;turn=0.;mo=pd.Timestamp(dates[0]).month
    for j in range(len(cr)):
        cv=uc*cr[j];rv=ur*rf[j];tot=cv+rv
        if pend:
            tc=tot*w;m=abs(tc-cv);tot-=m*FEE;tc=tot*w;tr=tot-tc;uc=tc/cr[j];ur=tr/rf[j];cv=tc;rv=tr;tot=tc+tr;reb+=1;turn+=m;pend=False
        v.append(tot)
        if j>0:
            nm=pd.Timestamp(dates[j]).month
            if nm!=mo:
                if abs(cv/tot-w)>BAND:pend=True
                mo=nm
    return np.array(v),reb,turn
def seg(y,h):
    if h==3:return 'TRAIN' if y<=2020 else ('VALID' if y==2021 else 'OOS')
    return 'TRAIN' if y<=2019 else ('VALID' if y==2020 else 'OOS')
def sm(z):return pd.Series({'n':len(z),'median_cagr':z.cagr.median(),'p10_cagr':z.cagr.quantile(.1),'worst_cagr':z.cagr.min(),'median_mdd':z.mdd.median(),'worst_mdd':z.mdd.min(),'median_reb':z.reb.median()})

def main():
    d,rr=prep();d.to_csv(OUT/'daily_inputs.csv',index=False);rr.to_csv(OUT/'kr_short_rate_source.csv',index=False)
    starts=d.groupby(d.date.dt.to_period('M')).head(1).index.tolist();starts=[i for i in starts if d.loc[i,'date']>=pd.Timestamp('2018-09-01') and i>=220];rows=[]
    for h in [3,5]:
      for s in starts:
        e=endi(d,s,h)
        if e is None:continue
        dates=d.loc[s:e,'date'].to_numpy();y=int(d.loc[s,'date'].year)
        for mode,t in [('KR_RATE',True),('ZERO',False)]:
          bb=btc(d,s,e,t);ee=eth(d,s,e,t);rf=rfi(d,s,e,t)
          for es in ES:
            cr=mix(bb,ee,es,dates)
            for w in CW:
              eq,re,tu=port(cr,rf,w,dates);rows.append({'horizon':h,'start':d.loc[s,'date'],'end':d.loc[e,'date'],'year':y,'segment':seg(y,h),'crypto_weight':w,'eth_share':es,'cash_mode':mode,'cagr':cg(eq[-1],d.loc[s,'date'],d.loc[e,'date']),'return':eq[-1]-1,'mdd':md(np.r_[1.,eq]),'reb':re,'turn':tu})
    R=pd.DataFrame(rows);R.to_csv(OUT/'cohorts.csv',index=False)
    tv=R[(R.segment.isin(['TRAIN','VALID']))&(R.cash_mode=='KR_RATE')];A=tv.groupby(['crypto_weight','eth_share']).apply(sm,include_groups=False).reset_index();sel=[]
    for goal in GOALS:
      ok=A[A.worst_mdd>=goal].copy()
      for c in ['median_cagr','p10_cagr','worst_cagr']:ok['r_'+c]=ok[c].rank(ascending=False,pct=True)
      ok['score']=ok[['r_median_cagr','r_p10_cagr','r_worst_cagr']].mean(axis=1);q=ok.sort_values(['score','crypto_weight']).iloc[0].to_dict();q['goal']=goal;sel.append(q)
    S=pd.DataFrame(sel);S.to_csv(OUT/'selected_trainvalid.csv',index=False);oo=[]
    for _,q in S.iterrows():
      for mode in ['KR_RATE','ZERO']:
        z=R[(R.segment=='OOS')&(R.cash_mode==mode)&(R.crypto_weight==q.crypto_weight)&(R.eth_share==q.eth_share)];a=sm(z).to_dict();a.update({'goal':q.goal,'mode':mode,'crypto_weight':q.crypto_weight,'eth_share':q.eth_share});oo.append(a)
    O=pd.DataFrame(oo);O.to_csv(OUT/'selected_oos.csv',index=False)
    state={'data_start':str(d.date.min().date()),'data_end':str(d.date.max().date()),'rate_last_observation':str(rr.obs_date.max().date()),'rate_last_pct':float(rr.iloc[-1].yield_pct),'rate_availability_lag_days':50,'fee':FEE,'band':BAND}
    (OUT/'state.json').write_text(json.dumps(state,indent=2));(OUT/'README.md').write_text('# KRW validation\n\n'+json.dumps(state,indent=2)+'\n\n## Selected\n'+S.to_markdown(index=False)+'\n\n## OOS\n'+O.to_markdown(index=False));print(state);print(S.to_string(index=False));print(O.to_string(index=False))
if __name__=='__main__':main()
