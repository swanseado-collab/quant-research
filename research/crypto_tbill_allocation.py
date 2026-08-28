#!/usr/bin/env python3
from __future__ import annotations
import io, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

OUT=Path('results/crypto_tbill_allocation'); OUT.mkdir(parents=True,exist_ok=True)
FEE=.0005
BAND=.10
CRYPTO_WEIGHTS=np.round(np.arange(.10,1.001,.05),2)
ETH_SHARES=[0.,.10,.20,.40,.60,1.0]
GOALS=[-.20,-.30,-.40]


def fetch_binance(symbol):
    start=int(pd.Timestamp('2017-08-01',tz='UTC').timestamp()*1000)
    rows=[]; cur=start
    for base in ['https://data-api.binance.vision/api/v3/klines','https://api.binance.com/api/v3/klines']:
        try:
            rows=[]; cur=start
            while True:
                r=requests.get(base,params={'symbol':symbol,'interval':'1d','startTime':cur,'limit':1000},timeout=30)
                r.raise_for_status(); x=r.json()
                if not x: break
                rows+=x; nxt=int(x[-1][0])+86400000
                if len(x)<1000 or nxt<=cur: break
                cur=nxt; time.sleep(.03)
            if len(rows)>2500: break
        except Exception:
            rows=[]
    if not rows: raise RuntimeError(f'binance fetch failed {symbol}')
    d=pd.DataFrame(rows,columns=['ts','open','high','low','close','vol','ct','qv','n','tb','tq','ign'])
    d['date']=pd.to_datetime(d.ts,unit='ms',utc=True).dt.tz_localize(None)
    for c in ['open','high','low','close']: d[c]=pd.to_numeric(d[c],errors='coerce')
    return d[['date','open','high','low','close']].dropna().drop_duplicates('date').sort_values('date').reset_index(drop=True)


def fetch_fred():
    url='https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO'
    r=requests.get(url,timeout=30); r.raise_for_status()
    d=pd.read_csv(io.StringIO(r.text))
    d.columns=['date','yield_pct']; d['date']=pd.to_datetime(d.date)
    d['yield_pct']=pd.to_numeric(d.yield_pct,errors='coerce')
    return d.dropna().sort_values('date').reset_index(drop=True)


def prepare():
    b=fetch_binance('BTCUSDT').rename(columns={c:f'btc_{c}' for c in ['open','high','low','close']})
    e=fetch_binance('ETHUSDT').rename(columns={c:f'eth_{c}' for c in ['open','high','low','close']})
    d=b.merge(e,on='date',how='inner')
    y=fetch_fred(); d=d.merge(y,on='date',how='left'); d['yield_pct']=d.yield_pct.ffill()
    # Use the last yield known BEFORE the current crypto day. Weekend days naturally inherit Friday.
    d['yield_known']=d.yield_pct.shift(1).ffill()
    # annual quoted yield -> daily accrual proxy for a continuously rolled 3m Treasury sleeve
    d['rf_factor']=(1+d.yield_known.clip(lower=0)/100.)**(1/365.2425)
    d['btc_ma150']=d.btc_close.rolling(150,min_periods=150).mean()
    d['eth_ma200']=d.eth_close.rolling(200,min_periods=200).mean()
    above=d.btc_close>d.btc_ma150; below=d.btc_close<d.btc_ma150
    st=np.zeros(len(d),dtype=int); state=0
    for i in range(len(d)):
        if i>=2 and bool(above.iloc[i-2:i+1].all()): state=1
        elif i>=2 and bool(below.iloc[i-2:i+1].all()): state=0
        st[i]=state
    d['btc_state_close']=st
    d['eth_state_close']=(d.eth_close>d.eth_ma200).astype(int)
    return d.dropna(subset=['yield_known']).reset_index(drop=True)


def md(a):
    a=np.asarray(a,float); return float(np.min(a/np.maximum.accumulate(a)-1))
def cagr(fin,ini,s,e):
    yrs=(e-s).days/365.2425; return float((fin/ini)**(1/yrs)-1)
def end_idx(d,s,years):
    t=d.loc[s,'date']+pd.DateOffset(years=years); i=int(np.searchsorted(d.date.values,t.to_datetime64(),'right')-1)
    return i if i>s and (t-d.loc[i,'date']).days<=2 else None


def btc_sleeve(d,s,e):
    cash=1.; qty=0.; held=0; vals=[]
    for i in range(s,e+1):
        if i>s: cash*=float(d.loc[i,'rf_factor'])
        desired=int(d.loc[i-1,'btc_state_close']) if i>0 else 0
        op=float(d.loc[i,'btc_open'])
        if desired!=held:
            if desired==1 and cash>0:
                pr=cash/(1+FEE); qty=pr/op; cash=0.; held=1
            elif desired==0 and qty>0:
                cash=qty*op*(1-FEE); qty=0.; held=0
        vals.append(cash+qty*float(d.loc[i,'btc_close']))
    return np.asarray(vals,float)


def eth_sleeve(d,s,e):
    # 40% activated at start, then 5% of original sleeve at months 1..12.
    reserve=1.; reserve_pr=1.; active=0.; qty=0.; held=0; vals=[]
    dates=d.date.values
    schedule={}
    for k,p in [(0,.40)]+[(k,.05) for k in range(1,13)]:
        t=(pd.Timestamp(d.loc[s,'date'])+pd.DateOffset(months=k)).to_datetime64()
        ix=int(np.searchsorted(dates,t,'left'))
        if ix<=e: schedule[ix]=schedule.get(ix,0)+p
    for i in range(s,e+1):
        if i>s:
            f=float(d.loc[i,'rf_factor']); reserve*=f; active*=f
        desired=int(d.loc[i-1,'eth_state_close']) if i>0 else 0
        op=float(d.loc[i,'eth_open'])
        if desired!=held:
            if desired==0 and qty>0:
                active+=qty*op*(1-FEE); qty=0.; held=0
            elif desired==1:
                if active>0:
                    pr=active/(1+FEE); qty+=pr/op; active=0.
                held=1
        p=schedule.get(i,0.)
        if p>0 and reserve_pr>1e-12:
            frac=min(1.,p/reserve_pr); x=reserve*frac; reserve-=x; reserve_pr-=p; active+=x
            if desired==1 and active>0:
                pr=active/(1+FEE); qty+=pr/op; active=0.; held=1
        vals.append(reserve+active+qty*float(d.loc[i,'eth_close']))
    return np.asarray(vals,float)


def mix_crypto(btc,eth,eth_share,dates):
    if eth_share<=0: return btc/btc[0]
    if eth_share>=1: return eth/eth[0]
    # annual rebalance between the two strategy sleeves; decisions use only current sleeve values.
    ub=(1-eth_share)/btc[0]; ue=eth_share/eth[0]; vals=[]; last_year=pd.Timestamp(dates[0]).year
    for j in range(len(btc)):
        v=ub*btc[j]+ue*eth[j]; vals.append(v)
        y=pd.Timestamp(dates[j]).year
        if j>0 and y!=last_year:
            cb=ub*btc[j]; ce=ue*eth[j]; total=cb+ce
            nb=total*(1-eth_share); ne=total*eth_share
            traded=abs(nb-cb)+abs(ne-ce)
            total_after=total-traded*FEE
            nb=total_after*(1-eth_share); ne=total_after*eth_share
            ub=nb/btc[j]; ue=ne/eth[j]; vals[-1]=total_after
            last_year=y
    return np.asarray(vals,float)


def rf_index(d,s,e):
    x=np.ones(e-s+1,float)
    for j,i in enumerate(range(s+1,e+1),start=1): x[j]=x[j-1]*float(d.loc[i,'rf_factor'])
    return x


def top_portfolio(crypto,rf,target_crypto,dates,zero_rf=False):
    if zero_rf: rf=np.ones_like(rf)
    uc=target_crypto/crypto[0]; ur=(1-target_crypto)/rf[0]
    vals=[]; rebalances=0; turnover=0.
    pending=False
    prev_month=pd.Timestamp(dates[0]).month
    for j in range(len(crypto)):
        cv=uc*crypto[j]; rv=ur*rf[j]; total=cv+rv
        # Execute a rebalance one day after the monthly check, using today's values.
        if pending:
            tc=total*target_crypto; tr=total-tc; moved=abs(tc-cv)
            cost=moved*FEE; total-=cost; tc=total*target_crypto; tr=total-tc
            uc=tc/crypto[j]; ur=tr/rf[j]; cv=tc; rv=tr; total=tc+tr
            rebalances+=1; turnover+=moved
            pending=False
        vals.append(total)
        if j>0:
            m=pd.Timestamp(dates[j]).month
            if m!=prev_month:
                w=cv/total if total>0 else 0
                if abs(w-target_crypto)>BAND: pending=True
                prev_month=m
    return np.asarray(vals,float),rebalances,turnover


def segment_for(start_year,horizon):
    if horizon==3:
        if start_year<=2020:return 'TRAIN'
        if start_year==2021:return 'VALID'
        return 'OOS'
    if start_year<=2019:return 'TRAIN'
    if start_year==2020:return 'VALID'
    return 'OOS'


def summarize(z):
    return pd.Series({'cohorts':len(z),'median_cagr':z.cagr.median(),'p10_cagr':z.cagr.quantile(.1),'worst_cagr':z.cagr.min(),'median_mdd':z.mdd.median(),'worst_mdd':z.mdd.min(),'median_rebalances':z.rebalances.median(),'median_turnover':z.turnover.median()})


def main():
    d=prepare(); d.to_csv(OUT/'daily_inputs.csv',index=False)
    starts=d.groupby(d.date.dt.to_period('M')).head(1).index.tolist()
    starts=[i for i in starts if d.loc[i,'date']>=pd.Timestamp('2018-09-01') and i>=220]
    rows=[]
    for h in [3,5]:
        for s in starts:
            e=end_idx(d,s,h)
            if e is None: continue
            btc=btc_sleeve(d,s,e); eth=eth_sleeve(d,s,e); rf=rf_index(d,s,e); dates=d.loc[s:e,'date'].to_numpy(); sy=int(d.loc[s,'date'].year)
            for es in ETH_SHARES:
                cr=mix_crypto(btc,eth,es,dates)
                for cw in CRYPTO_WEIGHTS:
                    for zr in [False,True]:
                        eq,nr,to=top_portfolio(cr,rf,cw,dates,zero_rf=zr)
                        rows.append({'horizon':h,'start':d.loc[s,'date'],'end':d.loc[e,'date'],'start_year':sy,'segment':segment_for(sy,h),'crypto_weight':cw,'eth_share':es,'cash_mode':'ZERO' if zr else 'TBILL','cagr':cagr(eq[-1],eq[0],d.loc[s,'date'],d.loc[e,'date']),'return':eq[-1]/eq[0]-1,'mdd':md(eq),'rebalances':nr,'turnover':to})
    R=pd.DataFrame(rows); R.to_csv(OUT/'cohorts.csv',index=False)
    S=R.groupby(['cash_mode','horizon','segment','crypto_weight','eth_share']).apply(summarize,include_groups=False).reset_index(); S.to_csv(OUT/'segment_summary.csv',index=False)
    # Train+Valid selection only. Pool 3Y and 5Y cohorts to avoid choosing a profile on one horizon.
    tv=R[R.segment.isin(['TRAIN','VALID']) & (R.cash_mode=='TBILL')]
    agg=tv.groupby(['crypto_weight','eth_share']).apply(summarize,include_groups=False).reset_index()
    selected=[]
    for goal in GOALS:
        ok=agg[agg.worst_mdd>=goal].copy()
        if ok.empty: continue
        # rank robust return distribution, not median alone
        for c in ['median_cagr','p10_cagr','worst_cagr']:
            ok['r_'+c]=ok[c].rank(ascending=False,pct=True,method='average')
        ok['score']=ok[['r_median_cagr','r_p10_cagr','r_worst_cagr']].mean(axis=1)
        q=ok.sort_values(['score','crypto_weight']).iloc[0].to_dict(); q['goal_mdd']=goal; selected.append(q)
    SEL=pd.DataFrame(selected); SEL.to_csv(OUT/'selected_trainvalid.csv',index=False)
    # OOS evaluation for selected profiles and their zero-yield counterfactual.
    oo=[]
    for _,q in SEL.iterrows():
        for mode in ['TBILL','ZERO']:
            z=R[(R.segment=='OOS')&(R.cash_mode==mode)&(R.crypto_weight==q.crypto_weight)&(R.eth_share==q.eth_share)]
            a=summarize(z).to_dict(); a.update({'goal_mdd':q.goal_mdd,'cash_mode':mode,'crypto_weight':q.crypto_weight,'eth_share':q.eth_share}); oo.append(a)
    O=pd.DataFrame(oo); O.to_csv(OUT/'selected_oos.csv',index=False)
    # Also preserve prior-profile comparisons: BTC-only 30/50/75 and 80/20 crypto 30/45/70.
    profiles=[('BTC_DEF',.30,0.),('BTC_BAL',.50,0.),('BTC_AGG',.75,0.),('ETH20_DEF',.30,.20),('ETH20_BAL',.45,.20),('ETH20_AGG',.70,.20)]
    comp=[]
    for name,cw,es in profiles:
        for mode in ['TBILL','ZERO']:
            for seg in ['TRAIN','VALID','OOS']:
                z=R[(R.cash_mode==mode)&(R.segment==seg)&(R.crypto_weight==cw)&(R.eth_share==es)]
                if z.empty: continue
                a=summarize(z).to_dict();a.update({'profile':name,'cash_mode':mode,'segment':seg,'crypto_weight':cw,'eth_share':es});comp.append(a)
    pd.DataFrame(comp).to_csv(OUT/'prior_profile_comparison.csv',index=False)
    state={'data_start':str(d.date.min().date()),'data_end':str(d.date.max().date()),'fred_last_known_yield_pct':float(d.yield_pct.dropna().iloc[-1]),'fred_last_date_in_merged':str(d.loc[d.yield_pct.notna(),'date'].iloc[-1].date()),'fee':FEE,'band_pp':BAND,'cohort_rows':len(R)}
    (OUT/'state.json').write_text(json.dumps(state,indent=2))
    lines=['# BTC + ETH + 3M Treasury allocation research','',f"State: `{json.dumps(state)}`",'', '## Train+Validation selected profiles','',SEL.to_markdown(index=False),'','## OOS selected profiles','',O.to_markdown(index=False)]
    (OUT/'README.md').write_text('\n'.join(lines))
    print('STATE',state); print('\nSELECTED\n',SEL.to_string(index=False)); print('\nOOS\n',O.to_string(index=False))

if __name__=='__main__': main()
