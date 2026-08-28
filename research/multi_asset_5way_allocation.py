#!/usr/bin/env python3
from __future__ import annotations
import io, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import yfinance as yf

OUT=Path('results/multi_asset_5way'); OUT.mkdir(parents=True,exist_ok=True)
FEE=.0005
GOALS=[-.15,-.20,-.25,-.30,-.40]
MA_WINDOWS=[100,150,200,250]
CONFIRMS=[1,3,5]


def fred():
    r=requests.get('https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO',timeout=30); r.raise_for_status()
    x=pd.read_csv(io.StringIO(r.text)); x.columns=['date','yield_pct']; x.date=pd.to_datetime(x.date); x.yield_pct=pd.to_numeric(x.yield_pct,errors='coerce')
    return x.dropna().sort_values('date').reset_index(drop=True)

def eqdata(t):
    d=yf.download(t,start='1999-01-01',auto_adjust=True,progress=False,threads=False)
    if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
    d=d.reset_index().rename(columns={'Date':'date','Open':'open','High':'high','Low':'low','Close':'close'})
    d.date=pd.to_datetime(d.date).dt.tz_localize(None)
    return d[['date','open','high','low','close']].dropna().sort_values('date').reset_index(drop=True)

def binance(sym):
    start=int(pd.Timestamp('2017-08-01',tz='UTC').timestamp()*1000); rows=[]
    for base in ['https://data-api.binance.vision/api/v3/klines','https://api.binance.com/api/v3/klines']:
        try:
            rows=[]; cur=start
            while True:
                r=requests.get(base,params={'symbol':sym,'interval':'1d','startTime':cur,'limit':1000},timeout=30); r.raise_for_status(); z=r.json()
                if not z: break
                rows+=z; nxt=int(z[-1][0])+86400000
                if len(z)<1000 or nxt<=cur: break
                cur=nxt; time.sleep(.03)
            if len(rows)>2500: break
        except Exception: rows=[]
    if not rows: raise RuntimeError(sym)
    d=pd.DataFrame(rows,columns=['ts','open','high','low','close','v','ct','qv','n','tb','tq','ig'])
    d['date']=pd.to_datetime(d.ts,unit='ms',utc=True).dt.tz_localize(None)
    for c in ['open','high','low','close']: d[c]=pd.to_numeric(d[c],errors='coerce')
    return d[['date','open','high','low','close']].dropna().drop_duplicates('date').sort_values('date').reset_index(drop=True)

def mdd(a):
    x=np.asarray(a,float); x=np.r_[1.,x]; return float(np.min(x/np.maximum.accumulate(x)-1))
def cagr(fin,s,e):
    y=(e-s).days/365.2425; return float(fin**(1/y)-1)
def state_from(close,ma,c):
    above=close>ma; below=close<ma; st=np.zeros(len(close),int); cur=0
    for i in range(len(close)):
        if i>=c-1 and bool(above.iloc[i-c+1:i+1].all()): cur=1
        elif i>=c-1 and bool(below.iloc[i-c+1:i+1].all()): cur=0
        st[i]=cur
    return st

def add_rf(d,y):
    z=d.merge(y,on='date',how='left'); z.yield_pct=z.yield_pct.ffill(); z['yk']=z.yield_pct.shift(1).ffill(); return z.dropna(subset=['yk']).reset_index(drop=True)
def sim_trend(d,state,si,ei):
    cash=1.; qty=0.; held=0; vals=[]; trades=0; prevdate=None
    for i in range(si,ei+1):
        if prevdate is not None:
            days=(d.loc[i,'date']-prevdate).days; cash*=float((1+max(d.loc[i,'yk'],0)/100.)**(days/365.2425))
        desired=int(state[i-1]) if i>0 else 0; op=float(d.loc[i,'open'])
        if desired!=held:
            if desired and cash>0: qty=(cash/(1+FEE))/op; cash=0.; held=1; trades+=1
            elif (not desired) and qty>0: cash=qty*op*(1-FEE); qty=0.; held=0; trades+=1
        vals.append(cash+qty*float(d.loc[i,'close'])); prevdate=d.loc[i,'date']
    return np.asarray(vals),trades

def choose_equity_rule(ticker,d):
    for w in MA_WINDOWS: d[f'ma{w}']=d.close.rolling(w,min_periods=w).mean()
    rules=[('BH',None,None)] + [(f'MA{w}_C{c}',w,c) for w in MA_WINDOWS for c in CONFIRMS]
    starts=d.groupby(d.date.dt.year).head(1).index.tolist(); rows=[]
    for name,w,c in rules:
        st=np.ones(len(d),int) if name=='BH' else state_from(d.close,d[f'ma{w}'],c)
        for h in [3,5]:
            for s in starts:
                if d.loc[s,'date']<pd.Timestamp('2001-01-01') or s<260: continue
                target=d.loc[s,'date']+pd.DateOffset(years=h); e=int(np.searchsorted(d.date.values,target.to_datetime64(),'right')-1)
                if e<=s or e>=len(d) or d.loc[e,'date']>pd.Timestamp('2017-12-31') or (target-d.loc[e,'date']).days>5: continue
                eq,tr=sim_trend(d,st,s,e); sy=d.loc[s,'date'].year; seg='TRAIN' if sy<=2008 else 'VALID'
                rows.append({'ticker':ticker,'rule':name,'h':h,'start':d.loc[s,'date'],'seg':seg,'cagr':cagr(eq[-1],d.loc[s,'date'],d.loc[e,'date']),'mdd':mdd(eq),'trades':tr})
    R=pd.DataFrame(rows)
    S=R.groupby(['rule','seg']).agg(cohorts=('cagr','size'),median_cagr=('cagr','median'),p10_cagr=('cagr',lambda x:x.quantile(.1)),worst_cagr=('cagr','min'),median_mdd=('mdd','median'),worst_mdd=('mdd','min'),median_trades=('trades','median')).reset_index()
    tr=S[S.seg=='TRAIN'].drop(columns='seg'); va=S[S.seg=='VALID'].drop(columns='seg'); q=tr.merge(va,on='rule',suffixes=('_tr','_va'))
    for col in ['median_cagr_tr','p10_cagr_tr','worst_cagr_tr','median_mdd_tr','worst_mdd_tr','median_cagr_va','p10_cagr_va','worst_cagr_va','median_mdd_va','worst_mdd_va']:
        q['r_'+col]=q[col].rank(ascending=False,pct=True,method='average')
    q['score']=q[[c for c in q if c.startswith('r_')]].mean(axis=1); q=q.sort_values(['score','median_trades_va'])
    q.to_csv(OUT/f'{ticker.lower()}_rule_rank.csv',index=False); R.to_csv(OUT/f'{ticker.lower()}_rule_cohorts.csv',index=False)
    return str(q.iloc[0].rule),q

def parse_rule(rule,d,prefix):
    if rule=='BH': return np.ones(len(d),int)
    a,b=rule.replace('MA','').split('_C'); w=int(a); c=int(b)
    return state_from(d[f'{prefix}_close'],d[f'{prefix}_ma{w}'],c)

def prepare():
    y=fred(); spy=add_rf(eqdata('SPY'),y); qqq=add_rf(eqdata('QQQ'),y)
    sr,_=choose_equity_rule('SPY',spy.copy()); qr,_=choose_equity_rule('QQQ',qqq.copy())
    # calculate chosen states on native trading-day calendars before merging into daily crypto calendar
    def native(d,rule):
        if rule=='BH': d['state']=1
        else:
            a,b=rule.replace('MA','').split('_C'); w=int(a); c=int(b); d['ma']=d.close.rolling(w,min_periods=w).mean(); d['state']=state_from(d.close,d.ma,c)
        d['trade_day']=1; return d
    spy=native(spy,sr).rename(columns={c:f'spy_{c}' for c in ['open','close','state','trade_day']})[['date','spy_open','spy_close','spy_state','spy_trade_day']]
    qqq=native(qqq,qr).rename(columns={c:f'qqq_{c}' for c in ['open','close','state','trade_day']})[['date','qqq_open','qqq_close','qqq_state','qqq_trade_day']]
    b=binance('BTCUSDT').rename(columns={c:f'btc_{c}' for c in ['open','close']})[['date','btc_open','btc_close']]
    e=binance('ETHUSDT').rename(columns={c:f'eth_{c}' for c in ['open','close']})[['date','eth_open','eth_close']]
    d=b.merge(e,on='date').merge(spy,on='date',how='left').merge(qqq,on='date',how='left').merge(y,on='date',how='left')
    d.yield_pct=d.yield_pct.ffill(); d['yk']=d.yield_pct.shift(1).ffill(); d[['spy_close','spy_state','qqq_close','qqq_state']]=d[['spy_close','spy_state','qqq_close','qqq_state']].ffill(); d[['spy_trade_day','qqq_trade_day']]=d[['spy_trade_day','qqq_trade_day']].fillna(0)
    d['btc_ma150']=d.btc_close.rolling(150,min_periods=150).mean(); d['eth_ma200']=d.eth_close.rolling(200,min_periods=200).mean()
    d['btc_state']=state_from(d.btc_close,d.btc_ma150,3); d['eth_state']=(d.eth_close>d.eth_ma200).astype(int)
    return d.dropna(subset=['yk','spy_close','qqq_close']).reset_index(drop=True),sr,qr

def rf_factor(y): return float((1+max(y,0)/100.)**(1/365.2425))
def sleeve_simple(d,s,e,asset,statecol,tradeday=None):
    cash=1.; qty=0.; held=0; vals=[]
    for i in range(s,e+1):
        if i>s: cash*=rf_factor(d.loc[i,'yk'])
        can=True if tradeday is None else bool(d.loc[i,tradeday])
        desired=int(d.loc[i-1,statecol]) if i>0 else 0
        if can and desired!=held:
            op=float(d.loc[i,f'{asset}_open'])
            if desired and cash>0: qty=(cash/(1+FEE))/op; cash=0.; held=1
            elif (not desired) and qty>0: cash=qty*op*(1-FEE); qty=0.; held=0
        vals.append(cash+qty*float(d.loc[i,f'{asset}_close']))
    return np.asarray(vals,float)
def eth_sleeve(d,s,e):
    reserve=1.; reserve_pr=1.; active=0.; qty=0.; held=0; vals=[]; dates=d.date.values; sch={}
    for k,p in [(0,.40)]+[(k,.05) for k in range(1,13)]:
        ix=int(np.searchsorted(dates,(pd.Timestamp(d.loc[s,'date'])+pd.DateOffset(months=k)).to_datetime64(),'left'))
        if ix<=e: sch[ix]=sch.get(ix,0)+p
    for i in range(s,e+1):
        if i>s:
            f=rf_factor(d.loc[i,'yk']); reserve*=f; active*=f
        desired=int(d.loc[i-1,'eth_state']) if i>0 else 0; op=float(d.loc[i,'eth_open'])
        if desired!=held:
            if not desired and qty>0: active+=qty*op*(1-FEE); qty=0.; held=0
            elif desired and active>0: qty+=(active/(1+FEE))/op; active=0.; held=1
            else: held=desired
        p=sch.get(i,0.)
        if p>0 and reserve_pr>1e-12:
            frac=min(1.,p/reserve_pr); x=reserve*frac; reserve-=x; reserve_pr-=p; active+=x
            if desired and active>0: qty+=(active/(1+FEE))/op; active=0.; held=1
        vals.append(reserve+active+qty*float(d.loc[i,'eth_close']))
    return np.asarray(vals,float)
def tbill_sleeve(d,s,e):
    v=1.; out=[]
    for i in range(s,e+1):
        if i>s: v*=rf_factor(d.loc[i,'yk'])
        out.append(v)
    return np.asarray(out,float)
def end_idx(d,s,h):
    t=d.loc[s,'date']+pd.DateOffset(years=h); i=int(np.searchsorted(d.date.values,t.to_datetime64(),'right')-1)
    return i if i>s and (t-d.loc[i,'date']).days<=2 else None
def weights10():
    arr=[]
    for a in range(11):
      for b in range(11-a):
       for c in range(11-a-b):
        for e in range(11-a-b-c):
         f=10-a-b-c-e; arr.append([a,b,c,e,f])
    return np.asarray(arr,float)/10.
def port_matrix(curves,W,dates,mode='QUARTERLY',band=False):
    n=len(dates); k=len(W); pos=W.copy(); peak=np.ones(k); worst=np.zeros(k); turns=np.zeros(k); prev_total=np.ones(k)
    # component daily growth from pre-start capital=1
    G=np.empty_like(curves); G[0]=curves[0]; G[1:]=curves[1:]/curves[:-1]
    prev_m=pd.Timestamp(dates[0]).month; prev_q=(pd.Timestamp(dates[0]).year,pd.Timestamp(dates[0]).quarter); prev_y=pd.Timestamp(dates[0]).year
    for j in range(n):
        if j>0:
            dt=pd.Timestamp(dates[j]); check=False
            if mode=='MONTHLY' and dt.month!=prev_m: check=True
            elif mode=='QUARTERLY' and (dt.year,dt.quarter)!=prev_q: check=True
            elif mode=='ANNUAL' and dt.year!=prev_y: check=True
            if check:
                tot=pos.sum(1); cw=pos/np.maximum(tot[:,None],1e-15)
                do=np.ones(k,bool)
                if band:
                    thr=np.maximum(.025,.25*W); do=(np.abs(cw-W)>thr).any(axis=1)
                if do.any():
                    target=tot[:,None]*W; traded=np.abs(target-pos).sum(1); cost=traded*FEE; tot2=tot-cost; pos[do]=tot2[do,None]*W[do]; turns[do]+=traded[do]
            prev_m=dt.month; prev_q=(dt.year,dt.quarter); prev_y=dt.year
        pos*=G[j][None,:]; total=pos.sum(1); peak=np.maximum(peak,total); worst=np.minimum(worst,total/peak-1); prev_total=total
    return prev_total,worst,turns
def segment(sy,h):
    if h==3:
        if sy<=2020:return 'TRAIN'
        if sy==2021:return 'VALID'
        return 'OOS'
    if sy<=2019:return 'TRAIN'
    if sy==2020:return 'VALID'
    return 'OOS'
def summarize(g):
    return pd.Series({'cohorts':len(g),'median_cagr':g.cagr.median(),'p10_cagr':g.cagr.quantile(.1),'worst_cagr':g.cagr.min(),'median_mdd':g.mdd.median(),'worst_mdd':g.mdd.min(),'median_turnover':g.turnover.median()})

def main():
    d,sr,qr=prepare(); d.to_csv(OUT/'daily_inputs.csv',index=False); W=weights10(); names=['SPY','QQQ','BTC','ETH','TBILL']
    starts=d.groupby(d.date.dt.to_period('M')).head(1).index.tolist(); starts=[s for s in starts if d.loc[s,'date']>=pd.Timestamp('2018-09-01') and s>=260]
    rows=[]
    for h in [3,5]:
      for s in starts:
        e=end_idx(d,s,h)
        if e is None: continue
        curves=np.column_stack([sleeve_simple(d,s,e,'spy','spy_state','spy_trade_day'),sleeve_simple(d,s,e,'qqq','qqq_state','qqq_trade_day'),sleeve_simple(d,s,e,'btc','btc_state'),eth_sleeve(d,s,e),tbill_sleeve(d,s,e)])
        fin,dd,to=port_matrix(curves,W,d.loc[s:e,'date'].to_numpy(),'QUARTERLY',False); years=(d.loc[e,'date']-d.loc[s,'date']).days/365.2425; cg=fin**(1/years)-1; sy=d.loc[s,'date'].year; sg=segment(sy,h)
        for ix,w in enumerate(W): rows.append({'horizon':h,'start':d.loc[s,'date'],'end':d.loc[e,'date'],'start_year':sy,'segment':sg,'wid':ix,**{names[j].lower():w[j] for j in range(5)},'cagr':cg[ix],'mdd':dd[ix],'turnover':to[ix]})
    R=pd.DataFrame(rows); R.to_csv(OUT/'cohorts_quarterly.csv',index=False)
    S=R.groupby(['segment','wid','spy','qqq','btc','eth','tbill']).apply(summarize,include_groups=False).reset_index(); S.to_csv(OUT/'segment_summary.csv',index=False)
    tv=R[R.segment.isin(['TRAIN','VALID'])]; A=tv.groupby(['wid','spy','qqq','btc','eth','tbill']).apply(summarize,include_groups=False).reset_index(); picks=[]
    for goal in GOALS:
        q=A[A.worst_mdd>=goal].copy()
        for c in ['median_cagr','p10_cagr','worst_cagr','median_mdd']:
            q['r_'+c]=q[c].rank(ascending=False,pct=True,method='average')
        q['score']=q[[c for c in q if c.startswith('r_')]].mean(axis=1); z=q.sort_values(['score','worst_cagr'],ascending=[True,False]).iloc[0].to_dict(); z['goal_mdd']=goal; picks.append(z)
    P=pd.DataFrame(picks); P.to_csv(OUT/'selected_trainvalid.csv',index=False)
    oo=[]
    for _,p in P.iterrows():
        z=R[(R.segment=='OOS')&(R.wid==int(p.wid))]; a=summarize(z).to_dict(); a.update({'goal_mdd':p.goal_mdd,'wid':int(p.wid),'spy':p.spy,'qqq':p.qqq,'btc':p.btc,'eth':p.eth,'tbill':p.tbill}); oo.append(a)
    O=pd.DataFrame(oo); O.to_csv(OUT/'selected_oos.csv',index=False)
    # low-overlap annual January starts for selected profiles
    low=[]
    for _,p in P.iterrows():
        z=R[(R.wid==int(p.wid)) & (R.start.dt.month==1)];
        for sg,g in z.groupby('segment'):
            a=summarize(g).to_dict(); a.update({'goal_mdd':p.goal_mdd,'segment':sg,'spy':p.spy,'qqq':p.qqq,'btc':p.btc,'eth':p.eth,'tbill':p.tbill}); low.append(a)
    pd.DataFrame(low).to_csv(OUT/'annual_start_sensitivity.csv',index=False)
    # Rebalancing sensitivity on selected fixed weights only
    sens=[]
    for _,p in P.iterrows():
      w=np.array([[p.spy,p.qqq,p.btc,p.eth,p.tbill]],float)
      for h in [3,5]:
       for s in starts:
        e=end_idx(d,s,h)
        if e is None: continue
        curves=np.column_stack([sleeve_simple(d,s,e,'spy','spy_state','spy_trade_day'),sleeve_simple(d,s,e,'qqq','qqq_state','qqq_trade_day'),sleeve_simple(d,s,e,'btc','btc_state'),eth_sleeve(d,s,e),tbill_sleeve(d,s,e)])
        for mode,band in [('ANNUAL',False),('QUARTERLY',False),('MONTHLY',False),('MONTHLY',True)]:
            fin,dd,to=port_matrix(curves,w,d.loc[s:e,'date'].to_numpy(),mode,band); years=(d.loc[e,'date']-d.loc[s,'date']).days/365.2425
            sens.append({'goal_mdd':p.goal_mdd,'mode':mode+('_BAND' if band else ''),'horizon':h,'start':d.loc[s,'date'],'segment':segment(d.loc[s,'date'].year,h),'cagr':fin[0]**(1/years)-1,'mdd':dd[0],'turnover':to[0]})
    SD=pd.DataFrame(sens); SS=SD.groupby(['goal_mdd','mode','segment']).apply(summarize,include_groups=False).reset_index(); SS.to_csv(OUT/'rebalance_sensitivity.csv',index=False)
    # current completed signals: equities latest own trade date, crypto only last fully completed UTC day
    now=pd.Timestamp.utcnow().tz_localize(None); cutoff=now.normalize()-pd.Timedelta(days=1) if now.hour<24 else now.normalize()
    dc=d[d.date<=cutoff].copy(); last=dc.iloc[-1]
    state={'data_end':str(d.date.max().date()),'signal_cutoff':str(cutoff.date()),'spy_rule':sr,'qqq_rule':qr,'spy_state':int(last.spy_state),'qqq_state':int(last.qqq_state),'btc_state':int(last.btc_state),'eth_state':int(last.eth_state),'last_yield_pct':float(last.yield_pct)}
    (OUT/'state.json').write_text(json.dumps(state,indent=2)); (OUT/'README.md').write_text('# Five-asset allocation research\n\n'+json.dumps(state,indent=2)+'\n\n## Selected Train+Validation\n\n'+P.to_markdown(index=False)+'\n\n## OOS\n\n'+O.to_markdown(index=False))
    print('STATE',state); print('\nSELECTED\n',P.to_string(index=False)); print('\nOOS\n',O.to_string(index=False))
if __name__=='__main__': main()
