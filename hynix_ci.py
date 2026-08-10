import json, urllib.request
import numpy as np, pandas as pd
from pathlib import Path
OLD='https://raw.githubusercontent.com/cye2020/VAIV-Data/4ebac938d0c4a3e6e43cf231310bd47ec8bd7cb6/Stock/Kospi/000660.csv'
NEW='https://raw.githubusercontent.com/powerpower2005/buyORsell/5c50805dc9e67dda46cae87e3b46fc0eed018c56/data/000660-KRX/1d.json'

def data():
    a=pd.read_csv(OLD); a.Date=pd.to_datetime(a.Date)
    with urllib.request.urlopen(NEW) as r: j=json.load(r)
    b=pd.DataFrame(j['ohlcv']).rename(columns={'date':'Date','open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}); b.Date=pd.to_datetime(b.Date)
    c=['Date','Open','High','Low','Close','Volume']; a=a[c]; b=b[c]
    ov=a.merge(b,on='Date',suffixes=('_a','_b')); print('OVERLAP',len(ov),'MAXCLOSE', (ov.Close_a-ov.Close_b).abs().max())
    x=pd.concat([a,b]).sort_values('Date').drop_duplicates('Date',keep='last'); x=x[(x.Date>='2010-01-04')&(x.Open>0)&(x.High>0)&(x.Low>0)&(x.Close>0)&(x.Volume>0)].reset_index(drop=True)
    return x

def lev2(x):
    z=pd.DataFrame({'Date':x.Date}); z[['Open','High','Low','Close']]=np.nan; z.loc[0,['Open','High','Low','Close']]=10000.
    for i in range(1,len(x)):
        pu=x.loc[i-1,'Close']; pl=z.loc[i-1,'Close']
        for c in ['Open','High','Low','Close']: z.loc[i,c]=pl*max(1+2*(x.loc[i,c]/pu-1),1e-12)
    return z

def mdd(a):
    a=np.asarray(a,float); return np.min(a/np.maximum.accumulate(a)-1)
def stats(eq,cap,s,e):
    final=float(eq[-1]); y=(e-s).days/365.2425; return final,final/cap-1,(final/cap)**(1/y)-1,mdd(eq)
def bh(w,cap=1e7,bps=0):
    f=bps/1e4; inv=cap/(1+f); q=inv/w.Close.iloc[0]; cash=cap-inv-inv*f; eq=cash+q*w.Close.values
    return stats(eq,cap,w.Date.iloc[0],w.Date.iloc[-1])+(1,1,0,)
def dca(w,cap=1e7,n=40,bps=0):
    f=bps/1e4; cash=cap; q=0; eq=[]
    for i,r in w.reset_index(drop=True).iterrows():
        if i<min(n,len(w)):
            p=(cash/(min(n,len(w))-i))/(1+f); cash-=p*(1+f); q+=p/r.Close
        eq.append(cash+q*r.Close)
    return stats(eq,cap,w.Date.iloc[0],w.Date.iloc[-1])+(min(n,len(w)),1,0,)
def buy(cash,q,avg,amt,p,f):
    amt=min(amt,cash/(1+f)); nq=amt/p if amt>0 else 0; nav=(q*(avg or 0)+amt)/(q+nq) if q+nq else None; return cash-amt*(1+f),q+nq,nav
def sell(cash,q,avg,n,p,f):
    n=min(n,q); cash+=n*p*(1-f); q-=n; return cash,q,None if q<1e-10 else avg

def v4(w,cap=1e7,bps=0):
    f=bps/1e4; cash=cap; q=0.; avg=None; T=0.; mode='g'; firstrev=False; cycles=rev=tr=0; eq=[]; closes=w.Close.values
    for i,r in w.reset_index(drop=True).iterrows():
        c=float(r.Close); h=float(r.High)
        if q<1e-10 and mode=='g':
            T=0; amt=cash/40/(1+f); cash,q,avg=buy(cash,q,avg,amt,c,f); T=1; cycles+=1; tr+=1; eq.append(cash+q*c); continue
        if mode=='g':
            ts=T; av=avg; qs=q; cs=cash; star=av*(1+(15-.75*ts)/100); tp=av*1.15; budget=cs/max(40-ts,1e-12) if ts<40 else 0
            fixed=h>=tp; quarter=c>=star
            if fixed: cash,q,avg=sell(cash,q,avg,qs*.75,tp,f); T=ts*.25; tr+=1
            if quarter and q>1e-10:
                cash,q,avg=sell(cash,q,avg,min(qs*.25,q),c,f); tr+=1
                if q<1e-10: T=0
                elif not fixed: T=ts*.75
            if q<1e-10: avg=None; mode='g'; T=0; eq.append(cash); continue
            inc=0
            if ts<40 and budget>0:
                if ts<20:
                    if c<star:
                        cash,q,avg=buy(cash,q,avg,budget*.5/(1+f),c,f); inc+=.5; tr+=1
                    if c<=av:
                        cash,q,avg=buy(cash,q,avg,budget*.5/(1+f),c,f); inc+=.5; tr+=1
                elif c<star:
                    cash,q,avg=buy(cash,q,avg,budget/(1+f),c,f); inc+=1; tr+=1
            T+=inc
            if T>39 and q>1e-10: mode='r'; firstrev=True; rev+=1
        else:
            if firstrev:
                cash,q,avg=sell(cash,q,avg,q/20,c,f); T*=.95; firstrev=False; tr+=1
            elif i>=5:
                rs=np.mean(closes[i-5:i])
                if c>rs and q>1e-10: cash,q,avg=sell(cash,q,avg,q/20,c,f); T*=.95; tr+=1
                elif c<rs and cash>1e-8: cash,q,avg=buy(cash,q,avg,cash/4/(1+f),c,f); T=T+(40-T)*.25; tr+=1
            if q>1e-10 and avg and c>avg*.85: mode='g'; firstrev=False
        eq.append(cash+q*c)
    return stats(eq,cap,w.Date.iloc[0],w.Date.iloc[-1])+(tr,cycles,rev,)
def window(x,s,y):
    z=x[x.Date>=s]
    if z.empty:return None
    a=z.Date.iloc[0]; e=a+pd.DateOffset(years=y)
    if x.Date.iloc[-1]<e:return None
    w=x[(x.Date>=a)&(x.Date<=e)].reset_index(drop=True); return w if len(w)>=200*y else None
def compare(w):
    B=bh(w); V=v4(w); D=dca(w)
    return {'start':w.Date.iloc[0],'end':w.Date.iloc[-1],'bh_return':B[1],'bh_mdd':B[3],'v4_return':V[1],'v4_mdd':V[3],'dca_return':D[1],'dca_mdd':D[3],'winner':'B&H' if B[0]>V[0] else 'V4','v4_trades':V[4],'v4_cycles':V[5],'v4_reverse':V[6]}
def rolling(x,y,monthly=False):
    ds=pd.date_range(x.Date.min().to_period('M').start_time,x.Date.max(),freq='MS') if monthly else [pd.Timestamp(f'{a}-01-01') for a in range(x.Date.min().year,x.Date.max().year+1)]
    out=[]; seen=set()
    for s in ds:
        z=x[x.Date>=s]
        if z.empty:continue
        a=z.Date.iloc[0]
        if a in seen:continue
        seen.add(a); w=window(x,a,y)
        if w is not None: out.append(compare(w))
    return pd.DataFrame(out)
def summary(n,z):
    b=(z.winner=='B&H').sum(); v=(z.winner=='V4').sum(); print(f'{n}: N={len(z)} BH={b} V4={v} BHrate={b/len(z):.1%} medret BH={z.bh_return.median():.1%} V4={z.v4_return.median():.1%} DCA={z.dca_return.median():.1%} medMDD BH={z.bh_mdd.median():.1%} V4={z.v4_mdd.median():.1%}')
def main():
    u=data(); L=lev2(u); o=Path('results'); o.mkdir(exist_ok=True); u.to_csv(o/'underlying.csv',index=False); L.to_csv(o/'synthetic2x.csv',index=False)
    rows=[]
    for name,fn in [('B&H',bh),('V4',v4),('DCA40',dca)]:
        r=fn(L); rows.append([name,*r]); print('FULL',name,r)
    pd.DataFrame(rows,columns=['strategy','final','return','cagr','mdd','trades','cycles','reverse']).to_csv(o/'full.csv',index=False)
    for y in [3,5]:
        for mo in [False,True]:
            z=rolling(L,y,mo); n=('monthly' if mo else 'annual')+f'_{y}y'; z.to_csv(o/(n+'.csv'),index=False); summary(n,z)
            if not mo: print(z.to_string(index=False))
    feat=u[['Date','Close']].copy(); feat['ret']=feat.Close.pct_change(); feat['peak']=feat.Close.rolling(252,min_periods=60).max(); feat['dd']=feat.Close/feat.peak-1; feat['vol']=feat.ret.rolling(20).std()*np.sqrt(252); feat['vp']=feat.vol.rank(pct=True); feat['m']=feat.Date.dt.to_period('M'); q=[]
    for _,r in feat.groupby('m').first().iterrows():
        if r.dd<=-.35 and r.vp>=.75:
            for y in [1,2,3]:
                w=window(L,r.Date,y)
                if w is not None: d=compare(w); d.update(years=y,dd=r.dd,vol=r.vol); q.append(d)
    q=pd.DataFrame(q); q.to_csv(o/'current_like.csv',index=False)
    if len(q):
        for y,g in q.groupby('years'): summary(f'current_like_{y}y',g)
    s=[]
    for bp in [0,2.5,5,10]:
        for n,fn in [('B&H',bh),('V4',v4)]:
            r=fn(L,bps=bp); s.append([bp,n,r[0],r[2],r[3]])
    pd.DataFrame(s,columns=['bps','strategy','final','cagr','mdd']).to_csv(o/'cost.csv',index=False)
if __name__=='__main__': main()
