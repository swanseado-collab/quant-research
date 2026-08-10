from hynix_ci import data,lev2,mdd,stats,buy,sell
import numpy as np,pandas as pd

def sim(w,tp_pct=.15, star_intercept=15., star_slope=.75, reverse=True, fixed_use_high=True):
    cap=1e7; cash=cap;q=0.;avg=None;T=0.;mode='g';firstrev=False;cycles=rev=tr=0;eq=[]; closes=w.Close.values; maxT=0
    for i,r in w.reset_index(drop=True).iterrows():
      c=float(r.Close);h=float(r.High)
      if q<1e-10 and mode=='g':
        T=0; amt=cash/40; cash,q,avg=buy(cash,q,avg,amt,c,0);T=1;cycles+=1;tr+=1;eq.append(cash+q*c);maxT=max(maxT,T);continue
      if mode=='g':
        ts=T;av=avg;qs=q;cs=cash;star=av*(1+(star_intercept-star_slope*ts)/100);tp=av*(1+tp_pct);budget=cs/max(40-ts,1e-12) if ts<40 else 0
        fixed=(h>=tp if fixed_use_high else c>=tp); quarter=c>=star
        if fixed: cash,q,avg=sell(cash,q,avg,qs*.75,tp,0);T=ts*.25;tr+=1
        if quarter and q>1e-10:
          cash,q,avg=sell(cash,q,avg,min(qs*.25,q),c,0);tr+=1
          if q<1e-10:T=0
          elif not fixed:T=ts*.75
        if q<1e-10:avg=None;mode='g';T=0;eq.append(cash);continue
        inc=0
        if ts<40 and budget>0:
          if ts<20:
            if c<star: cash,q,avg=buy(cash,q,avg,budget*.5,c,0);inc+=.5;tr+=1
            if c<=av: cash,q,avg=buy(cash,q,avg,budget*.5,c,0);inc+=.5;tr+=1
          elif c<star: cash,q,avg=buy(cash,q,avg,budget,c,0);inc+=1;tr+=1
        T+=inc
        if reverse and T>39 and q>1e-10:mode='r';firstrev=True;rev+=1
      else:
        if firstrev:cash,q,avg=sell(cash,q,avg,q/20,c,0);T*=.95;firstrev=False;tr+=1
        elif i>=5:
          rs=np.mean(closes[i-5:i])
          if c>rs and q>1e-10:cash,q,avg=sell(cash,q,avg,q/20,c,0);T*=.95;tr+=1
          elif c<rs and cash>1e-8:cash,q,avg=buy(cash,q,avg,cash/4,c,0);T=T+(40-T)*.25;tr+=1
        if q>1e-10 and avg and c>avg*(1-tp_pct): mode='g';firstrev=False
      maxT=max(maxT,T);eq.append(cash+q*c)
    s=stats(eq,cap,w.Date.iloc[0],w.Date.iloc[-1]);return (*s,tr,cycles,rev,maxT)

u=data();u=u[u.Date<=pd.Timestamp('2026-07-01')].reset_index(drop=True);L=lev2(u)
target=139489008
rows=[]
for tp in [.07,.08,.09,.10,.11,.12,.13,.14,.15,.16,.18,.20]:
 for hi in [True,False]:
  r=sim(L,tp_pct=tp,fixed_use_high=hi);rows.append((tp,hi,*r,abs(r[0]-target)))
for inter,slope,label in [(15,.75,'TQQQ'),(20,1.,'SOXL')]:
 r=sim(L,tp_pct=.15 if label=='TQQQ' else .20,star_intercept=inter,star_slope=slope);rows.append((label,True,*r,abs(r[0]-target)))
cols=['param','high','final','ret','cagr','mdd','trades','cycles','reverse','maxT','distance']
df=pd.DataFrame(rows,columns=cols).sort_values('distance');print(df.to_string(index=False));
