#!/usr/bin/env python3
from __future__ import annotations
import time, json
from pathlib import Path
import requests, pandas as pd, numpy as np

OUT=Path('results/eth_upbit_validation'); OUT.mkdir(parents=True,exist_ok=True)
CAP=10_000.0; FEE=.0005; HORIZON=730; INIT=.40; MONTHS=12; DIPS=[-.10,-.20,-.30,-.40]


def fetch_upbit():
    url='https://api.upbit.com/v1/candles/days'; rows=[]; to=None
    sess=requests.Session(); sess.headers.update({'User-Agent':'Mozilla/5.0'})
    for _ in range(40):
        p={'market':'KRW-ETH','count':200}
        if to: p['to']=to
        r=sess.get(url,params=p,timeout=30); r.raise_for_status(); x=r.json()
        if not x: break
        rows += x
        oldest=pd.Timestamp(x[-1]['candle_date_time_utc'])
        if oldest <= pd.Timestamp('2017-08-01'): break
        to=(oldest-pd.Timedelta(seconds=1)).isoformat()+'Z'
        time.sleep(.12)
    d=pd.DataFrame(rows)
    d['date']=pd.to_datetime(d['candle_date_time_utc']).dt.tz_localize(None)
    d=d.rename(columns={'opening_price':'open','high_price':'high','low_price':'low','trade_price':'close','candle_acc_trade_volume':'vol'})
    d=d[['date','open','high','low','close','vol']].sort_values('date').drop_duplicates('date').reset_index(drop=True)
    for c in ['open','high','low','close','vol']: d[c]=pd.to_numeric(d[c],errors='coerce')
    return d.dropna().reset_index(drop=True)


def features(d):
    d=d.copy(); d['peak365']=d.close.rolling(365,min_periods=365).max(); d['dd365']=d.close/d.peak365-1
    d['mom30']=d.close.pct_change(30); d['mom7']=d.close.pct_change(7); d['ma200']=d.close.rolling(200,min_periods=200).mean(); return d

def mdd(eq):
    a=np.asarray(eq,float); return float(np.min(a/np.maximum.accumulate(a)-1))
def end_idx(dates,s):
    t=dates[s]+np.timedelta64(HORIZON,'D'); i=int(np.searchsorted(dates,t,'right')-1)
    return i if i>s and i<len(dates) and dates[i]>=t-np.timedelta64(2,'D') else None
def month_idx(dates,sd,k,e):
    t=(pd.Timestamp(sd)+pd.DateOffset(months=k)).to_datetime64(); i=int(np.searchsorted(dates,t,'left')); return i if i<=e else None

def entry_events(d,s,e):
    dates=d.date.values; op=d.open.values; lo=d.low.values; ent=float(op[s]); ia=CAP*INIT; tr=(CAP-ia)/MONTHS
    ev=[(s,0,0,ia,ent,'initial')]; sch=[month_idx(dates,dates[s],k,e) for k in range(1,MONTHS+1)]; used=[False]*MONTHS; cand=[]
    for j,x in enumerate(sch):
        if x is not None: cand.append((x,0,j,'sch',None))
    for k,lev in enumerate(DIPS):
        tar=ent*(1+lev); h=np.flatnonzero(lo[s:e+1]<=tar)
        if len(h): cand.append((s+int(h[0]),1,k,'dip',tar))
    cand.sort(key=lambda x:(x[0],x[1],x[2])); ud=set()
    for i,p,k,t,px in cand:
        if t=='sch':
            if not used[k]: used[k]=True; ev.append((i,0,k,tr,float(op[i]),'schedule'))
        else:
            if k in ud: continue
            av=[j for j,u in enumerate(used) if (not u) and sch[j] is not None and sch[j]>=i]
            if not av: continue
            j=av[-1]; used[j]=True; ud.add(k); ev.append((i,1,k,tr,float(px),'dip'))
    ev.sort(key=lambda x:(x[0],x[1],x[2])); return ev

def sim(d,s,strategy):
    dates=d.date.values; op=d.open.values; cl=d.close.values; ma=d.ma200.values; e=end_idx(dates,s)
    ev=entry_events(d,s,e); by={}
    for x in ev: by.setdefault(x[0],[]).append(x)
    last=max(x[0] for x in ev); cash=CAP; qty=0.; active=0.; flat=False; eq=[]; sells=re=0; principal=0.
    for i in range(s,e+1):
        if strategy!='HOLD' and i>last and i>0 and np.isfinite(ma[i-1]):
            if (not flat) and cl[i-1] < ma[i-1] and qty>1e-12:
                net=qty*op[i]*(1-FEE); cash+=net; active+=net; qty=0.; flat=True; sells+=1
            elif flat and cl[i-1] > ma[i-1] and active>1e-10:
                amt=min(active,cash); pr=amt/(1+FEE); qty+=pr/op[i]; cash-=amt; active-=amt; flat=False; re+=1
        for x in by.get(i,[]):
            amt=x[3]; px=x[4]; pr=amt/(1+FEE); qty+=pr/px; cash-=amt; principal+=pr
        eq.append(cash+qty*cl[i])
    return {'strategy':strategy,'start':pd.Timestamp(dates[s]),'end':pd.Timestamp(dates[e]),'return':float(eq[-1]/CAP-1),'mdd':mdd(eq),'sells':sells,'reentries':re}

def summary(z,label):
    h=z[z.strategy=='HOLD'][['start','return','mdd']].rename(columns={'return':'hr','mdd':'hm'}); out=[]
    for st,g in z.groupby('strategy'):
        x=g.merge(h,on='start'); out.append({'segment':label,'strategy':st,'cohorts':len(g),'median_return':g['return'].median(),'p10_return':g['return'].quantile(.10),'worst_return':g['return'].min(),'median_mdd':g.mdd.median(),'worst_mdd':g.mdd.min(),'median_sells':g.sells.median(),'median_reentries':g.reentries.median(),'win_vs_hold':(x['return']>x.hr).mean(),'better_mdd':(x.mdd>x.hm+1e-12).mean()})
    return pd.DataFrame(out)

def main():
    d=features(fetch_upbit()); d.to_csv(OUT/'upbit_ethkrw_daily.csv',index=False)
    dates=d.date.values; gaps=pd.Series(dates).diff().dt.days
    quality={'rows':len(d),'start':str(d.date.min().date()),'end':str(d.date.max().date()),'gaps_gt_1d':int((gaps>1).sum()),'max_gap_days':int(gaps.max())}
    (OUT/'data_quality.json').write_text(json.dumps(quality,indent=2))
    first=d.groupby(d.date.dt.to_period('M'),as_index=False).head(1).index.tolist(); starts=[i for i in first if i>=365 and end_idx(dates,i) is not None]
    rows=[]
    for s in starts:
        sig=d.iloc[s-1]
        for st in ['HOLD','MA200_C1']:
            r=sim(d,s,st); r.update({'signal_date':sig.date,'dd365':sig.dd365,'mom30':sig.mom30,'mom7':sig.mom7,'price_vs_ma200':sig.close/sig.ma200-1}); rows.append(r)
    R=pd.DataFrame(rows); R.to_csv(OUT/'cohorts.csv',index=False); y=R.start.dt.year
    segs={'ALL':R,'TRAIN_2018_2021':R[y<=2021],'VALID_2022_2023':R[(y>=2022)&(y<=2023)],'OOS_2024_PLUS':R[y>=2024],'DEEP_DD40':R[R.dd365<=-.40],'CURRENTLIKE':R[(R.dd365<=-.40)&(R.mom30>=.15)]}
    S=[]
    for name,z in segs.items(): S.append(summary(z,name))
    S=pd.concat(S,ignore_index=True); S.to_csv(OUT/'summary.csv',index=False)
    latest=d.iloc[-1]; state={'date':str(latest.date.date()),'close_krw':float(latest.close),'ma200_krw':float(latest.ma200),'price_vs_ma200':float(latest.close/latest.ma200-1),'dd365':float(latest.dd365),'mom30':float(latest.mom30),'cohorts':len(starts)}
    (OUT/'latest_state.json').write_text(json.dumps(state,indent=2))
    lines=['# Upbit ETH/KRW independent validation','',f"Data {quality['start']} to {quality['end']}, rows={quality['rows']}, cohorts={len(starts)}",'', '## Latest','```json',json.dumps(state,indent=2),'```','',S.to_markdown(index=False)]
    (OUT/'README.md').write_text('\n'.join(lines)); print('QUALITY',quality); print('STATE',state); print(S.to_string(index=False))
if __name__=='__main__': main()
