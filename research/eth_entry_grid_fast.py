#!/usr/bin/env python3
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

OUT=Path('results/eth_entry_grid_fast'); OUT.mkdir(parents=True,exist_ok=True)
CAPITAL=10_000.0; FEE=0.0005; HORIZON=730
INITIAL=[.2,.3,.4,.5,.6,.7,.8]
MONTHS=[1,3,6,9,12]
DIPS={'none':[], '10_20_30_40':[-.10,-.20,-.30,-.40], '15_25_35_45':[-.15,-.25,-.35,-.45], '10_15_20_30':[-.10,-.15,-.20,-.30]}

def fetch():
    start=int(pd.Timestamp('2017-08-17',tz='UTC').timestamp()*1000)
    errs=[]
    for base in ['https://data-api.binance.vision/api/v3/klines','https://api.binance.com/api/v3/klines']:
        try:
            rows=[]; cur=start
            while True:
                r=requests.get(base,params={'symbol':'ETHUSDT','interval':'1d','startTime':cur,'limit':1000},timeout=30)
                r.raise_for_status(); x=r.json()
                if not x: break
                rows+=x; nxt=int(x[-1][0])+86400000
                if len(x)<1000 or nxt<=cur: break
                cur=nxt; time.sleep(.03)
            if len(rows)>2500:
                d=pd.DataFrame(rows,columns=['ts','open','high','low','close','vol','ct','qv','n','tb','tq','ign'])
                d['date']=pd.to_datetime(d.ts,unit='ms',utc=True).dt.tz_localize(None)
                for c in ['open','high','low','close','vol']: d[c]=pd.to_numeric(d[c],errors='coerce')
                return d[['date','open','high','low','close','vol']].dropna().drop_duplicates('date').sort_values('date').reset_index(drop=True)
        except Exception as e: errs.append(repr(e))
    raise RuntimeError(errs)

def features(d):
    d=d.copy(); d['peak365']=d.close.rolling(365,min_periods=365).max(); d['dd365']=d.close/d.peak365-1
    d['mom30']=d.close.pct_change(30); d['mom7']=d.close.pct_change(7); d['ma200']=d.close.rolling(200,min_periods=200).mean(); return d

def md(eq):
    p=np.maximum.accumulate(eq); return float(np.min(eq/p-1))

def end_index(dates,start_idx):
    target=dates[start_idx]+np.timedelta64(HORIZON,'D')
    i=int(np.searchsorted(dates,target,side='right')-1)
    if i<=start_idx or i>=len(dates) or dates[i] < target-np.timedelta64(2,'D'): return None
    return i

def month_idx(dates,start_date,k,end):
    t=(pd.Timestamp(start_date)+pd.DateOffset(months=k)).to_datetime64()
    i=int(np.searchsorted(dates,t,side='left')); return i if i<=end else None

def eval_events(d,start,init,m,dip_name,dip_levels):
    dates=d.date.values; end=end_index(dates,start)
    if end is None: return None
    opens=d.open.values; lows=d.low.values; closes=d.close.values
    entry=float(opens[start]); events=[]
    # event tuple: (idx, priority, suborder, total_amount, price, label)
    init_amt=CAPITAL*init
    events.append((start,0,0,init_amt,entry,'initial'))
    if init<1:
        tranche=(CAPITAL-init_amt)/m
        sched=[month_idx(dates,dates[start],k,end) for k in range(1,m+1)]
        consumed=[False]*m
        candidates=[]
        for j,si in enumerate(sched):
            if si is not None: candidates.append((si,0,j,'schedule',None))
        for k,lev in enumerate(dip_levels):
            target=entry*(1+lev)
            hit=np.flatnonzero(lows[start:end+1] <= target)
            if len(hit): candidates.append((start+int(hit[0]),1,k,'dip',target))
        candidates.sort(key=lambda x:(x[0],x[1],x[2])) # open DCA before intraday dip on same day
        used_dips=set()
        for idx,pri,key,typ,px in candidates:
            if typ=='schedule':
                j=key
                if not consumed[j]:
                    consumed[j]=True; events.append((idx,0,j,tranche,float(opens[idx]),'schedule'))
            else:
                k=key
                if k in used_dips: continue
                avail=[j for j,u in enumerate(consumed) if (not u) and sched[j] is not None and sched[j]>=idx]
                if not avail: continue
                j=avail[-1]; consumed[j]=True; used_dips.add(k)
                events.append((idx,1,k,tranche,float(px),'dip'))
    events.sort(key=lambda x:(x[0],x[1],x[2]))
    # build portfolio path vectorized
    n=end-start+1; cash_delta=np.zeros(n); qty_delta=np.zeros(n)
    principal=0.; qtot=0.; fees=0.; dipfills=0
    for idx,pri,sub,amt,px,label in events:
        pr=amt/(1+FEE); fe=amt-pr; q=pr/px
        cash_delta[idx-start]-=amt; qty_delta[idx-start]+=q
        principal+=pr; qtot+=q; fees+=fe; dipfills+=int(label=='dip')
    cash=CAPITAL+np.cumsum(cash_delta); qty=np.cumsum(qty_delta); eq=cash+qty*closes[start:end+1]
    spent=-cash_delta.sum(); deploy=np.nan
    if spent>=CAPITAL-1e-6:
        ix=np.flatnonzero(cash<=1e-6); deploy=int((dates[start+ix[0]]-dates[start])/np.timedelta64(1,'D')) if len(ix) else np.nan
    return {'strategy':('BASE_LUMP' if init==1 else f'P{int(init*100)}_DCA{m}_{dip_name}'), 'initial_pct':init,'dca_months':m,'dip_set':dip_name,
            'start':pd.Timestamp(dates[start]),'end':pd.Timestamp(dates[end]),'entry_open':entry,'final_equity':float(eq[-1]),'return':float(eq[-1]/CAPITAL-1),
            'mdd':md(eq),'avg_buy_price':float(principal/qtot),'cash_end':float(cash[-1]),'buys':len(events),'deploy_days':deploy,'dip_fills':dipfills}

def pure_dca(d,start,m):
    dates=d.date.values; end=end_index(dates,start)
    if end is None:return None
    opens=d.open.values; closes=d.close.values; tranche=CAPITAL/m
    idxs=[start]+[month_idx(dates,dates[start],k,end) for k in range(1,m)]
    idxs=[i for i in idxs if i is not None]
    n=end-start+1; cd=np.zeros(n); qd=np.zeros(n); principal=0.;qtot=0.
    for i in idxs:
        pr=tranche/(1+FEE); q=pr/opens[i]; cd[i-start]-=tranche; qd[i-start]+=q;principal+=pr;qtot+=q
    cash=CAPITAL+np.cumsum(cd);qty=np.cumsum(qd);eq=cash+qty*closes[start:end+1]
    return {'strategy':f'BASE_DCA{m}','initial_pct':0.,'dca_months':m,'dip_set':'baseline','start':pd.Timestamp(dates[start]),'end':pd.Timestamp(dates[end]),
            'entry_open':float(opens[start]),'final_equity':float(eq[-1]),'return':float(eq[-1]/CAPITAL-1),'mdd':md(eq),'avg_buy_price':float(principal/qtot),
            'cash_end':float(cash[-1]),'buys':len(idxs),'deploy_days':int((dates[idxs[-1]]-dates[start])/np.timedelta64(1,'D')),'dip_fills':0}

def summary(z,label):
    if z.empty:return pd.DataFrame()
    g=z.groupby('strategy',as_index=False).agg(cohorts=('return','size'),median_return=('return','median'),mean_return=('return','mean'),p25_return=('return',lambda x:x.quantile(.25)),
        p10_return=('return',lambda x:x.quantile(.10)),worst_return=('return','min'),median_mdd=('mdd','median'),worst_mdd=('mdd','min'),median_avg_buy=('avg_buy_price','median'),
        median_deploy_days=('deploy_days','median'),median_dip_fills=('dip_fills','median'))
    b=z[z.strategy=='BASE_LUMP'][['start','return']].rename(columns={'return':'lump'})
    q=z.merge(b,on='start',how='left'); w=q.groupby('strategy').apply(lambda x:float((x['return']>x['lump']).mean()),include_groups=False).rename('win_vs_lump').reset_index()
    g=g.merge(w,on='strategy');g.insert(0,'segment',label);return g.sort_values(['median_return','p10_return'],ascending=False).reset_index(drop=True)

def main():
    d=features(fetch()); dates=d.date.values
    gaps=pd.Series(dates).diff().dt.days
    quality={'rows':len(d),'start':str(d.date.iloc[0].date()),'end':str(d.date.iloc[-1].date()),'gaps_gt_1d':int((gaps>1).sum()),'max_gap_days':int(gaps.max())}
    (OUT/'data_quality.json').write_text(json.dumps(quality,indent=2)); d.to_csv(OUT/'ethusdt_daily_used.csv',index=False)
    first=d.groupby(d.date.dt.to_period('M'),as_index=False).head(1).index.tolist()
    starts=[i for i in first if i>=365 and end_index(dates,i) is not None]
    rows=[]
    for si in starts:
        p=d.iloc[si-1]; meta={'signal_date':pd.Timestamp(p.date),'dd365':float(p.dd365),'mom30':float(p.mom30),'mom7':float(p.mom7),'price_vs_ma200':float(p.close/p.ma200-1)}
        r=eval_events(d,si,1.,0,'lump',[]);r.update(meta);rows.append(r)
        for m in [3,6,12]:
            r=pure_dca(d,si,m);r.update(meta);rows.append(r)
        for init in INITIAL:
            for m in MONTHS:
                for name,lev in DIPS.items():
                    r=eval_events(d,si,init,m,name,lev);r.update(meta);rows.append(r)
    cr=pd.DataFrame(rows);cr.to_csv(OUT/'cohort_results.csv',index=False)
    yrs=cr.start.dt.year
    seg={'ALL':cr,'TRAIN_2018_2021':cr[yrs<=2021],'VALID_2022_2023':cr[(yrs>=2022)&(yrs<=2023)],'OOS_2024_PLUS':cr[yrs>=2024],
         'DEEP_DD40':cr[cr.dd365<=-.40],'CURRENTLIKE_LOOSE':cr[(cr.dd365<=-.35)&(cr.mom30>=.10)],'CURRENTLIKE_STRICT':cr[(cr.dd365<=-.40)&(cr.mom30>=.15)]}
    ss=[]
    for name,z in seg.items():
        s=summary(z,name);s.to_csv(OUT/f'summary_{name.lower()}.csv',index=False);ss.append(s)
    a=pd.concat(ss,ignore_index=True);tr=a[a.segment=='TRAIN_2018_2021'];va=a[a.segment=='VALID_2022_2023'];c=tr.merge(va,on='strategy',suffixes=('_tr','_va'))
    for col,asc in [('median_return_tr',False),('p10_return_tr',False),('median_return_va',False),('p10_return_va',False),('median_mdd_va',False)]:c['rank_'+col]=c[col].rank(ascending=asc,method='average',pct=True)
    c['robust_rank_score']=c[[x for x in c if x.startswith('rank_')]].mean(axis=1);c=c.sort_values('robust_rank_score')
    oo=a[a.segment=='OOS_2024_PLUS'][['strategy','cohorts','median_return','p10_return','worst_return','median_mdd','worst_mdd','win_vs_lump']].rename(columns=lambda x:x if x=='strategy' else x+'_oos')
    c=c.merge(oo,on='strategy',how='left');c.head(50).to_csv(OUT/'robust_shortlist_top50.csv',index=False)
    latest=d.iloc[-1];state={'latest_date':str(latest.date.date()),'latest_close':float(latest.close),'dd365':float(latest.dd365),'mom30':float(latest.mom30),'mom7':float(latest.mom7),
      'price_vs_ma200':float(latest.close/latest.ma200-1),'cohorts':len(starts),'deep_dd40_cohorts':int(seg['DEEP_DD40'].start.nunique()),'loose_currentlike_cohorts':int(seg['CURRENTLIKE_LOOSE'].start.nunique()),'strict_currentlike_cohorts':int(seg['CURRENTLIKE_STRICT'].start.nunique())}
    (OUT/'latest_state.json').write_text(json.dumps(state,indent=2))
    lines=['# ETH daily entry grid','',f"Data {quality['start']} to {quality['end']}, rows={quality['rows']}, cohorts={len(starts)}, fee={FEE:.3%}",'', '## Current state','```json',json.dumps(state,indent=2),'```','', '## Robust shortlist']
    cols=['strategy','robust_rank_score','median_return_tr','p10_return_tr','median_return_va','p10_return_va','median_return_oos','p10_return_oos','median_mdd_oos','win_vs_lump_oos'];lines.append(c.head(20)[[x for x in cols if x in c]].to_markdown(index=False))
    for name in ['DEEP_DD40','CURRENTLIKE_LOOSE','CURRENTLIKE_STRICT','OOS_2024_PLUS']:
        s=a[a.segment==name];lines+=['',f'## {name}',s.head(20)[['strategy','cohorts','median_return','p10_return','worst_return','median_mdd','worst_mdd','win_vs_lump']].to_markdown(index=False) if len(s) else 'No cohorts']
    (OUT/'README_results.md').write_text('\n'.join(lines));print('DONE',json.dumps(state),flush=True)
if __name__=='__main__':main()
