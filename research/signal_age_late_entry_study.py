#!/usr/bin/env python3
from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from research.multi_asset_5way_allocation import state_from

OUT=Path('results/signal_age_late_entry'); OUT.mkdir(parents=True,exist_ok=True)
RULES={
    'SPY': {'kind':'yf','ticker':'SPY','ma':250,'confirm':5,'fee':0.0007},
    'QQQ': {'kind':'yf','ticker':'QQQ','ma':250,'confirm':3,'fee':0.0007},
    'BTC': {'kind':'binance','ticker':'BTCUSDT','ma':150,'confirm':3,'fee':0.0005},
    'KOSPI200': {'kind':'yf','ticker':'069500.KS','ma':100,'confirm':3,'fee':0.00015},
}
AGE_EDGES=[0,21,61,121,251,10**9]
AGE_LABELS=['0-20','21-60','61-120','121-250','251+']
CHECKPOINTS=[0,20,60,120,250]
HORIZONS=[63,126,252]


def yfdata(ticker,start):
    x=yf.download(ticker,start=start,auto_adjust=True,progress=False,threads=False)
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    x=x.reset_index().rename(columns={'Date':'date','Open':'open','High':'high','Low':'low','Close':'close'})
    x['date']=pd.to_datetime(x.date).dt.tz_localize(None).dt.normalize()
    for c in ['open','high','low','close']: x[c]=pd.to_numeric(x[c],errors='coerce')
    utc_today=pd.Timestamp.now('UTC').tz_localize(None).normalize()
    return x[x.date<utc_today][['date','open','high','low','close']].dropna().sort_values('date').reset_index(drop=True)


def binance(sym,start='2017-08-01'):
    cur=int(pd.Timestamp(start,tz='UTC').timestamp()*1000); rows=[]
    for base in ['https://data-api.binance.vision/api/v3/klines','https://api.binance.com/api/v3/klines']:
        try:
            rows=[]; q=cur
            while True:
                r=requests.get(base,params={'symbol':sym,'interval':'1d','startTime':q,'limit':1000},timeout=30); r.raise_for_status(); z=r.json()
                if not z: break
                rows+=z; nxt=int(z[-1][0])+86400000
                if len(z)<1000 or nxt<=q: break
                q=nxt; time.sleep(.03)
            if len(rows)>1000: break
        except Exception:
            rows=[]
    if not rows: raise RuntimeError('No Binance data')
    d=pd.DataFrame(rows,columns=['ts','open','high','low','close','v','ct','qv','n','tb','tq','ig'])
    d['date']=pd.to_datetime(d.ts,unit='ms',utc=True).dt.tz_localize(None).dt.normalize()
    for c in ['open','high','low','close']: d[c]=pd.to_numeric(d[c],errors='coerce')
    utc_today=pd.Timestamp.now('UTC').tz_localize(None).normalize()
    return d[d.date<utc_today][['date','open','high','low','close']].dropna().drop_duplicates('date').sort_values('date').reset_index(drop=True)


def prepare_asset(name,cfg):
    if cfg['kind']=='yf':
        start={'SPY':'1993-01-01','QQQ':'1999-01-01','KOSPI200':'2006-01-01'}[name]
        d=yfdata(cfg['ticker'],start)
    else:
        d=binance(cfg['ticker'])
    d['ma']=d.close.rolling(cfg['ma'],min_periods=cfg['ma']).mean()
    d['signal_state']=state_from(d.close,d.ma,cfg['confirm'])
    # State executable at today's open comes only from yesterday's completed close.
    d['exec_state']=np.r_[0,d.signal_state.to_numpy(int)[:-1]]
    # Executable ON episode id and age in native trading days.
    on=d.exec_state.eq(1)
    starts=on & ~on.shift(1,fill_value=False)
    d['episode_id']=starts.cumsum().where(on)
    d['age_td']=np.nan
    d.loc[on,'age_td']=d[on].groupby('episode_id').cumcount().astype(float)
    return d


def simulate_from(d,start_i,horizon,fee):
    """Start fresh at start_i open (must be exec_state=1), then follow future ON/OFF. Cash return=0 to isolate timing."""
    end=min(len(d)-1,start_i+horizon)
    wealth=1.0; qty=0.0; held=False; curve=[]
    for i in range(start_i,end+1):
        want=bool(d.exec_state.iloc[i])
        op=float(d.open.iloc[i]); cl=float(d.close.iloc[i])
        if want!=held:
            if want:
                qty=(wealth*(1-fee))/op; wealth=0.0; held=True
            else:
                wealth=qty*op*(1-fee); qty=0.0; held=False
        curve.append(wealth+qty*cl)
    final=curve[-1]
    peak=np.maximum.accumulate(np.r_[1.0,np.asarray(curve,float)])
    arr=np.r_[1.0,np.asarray(curve,float)]
    return final-1.0,float(np.min(arr/peak-1.0))


def row_for_entry(d,i,fee):
    age=int(d.age_td.iloc[i]); eid=int(d.episode_id.iloc[i])
    future=np.flatnonzero((d.index.to_numpy()>i) & (d.exec_state.to_numpy()==0))
    exit_i=int(future[0]) if len(future) else None
    rec={'entry_date':d.date.iloc[i],'entry_i':i,'episode_id':eid,'age_td':age,'censored':exit_i is None}
    if exit_i is not None:
        entry=float(d.open.iloc[i]); exitp=float(d.open.iloc[exit_i]); path=d.close.iloc[i:exit_i].to_numpy(float)/entry-1.0
        # Approx execution costs at entry and exit; adjusted prices already include distributions for ETFs.
        rec['to_off_return']=(exitp/entry)*(1-fee)**2-1.0
        rec['remaining_td']=exit_i-i
        rec['mae_to_off']=float(min(0.0,np.min(path))) if len(path) else 0.0
        rec['mfe_to_off']=float(max(0.0,np.max(path))) if len(path) else 0.0
        rec['off_date']=d.date.iloc[exit_i]
    else:
        rec.update({'to_off_return':np.nan,'remaining_td':np.nan,'mae_to_off':np.nan,'mfe_to_off':np.nan,'off_date':pd.NaT})
    for h in HORIZONS:
        if i+h < len(d):
            r,m=simulate_from(d,i,h,fee); rec[f'strat_ret_{h}']=r; rec[f'strat_mdd_{h}']=m
        else:
            rec[f'strat_ret_{h}']=np.nan; rec[f'strat_mdd_{h}']=np.nan
    return rec


def age_bucket(a):
    for lo,hi,lab in zip(AGE_EDGES[:-1],AGE_EDGES[1:],AGE_LABELS):
        if lo<=a<hi:return lab
    return AGE_LABELS[-1]


def summarize(g):
    z=g.dropna(subset=['to_off_return'])
    out={
        'entries':len(g),'completed_to_off':len(z),
        'median_to_off':z.to_off_return.median() if len(z) else np.nan,
        'p10_to_off':z.to_off_return.quantile(.10) if len(z) else np.nan,
        'worst_to_off':z.to_off_return.min() if len(z) else np.nan,
        'win_rate_to_off':(z.to_off_return>0).mean() if len(z) else np.nan,
        'median_remaining_td':z.remaining_td.median() if len(z) else np.nan,
        'median_mae_to_off':z.mae_to_off.median() if len(z) else np.nan,
        'p10_mae_to_off':z.mae_to_off.quantile(.10) if len(z) else np.nan,
        'median_mfe_to_off':z.mfe_to_off.median() if len(z) else np.nan,
    }
    for h in HORIZONS:
        q=g.dropna(subset=[f'strat_ret_{h}'])
        out[f'n_{h}']=len(q)
        out[f'median_strat_ret_{h}']=q[f'strat_ret_{h}'].median() if len(q) else np.nan
        out[f'p10_strat_ret_{h}']=q[f'strat_ret_{h}'].quantile(.10) if len(q) else np.nan
        out[f'win_rate_strat_{h}']=(q[f'strat_ret_{h}']>0).mean() if len(q) else np.nan
        out[f'median_strat_mdd_{h}']=q[f'strat_mdd_{h}'].median() if len(q) else np.nan
    return pd.Series(out)


def analyze_asset(name,cfg):
    d=prepare_asset(name,cfg)
    eligible=d.index[(d.exec_state==1)&d.ma.notna()].tolist()
    rows=[row_for_entry(d,i,cfg['fee']) for i in eligible]
    R=pd.DataFrame(rows); R['asset']=name; R['age_bucket']=R.age_td.map(age_bucket); R['period']=np.where(R.entry_date<pd.Timestamp('2018-01-01'),'PRE2018','2018+')
    # Calendar-entry: every possible fresh start while ON.
    cal=R.groupby(['asset','period','age_bucket'],sort=False).apply(summarize,include_groups=False).reset_index()
    full=R.groupby(['asset','age_bucket'],sort=False).apply(summarize,include_groups=False).reset_index(); full.insert(1,'period','FULL')
    cal=pd.concat([cal,full],ignore_index=True)
    # Episode checkpoint: at most one observation per episode per checkpoint.
    cps=[]
    for eid,g in R.groupby('episode_id'):
        for cp in CHECKPOINTS:
            q=g[g.age_td>=cp]
            if len(q):
                r=q.sort_values('age_td').iloc[0].copy(); r['checkpoint']=cp; cps.append(r)
    C=pd.DataFrame(cps)
    if len(C):
        cs=C.groupby(['asset','checkpoint']).apply(summarize,include_groups=False).reset_index()
    else: cs=pd.DataFrame()
    # Current state/age is based on latest completed close signal for next executable open.
    sig=d.signal_state.to_numpy(int); cur=int(sig[-1]); cur_age=None; cur_start=None
    if cur==1:
        j=len(d)-1
        while j>0 and sig[j-1]==1:j-=1
        cur_start=d.date.iloc[j]; cur_age=(len(d)-1)-j
    current={'asset':name,'data_end':str(d.date.iloc[-1].date()),'signal_state_next_open':cur,'signal_start_close_date':str(cur_start.date()) if cur_start is not None else None,'signal_age_native_bars':int(cur_age) if cur_age is not None else None,'last_close':float(d.close.iloc[-1]),'last_ma':float(d.ma.iloc[-1]),'close_over_ma_pct':float(d.close.iloc[-1]/d.ma.iloc[-1]-1) if pd.notna(d.ma.iloc[-1]) else None,'episodes':int(d.episode_id.max()) if d.episode_id.notna().any() else 0}
    return R,cal,C,cs,current


def main():
    raw=[]; sums=[]; cps=[]; cpsums=[]; current=[]
    for name,cfg in RULES.items():
        R,S,C,CS,cur=analyze_asset(name,cfg); raw.append(R); sums.append(S); current.append(cur)
        if len(C):cps.append(C)
        if len(CS):cpsums.append(CS)
        print('\nCURRENT',json.dumps(cur,ensure_ascii=False))
        print('\nCALENDAR',name); print(S[S.period.isin(['FULL','2018+'])].to_string(index=False))
        if len(CS): print('\nCHECKPOINT',name); print(CS.to_string(index=False))
    pd.concat(raw,ignore_index=True).to_csv(OUT/'calendar_entries.csv',index=False)
    pd.concat(sums,ignore_index=True).to_csv(OUT/'age_bucket_summary.csv',index=False)
    if cps: pd.concat(cps,ignore_index=True).to_csv(OUT/'episode_checkpoints.csv',index=False)
    if cpsums: pd.concat(cpsums,ignore_index=True).to_csv(OUT/'episode_checkpoint_summary.csv',index=False)
    pd.DataFrame(current).to_csv(OUT/'current_signal_age.csv',index=False)
    meta={'rules':{k:{'ma':v['ma'],'confirm':v['confirm'],'fee':v['fee']} for k,v in RULES.items()},'age_buckets':AGE_LABELS,'checkpoints':CHECKPOINTS,'horizons_native_bars':HORIZONS,'method':'Signal from completed close; executable state at next native trading-day open. Calendar-entry plus one-checkpoint-per-episode analysis. Forward strategy returns continue following future ON/OFF, with zero cash yield to isolate entry-age effect.'}
    (OUT/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    print('\nCURRENT_ALL\n',pd.DataFrame(current).to_string(index=False))

if __name__=='__main__': main()
