#!/usr/bin/env python3
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from research.multi_asset_5way_allocation import state_from

OUT=Path('results/signal_age_late_entry'); OUT.mkdir(parents=True,exist_ok=True)
RULES={
 'SPY':{'kind':'yf','ticker':'SPY','ma':250,'confirm':5,'fee':.0007,'start':'1993-01-01'},
 'QQQ':{'kind':'yf','ticker':'QQQ','ma':250,'confirm':3,'fee':.0007,'start':'1999-01-01'},
 'BTC':{'kind':'binance','ticker':'BTCUSDT','ma':150,'confirm':3,'fee':.0005,'start':'2017-08-01'},
 'KOSPI200':{'kind':'yf','ticker':'069500.KS','ma':100,'confirm':3,'fee':.00015,'start':'2006-01-01'},
}
AGE_LABELS=['0-20','21-60','61-120','121-250','251+']; CHECKPOINTS=[0,20,60,120,250]; HORIZONS=[63,126,252]

def age_bucket(a):
    if a<=20:return '0-20'
    if a<=60:return '21-60'
    if a<=120:return '61-120'
    if a<=250:return '121-250'
    return '251+'

def yfdata(ticker,start):
    x=yf.download(ticker,start=start,auto_adjust=True,progress=False,threads=False)
    if isinstance(x.columns,pd.MultiIndex):x.columns=x.columns.get_level_values(0)
    x=x.reset_index().rename(columns={'Date':'date','Open':'open','High':'high','Low':'low','Close':'close'})
    x.date=pd.to_datetime(x.date).dt.tz_localize(None).dt.normalize()
    for c in ['open','high','low','close']:x[c]=pd.to_numeric(x[c],errors='coerce')
    today=pd.Timestamp.now('UTC').tz_localize(None).normalize()
    return x[x.date<today][['date','open','high','low','close']].dropna().sort_values('date').reset_index(drop=True)

def binance(sym,start):
    q=int(pd.Timestamp(start,tz='UTC').timestamp()*1000); rows=[]
    for base in ['https://data-api.binance.vision/api/v3/klines','https://api.binance.com/api/v3/klines']:
      try:
        rows=[]; cur=q
        while True:
            r=requests.get(base,params={'symbol':sym,'interval':'1d','startTime':cur,'limit':1000},timeout=30);r.raise_for_status();z=r.json()
            if not z:break
            rows+=z;nxt=int(z[-1][0])+86400000
            if len(z)<1000 or nxt<=cur:break
            cur=nxt;time.sleep(.02)
        if len(rows)>1000:break
      except Exception:rows=[]
    if not rows:raise RuntimeError('BTC download failed')
    d=pd.DataFrame(rows,columns=['ts','open','high','low','close','v','ct','qv','n','tb','tq','ig'])
    d.date=pd.to_datetime(d.ts,unit='ms',utc=True).dt.tz_localize(None).dt.normalize()
    for c in ['open','high','low','close']:d[c]=pd.to_numeric(d[c],errors='coerce')
    today=pd.Timestamp.now('UTC').tz_localize(None).normalize()
    return d[d.date<today][['date','open','high','low','close']].dropna().drop_duplicates('date').sort_values('date').reset_index(drop=True)

def prep(name,c):
    d=yfdata(c['ticker'],c['start']) if c['kind']=='yf' else binance(c['ticker'],c['start'])
    d['ma']=d.close.rolling(c['ma'],min_periods=c['ma']).mean();d['signal_state']=state_from(d.close,d.ma,c['confirm'])
    d['exec_state']=np.r_[0,d.signal_state.to_numpy(int)[:-1]]
    on=d.exec_state.eq(1);st=on & ~on.shift(1,fill_value=False);d['episode_id']=st.cumsum().where(on)
    d['age_td']=np.nan;d.loc[on,'age_td']=d[on].groupby('episode_id').cumcount().astype(float)
    return d

def standard_growth(d,fee):
    n=len(d);g=np.ones(n);s=d.exec_state.to_numpy(int);op=d.open.to_numpy(float);cl=d.close.to_numpy(float)
    for i in range(1,n):
        if s[i-1] and s[i]:g[i]=cl[i]/cl[i-1]
        elif s[i-1] and not s[i]:g[i]=op[i]*(1-fee)/cl[i-1]
        elif (not s[i-1]) and s[i]:g[i]=(1-fee)*cl[i]/op[i]
        else:g[i]=1.0
    return g

def build_entries(d,fee):
    n=len(d);s=d.exec_state.to_numpy(int);op=d.open.to_numpy(float);cl=d.close.to_numpy(float)
    g=standard_growth(d,fee);logp=np.zeros(n);logp[1:]=np.cumsum(np.log(np.maximum(g[1:],1e-300)))
    records=[]
    for eid,ixs in d[d.exec_state==1].groupby('episode_id').groups.items():
        ix=np.array(sorted(ixs),int);first,last=ix[0],ix[-1];exit_i=last+1 if last+1<n and s[last+1]==0 else None
        # suffix price extrema within the remaining ON episode, enabling MAE/MFE for every late entry without O(n^2).
        closes=cl[ix];smin=np.minimum.accumulate(closes[::-1])[::-1];smax=np.maximum.accumulate(closes[::-1])[::-1]
        for k,i in enumerate(ix):
            rec={'entry_date':d.date.iloc[i],'entry_i':i,'episode_id':int(eid),'age_td':int(d.age_td.iloc[i]),'censored':exit_i is None}
            if exit_i is not None:
                rec.update({'off_date':d.date.iloc[exit_i],'remaining_td':exit_i-i,'to_off_return':(op[exit_i]/op[i])*(1-fee)**2-1,
                            'mae_to_off':min(0.0,smin[k]/op[i]-1),'mfe_to_off':max(0.0,smax[k]/op[i]-1)})
            else:rec.update({'off_date':pd.NaT,'remaining_td':np.nan,'to_off_return':np.nan,'mae_to_off':np.nan,'mfe_to_off':np.nan})
            startfac=(1-fee)*cl[i]/op[i]
            for h in HORIZONS:
                j=i+h
                if j<n:
                    factor=startfac*np.exp(logp[j]-logp[i]);rec[f'strat_ret_{h}']=factor-1
                else:rec[f'strat_ret_{h}']=np.nan
            records.append(rec)
    return pd.DataFrame(records)

def summarize(g):
    z=g.dropna(subset=['to_off_return']);o={'entries':len(g),'completed_to_off':len(z),'episodes':g.episode_id.nunique()}
    for key,fun in [('median_to_off',lambda x:x.median()),('p10_to_off',lambda x:x.quantile(.1)),('worst_to_off',lambda x:x.min())]:o[key]=fun(z.to_off_return) if len(z) else np.nan
    o['win_rate_to_off']=(z.to_off_return>0).mean() if len(z) else np.nan;o['median_remaining_td']=z.remaining_td.median() if len(z) else np.nan
    o['median_mae_to_off']=z.mae_to_off.median() if len(z) else np.nan;o['p10_mae_to_off']=z.mae_to_off.quantile(.1) if len(z) else np.nan;o['median_mfe_to_off']=z.mfe_to_off.median() if len(z) else np.nan
    for h in HORIZONS:
        q=g.dropna(subset=[f'strat_ret_{h}']);o[f'n_{h}']=len(q);o[f'median_strat_ret_{h}']=q[f'strat_ret_{h}'].median() if len(q) else np.nan;o[f'p10_strat_ret_{h}']=q[f'strat_ret_{h}'].quantile(.1) if len(q) else np.nan;o[f'win_rate_strat_{h}']=(q[f'strat_ret_{h}']>0).mean() if len(q) else np.nan
    return pd.Series(o)

def analyze(name,c):
    d=prep(name,c);R=build_entries(d,c['fee']);R['asset']=name;R['age_bucket']=R.age_td.map(age_bucket);R['period']=np.where(R.entry_date<pd.Timestamp('2018-01-01'),'PRE2018','2018+')
    S=R.groupby(['asset','period','age_bucket'],sort=False).apply(summarize,include_groups=False).reset_index();F=R.groupby(['asset','age_bucket'],sort=False).apply(summarize,include_groups=False).reset_index();F.insert(1,'period','FULL');S=pd.concat([S,F],ignore_index=True)
    cp=[]
    for eid,g in R.groupby('episode_id'):
        for x in CHECKPOINTS:
            q=g[g.age_td>=x]
            if len(q):r=q.sort_values('age_td').iloc[0].copy();r['checkpoint']=x;cp.append(r)
    C=pd.DataFrame(cp);CS=C.groupby(['asset','checkpoint']).apply(summarize,include_groups=False).reset_index() if len(C) else pd.DataFrame()
    sig=d.signal_state.to_numpy(int);cur=int(sig[-1]);start=None;age=None
    if cur:
        j=len(d)-1
        while j>0 and sig[j-1]==1:j-=1
        start=d.date.iloc[j];age=len(d)-1-j
    current={'asset':name,'data_end':str(d.date.iloc[-1].date()),'signal_state_next_open':cur,'signal_start_close_date':str(start.date()) if start is not None else None,'signal_age_native_bars':int(age) if age is not None else None,'age_bucket':age_bucket(age) if age is not None else None,'last_close':float(d.close.iloc[-1]),'last_ma':float(d.ma.iloc[-1]),'close_over_ma_pct':float(d.close.iloc[-1]/d.ma.iloc[-1]-1),'episodes':int(d.episode_id.max()) if d.episode_id.notna().any() else 0}
    return R,S,C,CS,current

def main():
    rr=[];ss=[];cc=[];cs=[];cur=[]
    for a,c in RULES.items():
        R,S,C,CS,U=analyze(a,c);rr.append(R);ss.append(S);cur.append(U)
        if len(C):cc.append(C)
        if len(CS):cs.append(CS)
        print('\nCURRENT',json.dumps(U,ensure_ascii=False));print('\nSUMMARY',a);print(S[S.period.isin(['FULL','2018+'])].to_string(index=False));print('\nCHECKPOINT');print(CS.to_string(index=False))
    pd.concat(rr).to_csv(OUT/'calendar_entries.csv',index=False);pd.concat(ss).to_csv(OUT/'age_bucket_summary.csv',index=False)
    if cc:pd.concat(cc).to_csv(OUT/'episode_checkpoints.csv',index=False)
    if cs:pd.concat(cs).to_csv(OUT/'episode_checkpoint_summary.csv',index=False)
    pd.DataFrame(cur).to_csv(OUT/'current_signal_age.csv',index=False)
    meta={'rules':{a:{k:v for k,v in c.items() if k in ['ma','confirm','fee']} for a,c in RULES.items()},'age_buckets':AGE_LABELS,'checkpoints':CHECKPOINTS,'horizons_native_bars':HORIZONS,'method':'Exact existing state_from confirmation logic. Completed close -> next native open execution. Calendar-entry plus one observation per episode checkpoint. Forward strategy returns continue future ON/OFF with cash yield set to zero to isolate signal-age effect.'};(OUT/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    print('\nCURRENT_ALL\n',pd.DataFrame(cur).to_string(index=False))
if __name__=='__main__':main()
