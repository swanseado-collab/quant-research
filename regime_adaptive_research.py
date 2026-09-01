from pathlib import Path
import numpy as np
import pandas as pd

CAP=10_000_000.0
FEE=5/10000.0
RESERVE=0.10

# All adaptive decisions use yesterday's regime only.
# Regime is based on 60d realized vol percentile versus the PRIOR 3y (756d) history.

def build_features(u):
    z=u[['Date','Close']].copy()
    z['ret']=z.Close.pct_change()
    z['rv60']=z.ret.rolling(60,min_periods=40).std()*np.sqrt(252)
    z['peak252']=z.Close.rolling(252,min_periods=60).max()
    z['dd252']=z.Close/z.peak252-1
    rv=z.rv60.to_numpy(float)
    pct=np.full(len(z),np.nan)
    for i in range(len(z)):
        if not np.isfinite(rv[i]): continue
        hist=rv[max(0,i-756):i]  # strictly prior observations
        hist=hist[np.isfinite(hist)]
        if len(hist)>=120:
            pct[i]=np.mean(hist<=rv[i])
    z['vp3y']=pct
    return z


def regime(vp):
    if not np.isfinite(vp): return 0
    if vp>=0.95: return 2
    if vp>=0.75: return 1
    return 0

CANDIDATES={
    # fixed baselines
    'HOLD90': dict(kind='hold'),
    'STATIC80_TP50_RB20': dict(kind='static',trade=[.20,.20,.20],tp=[.50,.50,.50],rb=[.20,.20,.20],crash_pause=False),
    'STATIC70_TP40_RB20': dict(kind='static',trade=[.30,.30,.30],tp=[.40,.40,.40],rb=[.20,.20,.20],crash_pause=False),
    # adaptive A: keep 80/20 but widen trading bands as volatility rises
    'ADAPT_BANDS_80': dict(kind='adapt',trade=[.20,.20,.20],tp=[.40,.50,.60],rb=[.15,.20,.25],crash_pause=False),
    # adaptive B: normal regime uses more trading; high/extreme regime pushes new capital into core
    'ADAPT_CORE_70_80_90': dict(kind='adapt',trade=[.30,.20,.10],tp=[.40,.50,.60],rb=[.15,.20,.25],crash_pause=False),
    # adaptive C: same allocation, but in extreme-vol + <=-30% drawdown, stop profit-taking; allow re-entry
    'ADAPT_CRASH_HOLD': dict(kind='adapt',trade=[.30,.20,.10],tp=[.40,.50,.60],rb=[.15,.20,.25],crash_pause=True),
}


def simulate(wL,wF,cfg):
    protected=CAP*RESERVE
    entry_cash=CAP-protected
    tranche=entry_cash/20.0
    core_q=trade_q=0.0
    core_avg=trade_avg=0.0
    trade_cash=0.0
    last_sale=0.0
    eq=[]; trades=0; reg_days=[0,0,0]; paused_days=0
    for i,r in wL.reset_index(drop=True).iterrows():
        op=float(r.Open); hi=float(r.High); lo=float(r.Low); cl=float(r.Close)
        # Only yesterday's close-derived features are available for today's actions.
        if i==0:
            rg=0; prev_dd=0.0
        else:
            pf=wF.iloc[i-1]
            rg=regime(float(pf.vp3y)); prev_dd=float(pf.dd252) if np.isfinite(pf.dd252) else 0.0
        reg_days[rg]+=1
        crash=(rg==2 and prev_dd<=-0.30)

        if cfg['kind']!='hold':
            tp=cfg['tp'][rg]; rb=cfg['rb'][rg]
            # Existing trade sleeve: intraday event first. No same-day sell/rebuy.
            sold=False
            allow_tp=not (cfg.get('crash_pause',False) and crash)
            if not allow_tp: paused_days+=1
            if trade_q>1e-12 and trade_avg>0 and allow_tp:
                target=trade_avg*(1+tp)
                if hi>=target:
                    trade_cash += trade_q*target*(1-FEE)
                    trade_q=0.; trade_avg=0.; last_sale=target; trades+=1; sold=True
            if (not sold) and trade_q<=1e-12 and trade_cash>1e-12 and last_sale>0:
                reprice=last_sale*(1-rb)
                if lo<=reprice:
                    inv=trade_cash/(1+FEE); add=inv/reprice
                    trade_q=add; trade_avg=reprice
                    trade_cash-=inv*(1+FEE); last_sale=0.; trades+=1

        # Initial 20-day close entry; allocation uses yesterday's regime.
        if i<20 and entry_cash>1e-8:
            gross=min(tranche,entry_cash)
            if cfg['kind']=='hold': tshare=0.0
            else: tshare=cfg['trade'][rg]
            tamt=gross*tshare; camt=gross-tamt
            if camt>0:
                inv=camt/(1+FEE); add=inv/cl
                core_avg=cl if core_q<=0 else (core_q*core_avg+add*cl)/(core_q+add)
                core_q+=add; trades+=1
            if tamt>0:
                inv=tamt/(1+FEE); add=inv/cl
                trade_avg=cl if trade_q<=0 else (trade_q*trade_avg+add*cl)/(trade_q+add)
                trade_q+=add; trades+=1
            entry_cash-=gross
        value=protected+entry_cash+trade_cash+(core_q+trade_q)*cl
        eq.append(value)
    eq=np.asarray(eq,float)
    years=(wL.Date.iloc[-1]-wL.Date.iloc[0]).days/365.2425
    ret=eq[-1]/CAP-1
    cagr=(eq[-1]/CAP)**(1/years)-1 if years>0 else np.nan
    mdd=np.min(eq/np.maximum.accumulate(eq)-1)
    return dict(ret=ret,cagr=cagr,mdd=mdd,trades=trades,
                normal_share=reg_days[0]/len(wL),high_share=reg_days[1]/len(wL),extreme_share=reg_days[2]/len(wL),paused_days=paused_days)


def bh(w):
    inv=CAP/(1+FEE); q=inv/w.Close.iloc[0]; eq=q*w.Close.to_numpy(float)
    years=(w.Date.iloc[-1]-w.Date.iloc[0]).days/365.2425
    return dict(ret=eq[-1]/CAP-1,cagr=(eq[-1]/CAP)**(1/years)-1,mdd=np.min(eq/np.maximum.accumulate(eq)-1),trades=1)


def make_windows(dates,years,start=None,end=None,monthly=True):
    d=dates
    if start is None: start=d.iloc[0]
    if end is None: end=d.iloc[-1]
    if monthly:
        starts=pd.date_range(pd.Timestamp(start).to_period('M').start_time,pd.Timestamp(end),freq='MS')
    else:
        starts=[pd.Timestamp(f'{y}-01-01') for y in range(pd.Timestamp(start).year,pd.Timestamp(end).year+1)]
    arr=d.to_numpy(); out=[];seen=set()
    for s in starts:
        si=int(np.searchsorted(arr,np.datetime64(s),'left'))
        if si>=len(d): continue
        actual=d.iloc[si]
        if actual in seen or actual<pd.Timestamp(start) or actual>pd.Timestamp(end): continue
        seen.add(actual)
        target=actual+pd.DateOffset(years=years)
        if target>pd.Timestamp(end) or d.iloc[-1]<target: continue
        ei=int(np.searchsorted(arr,np.datetime64(target),'right'))-1
        if ei-si+1>=200*years: out.append((si,ei,actual,d.iloc[ei]))
    return out


def eval_windows(L,F,wins,label):
    rows=[]
    for si,ei,s,e in wins:
        wL=L.iloc[si:ei+1].reset_index(drop=True); wF=F.iloc[si:ei+1].reset_index(drop=True)
        b=bh(wL); rows.append(dict(window=label,start=s,end=e,strategy='BH',**b))
        for name,cfg in CANDIDATES.items():
            rows.append(dict(window=label,start=s,end=e,strategy=name,**simulate(wL,wF,cfg)))
    return pd.DataFrame(rows)


def summarize(d):
    return d.groupby(['window','strategy'],as_index=False).agg(
        n=('ret','size'),med_ret=('ret','median'),med_cagr=('cagr','median'),med_mdd=('mdd','median'),
        p25_ret=('ret',lambda x:x.quantile(.25)),p10_ret=('ret',lambda x:x.quantile(.10)),
        loss_rate=('ret',lambda x:(x<0).mean()),worst_ret=('ret','min'),worst_mdd=('mdd','min'),med_trades=('trades','median'))


def period_result(L,F,start,end,label):
    m=(L.Date>=pd.Timestamp(start))&(L.Date<=pd.Timestamp(end))
    wL=L.loc[m].reset_index(drop=True); wF=F.loc[m].reset_index(drop=True)
    rows=[dict(period=label,start=wL.Date.iloc[0],end=wL.Date.iloc[-1],strategy='BH',**bh(wL))]
    for name,cfg in CANDIDATES.items(): rows.append(dict(period=label,start=wL.Date.iloc[0],end=wL.Date.iloc[-1],strategy=name,**simulate(wL,wF,cfg)))
    return pd.DataFrame(rows)


def start_regime_detail(d,F,L,years=1):
    # Monthly starts, grouped by the regime known at the prior close.
    wins=make_windows(L.Date,years,monthly=True)
    rows=[]
    for si,ei,s,e in wins:
        j=max(si-1,0); rg=regime(float(F.vp3y.iloc[j])); dd=float(F.dd252.iloc[j]) if np.isfinite(F.dd252.iloc[j]) else 0
        group='EXTREME_CRASH' if rg==2 and dd<=-.30 else ('EXTREME' if rg==2 else ('HIGH' if rg==1 else 'NORMAL'))
        wL=L.iloc[si:ei+1].reset_index(drop=True); wF=F.iloc[si:ei+1].reset_index(drop=True)
        for name in ['BH','STATIC80_TP50_RB20','ADAPT_BANDS_80','ADAPT_CORE_70_80_90','ADAPT_CRASH_HOLD']:
            r=bh(wL) if name=='BH' else simulate(wL,wF,CANDIDATES[name])
            rows.append(dict(start=s,group=group,strategy=name,ret=r['ret'],mdd=r['mdd']))
    q=pd.DataFrame(rows)
    sm=q.groupby(['group','strategy'],as_index=False).agg(n=('ret','size'),med_ret=('ret','median'),med_mdd=('mdd','median'),loss_rate=('ret',lambda x:(x<0).mean()))
    return q,sm


def main():
    out=Path('results_regime');out.mkdir(exist_ok=True)
    u=pd.read_csv('results/underlying.csv',parse_dates=['Date'])
    L=pd.read_csv('results/synthetic2x.csv',parse_dates=['Date'])
    F=build_features(u)
    F.to_csv(out/'causal_regime_features.csv',index=False)

    # Full rolling windows, plus training-only windows that end by 2023-12-29.
    details=[]
    for y in [1,2,3,5,7,10]:
        w=make_windows(L.Date,y,monthly=True)
        details.append(eval_windows(L,F,w,f'ALL_{y}Y'))
        wt=make_windows(L.Date,y,start='2010-01-04',end='2023-12-29',monthly=True)
        details.append(eval_windows(L,F,wt,f'TRAIN_PRE2024_{y}Y'))
    det=pd.concat(details,ignore_index=True); det.to_csv(out/'rolling_details.csv',index=False)
    sm=summarize(det); sm.to_csv(out/'rolling_summary.csv',index=False)

    periods=pd.concat([
        period_result(L,F,'2010-01-04','2023-12-29','PRE2024'),
        period_result(L,F,'2024-01-02','2026-08-07','OOS_2024_2026'),
        period_result(L,F,'2025-01-02','2026-08-07','NEW_REGIME_2025_2026'),
        period_result(L,F,'2010-01-04','2026-08-07','FULL'),
    ],ignore_index=True)
    periods.to_csv(out/'period_comparison.csv',index=False)

    q,qs=start_regime_detail(det,F,L,1);q.to_csv(out/'start_regime_1y_details.csv',index=False);qs.to_csv(out/'start_regime_1y_summary.csv',index=False)

    # Point-in-time regime diagnostics.
    diag=[]
    for dt in ['2011-09-01','2020-03-20','2024-01-02','2025-01-02','2026-06-22','2026-07-30','2026-08-07']:
        z=F[F.Date<=pd.Timestamp(dt)]
        if len(z):
            r=z.iloc[-1]; rg=regime(float(r.vp3y)); diag.append(dict(date=r.Date,close=r.Close,rv60=r.rv60,vp3y=r.vp3y,dd252=r.dd252,regime=['NORMAL','HIGH','EXTREME'][rg],crash=(rg==2 and r.dd252<=-.30)))
    dg=pd.DataFrame(diag);dg.to_csv(out/'regime_diagnostics.csv',index=False)

    print('\n=== POINT-IN-TIME REGIMES ===');print(dg.to_string(index=False))
    print('\n=== PRE-2024 TRAIN PERIOD ===');print(periods[periods.period=='PRE2024'][['strategy','ret','cagr','mdd','trades']].sort_values('cagr',ascending=False).to_string(index=False))
    print('\n=== OOS 2024-2026 ===');print(periods[periods.period=='OOS_2024_2026'][['strategy','ret','cagr','mdd','trades']].sort_values('ret',ascending=False).to_string(index=False))
    print('\n=== NEW REGIME 2025-2026 ===');print(periods[periods.period=='NEW_REGIME_2025_2026'][['strategy','ret','cagr','mdd','trades']].sort_values('ret',ascending=False).to_string(index=False))
    print('\n=== TRAIN ROLLING 3Y/5Y ===');print(sm[(sm.window.isin(['TRAIN_PRE2024_3Y','TRAIN_PRE2024_5Y']))][['window','strategy','n','med_cagr','med_mdd','loss_rate']].to_string(index=False))
    print('\n=== START REGIME 1Y ===');print(qs.to_string(index=False))

if __name__=='__main__': main()
