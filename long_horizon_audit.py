from pathlib import Path
import numpy as np
import pandas as pd

CAP=10_000_000.0
FEE_BPS=5.0
FEE=FEE_BPS/10000.0
HORIZONS=[7,10,12,15]


def buy_into(q, avg, cash_amt, price):
    inv=cash_amt/(1+FEE)
    add=inv/price
    new_avg=price if q<=0 else (q*avg+add*price)/(q+add)
    return q+add,new_avg,cash_amt


def simulate_time20(w, mode):
    # mode: HOLD90, CORE80_TP50, CORE80_SPLIT40_50
    protected=CAP*0.10
    entry_cash=CAP-protected
    tranche=entry_cash/20.0
    core_q=0.0; core_avg=0.0
    t1_q=0.0; t1_avg=0.0; t1_cash=0.0; t1_sale=0.0
    t2_q=0.0; t2_avg=0.0; t2_cash=0.0; t2_sale=0.0
    eq=[]; trades=0
    for i,r in w.reset_index(drop=True).iterrows():
        op=float(r.Open); hi=float(r.High); lo=float(r.Low); cl=float(r.Close)
        # Existing trading sleeves only: no same-day sell/rebuy and no use of close entry before intraday high/low.
        if mode=='CORE80_TP50':
            if t1_q>1e-12 and t1_avg>0:
                target=t1_avg*1.50
                if hi>=target:
                    t1_cash+=t1_q*target*(1-FEE); t1_q=0.; t1_avg=0.; t1_sale=target; trades+=1
            elif t1_cash>1e-12 and t1_sale>0:
                rb=t1_sale*0.80
                if lo<=rb:
                    inv=t1_cash/(1+FEE); add=inv/rb
                    t1_q=add; t1_avg=rb; t1_cash-=inv*(1+FEE); t1_sale=0.; trades+=1
        elif mode=='CORE80_SPLIT40_50':
            for sleeve in (1,2):
                q=t1_q if sleeve==1 else t2_q
                avg=t1_avg if sleeve==1 else t2_avg
                cash=t1_cash if sleeve==1 else t2_cash
                sale=t1_sale if sleeve==1 else t2_sale
                tp=0.40 if sleeve==1 else 0.50
                acted=False
                if q>1e-12 and avg>0:
                    target=avg*(1+tp)
                    if hi>=target:
                        cash+=q*target*(1-FEE); q=0.; avg=0.; sale=target; trades+=1; acted=True
                if (not acted) and q<=1e-12 and cash>1e-12 and sale>0:
                    rb=sale*0.80
                    if lo<=rb:
                        inv=cash/(1+FEE); q=inv/rb; avg=rb; cash-=inv*(1+FEE); sale=0.; trades+=1
                if sleeve==1: t1_q,t1_avg,t1_cash,t1_sale=q,avg,cash,sale
                else: t2_q,t2_avg,t2_cash,t2_sale=q,avg,cash,sale

        # Time split entry at the close, after all intraday events.
        if i<20 and entry_cash>1e-8:
            gross=min(tranche,entry_cash)
            if mode=='HOLD90':
                c_amt=gross; a1=a2=0.
            elif mode=='CORE80_TP50':
                c_amt=gross*0.80; a1=gross*0.20; a2=0.
            else:
                c_amt=gross*0.80; a1=gross*0.10; a2=gross*0.10
            if c_amt>0:
                inv=c_amt/(1+FEE); add=inv/cl
                core_avg=cl if core_q<=0 else (core_q*core_avg+add*cl)/(core_q+add)
                core_q+=add; trades+=1
            if a1>0:
                inv=a1/(1+FEE); add=inv/cl
                t1_avg=cl if t1_q<=0 else (t1_q*t1_avg+add*cl)/(t1_q+add)
                t1_q+=add; trades+=1
            if a2>0:
                inv=a2/(1+FEE); add=inv/cl
                t2_avg=cl if t2_q<=0 else (t2_q*t2_avg+add*cl)/(t2_q+add)
                t2_q+=add; trades+=1
            entry_cash-=gross
        value=protected+entry_cash+t1_cash+t2_cash+(core_q+t1_q+t2_q)*cl
        eq.append(value)
    eq=np.asarray(eq,float)
    ret=eq[-1]/CAP-1
    years=(w.Date.iloc[-1]-w.Date.iloc[0]).days/365.2425
    cagr=(eq[-1]/CAP)**(1/years)-1
    mdd=np.min(eq/np.maximum.accumulate(eq)-1)
    return ret,cagr,mdd,trades


def bh(w):
    inv=CAP/(1+FEE)
    q=inv/w.Close.iloc[0]
    eq=q*w.Close.values
    ret=eq[-1]/CAP-1
    years=(w.Date.iloc[-1]-w.Date.iloc[0]).days/365.2425
    cagr=(eq[-1]/CAP)**(1/years)-1
    mdd=np.min(eq/np.maximum.accumulate(eq)-1)
    return ret,cagr,mdd,1


def make_windows(dates, years, monthly=True):
    starts=(pd.date_range(dates.min().to_period('M').start_time,dates.max(),freq='MS') if monthly
            else [pd.Timestamp(f'{y}-01-01') for y in range(dates.min().year,dates.max().year+1)])
    arr=dates.to_numpy(); out=[]; seen=set()
    for s in starts:
        si=int(np.searchsorted(arr,np.datetime64(s),side='left'))
        if si>=len(dates): continue
        actual=dates.iloc[si]
        if actual in seen: continue
        seen.add(actual)
        target=actual+pd.DateOffset(years=years)
        if dates.iloc[-1]<target: continue
        ei=int(np.searchsorted(arr,np.datetime64(target),side='right'))-1
        if ei-si+1 < 200*years: continue
        out.append((si,ei,actual,dates.iloc[ei]))
    return out


def summarize(rows, years, cadence):
    d=pd.DataFrame(rows)
    out=[]
    for strategy,g in d.groupby('strategy'):
        out.append(dict(horizon=years,cadence=cadence,strategy=strategy,n=len(g),
                        med_total=g.ret.median(),med_cagr=g.cagr.median(),med_mdd=g.mdd.median(),
                        p25_total=g.ret.quantile(.25),p10_total=g.ret.quantile(.10),
                        loss_rate=(g.ret<0).mean(),worst_total=g.ret.min(),worst_mdd=g.mdd.min()))
    return pd.DataFrame(out),d


def main():
    o=Path('results_long');o.mkdir(exist_ok=True)
    L=pd.read_csv('results/synthetic2x.csv',parse_dates=['Date'])
    strategies=['BH','HOLD90','CORE80_TP50','CORE80_SPLIT40_50']
    allsum=[]
    for y in HORIZONS:
        for monthly in [True,False]:
            wins=make_windows(L.Date,y,monthly)
            rows=[]
            for si,ei,s,e in wins:
                w=L.iloc[si:ei+1].reset_index(drop=True)
                for st in strategies:
                    r=bh(w) if st=='BH' else simulate_time20(w,st)
                    rows.append(dict(start=s,end=e,strategy=st,ret=r[0],cagr=r[1],mdd=r[2],trades=r[3]))
            cadence='monthly' if monthly else 'annual'
            sm,detail=summarize(rows,y,cadence)
            allsum.append(sm)
            detail.to_csv(o/f'{cadence}_{y}y_details.csv',index=False)
            print(f'\n=== {cadence.upper()} {y}Y windows={len(wins)} ===')
            print(sm[['strategy','n','med_total','med_cagr','med_mdd','p25_total','p10_total','loss_rate','worst_total','worst_mdd']].to_string(index=False))
    summary=pd.concat(allsum,ignore_index=True)
    summary.to_csv(o/'long_horizon_summary.csv',index=False)

    # Full available history single-start comparison.
    full=[]
    for st in strategies:
        r=bh(L) if st=='BH' else simulate_time20(L,st)
        full.append(dict(strategy=st,start=L.Date.iloc[0],end=L.Date.iloc[-1],years=(L.Date.iloc[-1]-L.Date.iloc[0]).days/365.2425,
                         total_return=r[0],cagr=r[1],mdd=r[2],trades=r[3],final=CAP*(1+r[0])))
    full=pd.DataFrame(full)
    full.to_csv(o/'full_period_comparison.csv',index=False)
    print('\n=== FULL PERIOD ===')
    print(full.to_string(index=False))

if __name__=='__main__':
    main()
