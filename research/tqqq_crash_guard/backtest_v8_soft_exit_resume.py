from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from backtest_v1 import load_panel, execute_target, perf, period_metrics
from backtest_v2 import add_features
from backtest_v3 import rolling_metrics
from backtest_v6_bull_hold import cycle_2022_metrics

OUT=Path(__file__).resolve().parent/'results_v8_soft_exit_resume'
OUT.mkdir(parents=True,exist_ok=True)

class Cfg:
    def __init__(self, exit_ma, exit_slope_lb, exit_dd, resume_ma):
        self.arm_dd=-.25
        self.entry_ma=80
        self.full_ma=110
        self.starter_w=.35
        self.exit_ma=int(exit_ma)
        self.exit_slope_lb=int(exit_slope_lb)
        self.exit_dd=float(exit_dd)
        self.resume_ma=int(resume_ma)
        self.entry_slope_lb=20
        self.confirm_days=3
    @property
    def name(self):
        return f'V8_X{self.exit_ma}S{self.exit_slope_lb}D{abs(int(self.exit_dd*100))}_R{self.resume_ma}'

def features(p):
    p=p.copy()
    for ma in (60,80,100,110,120):
        p[f'ma{ma}']=p.qqq.rolling(ma,min_periods=ma).mean()
        p[f'above_ma{ma}_3d']=(p.qqq>p[f'ma{ma}']).rolling(3,min_periods=3).sum().eq(3)
    if 'hi252' not in p:
        p['hi252']=p.qqq.rolling(252,min_periods=252).max()
    if 'dd252' not in p:
        p['dd252']=p.qqq/p.hi252-1
    return p

def simulate(p,cfg,save=False):
    q=p.qqq.to_numpy(float); t=p.tqqq.to_numpy(float); dd=p.dd252.to_numpy(float)
    dates=pd.DatetimeIndex(p.date); cy=p.cash_yield_pct.to_numpy(float)
    ema=p[f'ma{cfg.entry_ma}'].to_numpy(float); fma=p[f'ma{cfg.full_ma}'].to_numpy(float)
    xma=p[f'ma{cfg.exit_ma}'].to_numpy(float); rma=p[f'ma{cfg.resume_ma}'].to_numpy(float)
    e3=p[f'above_ma{cfg.entry_ma}_3d'].to_numpy(bool); r3=p[f'above_ma{cfg.resume_ma}_3d'].to_numpy(bool)

    cash=1.; sh=0.; avg=np.nan; pending=None; reason=None
    crash_armed=False; resume_armed=False; stage=0
    eqs=[]; ws=[]; states=[]; trades=[]

    for i in range(len(p)):
        if i>0: cash*=1+max(0.,cy[i-1]/100)/252
        if pending is not None:
            before=sh
            cash,sh,avg,notional,fee=execute_target(cash,sh,avg,t[i],pending)
            if abs(notional)>1e-12:
                trades.append((dates[i],reason,pending,t[i],notional,fee,cash,sh,avg))
            if before>0 and sh==0:
                stage=0; resume_armed=True
            elif sh>0:
                stage=2 if pending>=.999 else 1
                resume_armed=False
        pending=None; reason=None

        eq=cash+sh*t[i]; w=sh*t[i]/eq if eq>0 else 0.
        eqs.append(eq); ws.append(w)
        if np.isfinite(dd[i]) and dd[i]<=cfg.arm_dd: crash_armed=True

        if sh>0:
            exit_sig=False
            if i>=cfg.exit_slope_lb and np.isfinite(xma[i]) and np.isfinite(xma[i-cfg.exit_slope_lb]):
                exit_sig=(q[i]<xma[i] and xma[i]<xma[i-cfg.exit_slope_lb] and np.isfinite(dd[i]) and dd[i]<=cfg.exit_dd)
            if exit_sig:
                pending=0.; reason='SOFT_EXIT'; states.append('EXIT'); continue
            states.append('STARTER' if stage==1 else 'BULL')
            if stage==1 and i>=cfg.entry_slope_lb and np.isfinite(fma[i]) and np.isfinite(fma[i-cfg.entry_slope_lb]):
                if q[i]>fma[i] and fma[i]>fma[i-cfg.entry_slope_lb]:
                    pending=1.; reason='BULL_FULL'
            continue

        # Cash: either initial post-crash entry or resume after a soft exit.
        states.append('RESUME_WAIT' if resume_armed else ('ARMED' if crash_armed else 'CASH'))
        if i>=cfg.entry_slope_lb:
            if resume_armed and r3[i] and np.isfinite(rma[i-cfg.entry_slope_lb]) and rma[i]>rma[i-cfg.entry_slope_lb]:
                pending=cfg.starter_w; reason='RESUME_ENTRY'; continue
            if crash_armed and e3[i] and np.isfinite(ema[i-cfg.entry_slope_lb]) and ema[i]>ema[i-cfg.entry_slope_lb]:
                pending=cfg.starter_w; reason='CRASH_REVERSAL_ENTRY'; crash_armed=False; continue

    eq=pd.Series(eqs,index=dates); w=pd.Series(ws,index=dates)
    tr=pd.DataFrame(trades,columns=['date','reason','target_weight','price','notional','fee','cash_after','shares_after','avg_cost_after'])
    path=None
    if save: path=pd.DataFrame({'date':dates,'qqq':q,'tqqq':t,'equity':eqs,'weight':ws,'state':states,'dd252':dd})
    return eq,w,tr,path

def main():
    p=features(add_features(load_panel())); actual=p.loc[p.price_source.eq('actual'),'date'].min()
    cfgs=[Cfg(x,s,d,r) for x in (80,100,110,120) for s in (20,30,40,50) for d in (-.10,-.12,-.15,-.18) for r in (60,80,100)]
    rows=[]
    for cfg in cfgs:
        eq,w,tr,_=simulate(p,cfg)
        r={'strategy':cfg.name,'exit_ma':cfg.exit_ma,'exit_slope_lb':cfg.exit_slope_lb,'exit_dd':cfg.exit_dd,'resume_ma':cfg.resume_ma}
        r.update(perf(eq)); r.update(period_metrics(eq)); am=perf(eq.loc[actual:])
        r['ActualEra_CAGR']=am['CAGR']; r['ActualEra_MDD']=am['MDD']; r['TradeCount']=len(tr); r['InvestedDayPct']=float((w>.01).mean())
        r.update(cycle_2022_metrics(p,eq,w,tr)); rows.append(r)
    df=pd.DataFrame(rows); df.to_csv(OUT/'v8_sweep.csv',index=False)

    # Main frontier: maximize CAGR/capture with materially lower drawdown than V7 winner.
    fronts=[]
    for floor in (-.60,-.55,-.50,-.45,-.40):
        e=df[(df.MDD>=floor)&(df.DotCom_MDD>=floor)&(df.ActualEra_CAGR>=.25)]
        if len(e):
            a=e.sort_values(['CAGR','Cycle22_Capture'],ascending=[False,False]).iloc[0].copy(); a['risk_floor']=floor; a['selection']='best_CAGR'; fronts.append(a)
            b=e.sort_values(['Cycle22_Capture','CAGR'],ascending=[False,False]).iloc[0].copy(); b['risk_floor']=floor; b['selection']='best_capture'; fronts.append(b)
    front=pd.DataFrame(fronts); front.to_csv(OUT/'v8_frontier.csv',index=False)

    # Balanced score; capture has modest weight to avoid fitting 2022 alone.
    z=df.copy(); z['score']=2.5*z.CAGR+1.2*z.ActualEra_CAGR+.35*z.Cycle22_Capture+.55*z.MDD+.2*z.DotCom_MDD
    z=z.sort_values('score',ascending=False); z.head(100).to_csv(OUT/'v8_top100.csv',index=False)

    names=list(dict.fromkeys(list(z.head(12).strategy)+(list(front.strategy) if len(front) else [])))
    enr=[]
    for name in names:
        r=df.loc[df.strategy.eq(name)].iloc[0].to_dict(); cfg=Cfg(r['exit_ma'],r['exit_slope_lb'],r['exit_dd'],r['resume_ma'])
        eq,w,tr,path=simulate(p,cfg,True); r.update(rolling_metrics(eq,3)); r.update(rolling_metrics(eq,5)); enr.append(r)
        path.to_csv(OUT/f'path_{name}.csv',index=False); tr.to_csv(OUT/f'trades_{name}.csv',index=False)
    pd.DataFrame(enr).to_csv(OUT/'v8_candidates_rolling.csv',index=False)

    cols=['strategy','CAGR','MDD','DotCom_MDD','GFC_MDD','COVID_MDD','2022_Bear_MDD','ActualEra_CAGR','ActualEra_MDD','TradeCount','Cycle22_PortfolioMult','Cycle22_Capture','Cycle22_ExitCountBeforePeak']
    print('=== V8 TOP ==='); print(z[cols+['score']].head(20).to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\n=== V8 FRONTIER ===')
    if len(front): print(front[['risk_floor','selection']+cols].to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    else: print('none')

if __name__=='__main__': main()
