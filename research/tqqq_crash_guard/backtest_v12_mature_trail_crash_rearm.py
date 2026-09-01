from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from backtest_v1 import load_panel, execute_target, perf, period_metrics
from backtest_v2 import add_features
from backtest_v3 import rolling_metrics
from backtest_v6_bull_hold import cycle_2022_metrics

OUT=Path(__file__).resolve().parent/'results_v12_mature_trail_crash_rearm'; OUT.mkdir(parents=True,exist_ok=True)

class Cfg:
    def __init__(self,a,t): self.arm_dd=-.25; self.entry_ma=80; self.full_ma=110; self.starter=.35; self.lb=20; self.slow_ma=110; self.slow_slope=50; self.slow_dd=-.18; self.a=float(a); self.t=float(t)
    @property
    def name(self): return f'V12_A{self.a:.1f}_T{abs(int(self.t*100))}'.replace('.','p')

def features(p):
    p=p.copy(); p['ma80']=p.qqq.rolling(80,min_periods=80).mean(); p['ma110']=p.qqq.rolling(110,min_periods=110).mean(); p['above_ma80_3d']=(p.qqq>p.ma80).rolling(3,min_periods=3).sum().eq(3)
    if 'hi252' not in p: p['hi252']=p.qqq.rolling(252,min_periods=252).max()
    if 'dd252' not in p: p['dd252']=p.qqq/p.hi252-1
    return p

def simulate(p,cfg,save=False):
    q=p.qqq.to_numpy(float); t=p.tqqq.to_numpy(float); dd=p.dd252.to_numpy(float); d=pd.DatetimeIndex(p.date); cy=p.cash_yield_pct.to_numpy(float); m80=p.ma80.to_numpy(float); m110=p.ma110.to_numpy(float); e3=p.above_ma80_3d.to_numpy(bool)
    cash=1.; sh=0.; avg=np.nan; pending=None; reason=None; armed=False; mode='CASH'; peak=np.nan; active=False; eqs=[]; ws=[]; states=[]; trades=[]
    for i in range(len(p)):
        if i>0: cash*=1+max(0.,cy[i-1]/100)/252
        if pending is not None:
            before=sh; cash,sh,avg,n,fee=execute_target(cash,sh,avg,t[i],pending)
            if abs(n)>1e-12: trades.append((d[i],reason,pending,t[i],n,fee,cash,sh,avg))
            if before>0 and sh==0: mode='WAIT_CRASH'; armed=False; peak=np.nan; active=False
            elif sh>0: mode='BULL' if pending>=.999 else 'STARTER'; peak=t[i] if not np.isfinite(peak) else max(peak,t[i])
        pending=None; reason=None
        eq=cash+sh*t[i]; w=sh*t[i]/eq if eq>0 else 0.; eqs.append(eq); ws.append(w); states.append(mode)
        if np.isfinite(dd[i]) and dd[i]<=cfg.arm_dd: armed=True
        if sh>0:
            peak=t[i] if not np.isfinite(peak) else max(peak,t[i]); active=active or (np.isfinite(avg) and peak>=avg*cfg.a)
            slow=False
            if i>=cfg.slow_slope and np.isfinite(m110[i]) and np.isfinite(m110[i-cfg.slow_slope]): slow=(q[i]<m110[i] and m110[i]<m110[i-cfg.slow_slope] and np.isfinite(dd[i]) and dd[i]<=cfg.slow_dd)
            if slow: pending=0.; reason='SLOW_BEAR_EXIT'; continue
            if active and t[i]<=peak*(1+cfg.t): pending=0.; reason='MATURE_TRAIL_EXIT'; continue
            if mode=='STARTER' and i>=cfg.lb and np.isfinite(m110[i]) and np.isfinite(m110[i-cfg.lb]) and q[i]>m110[i] and m110[i]>m110[i-cfg.lb]: pending=1.; reason='BULL_FULL'; continue
        else:
            if armed and i>=cfg.lb and e3[i] and np.isfinite(m80[i-cfg.lb]) and m80[i]>m80[i-cfg.lb]: pending=cfg.starter; reason='CRASH_REVERSAL_ENTRY'; armed=False
    eq=pd.Series(eqs,index=d); w=pd.Series(ws,index=d); tr=pd.DataFrame(trades,columns=['date','reason','target_weight','price','notional','fee','cash_after','shares_after','avg_cost_after']); path=pd.DataFrame({'date':d,'qqq':q,'tqqq':t,'equity':eqs,'weight':ws,'state':states,'dd252':dd}) if save else None
    return eq,w,tr,path

def main():
    p=features(add_features(load_panel())); actual=p.loc[p.price_source.eq('actual'),'date'].min(); rows=[]
    cfgs=[Cfg(a,t) for a in (3.,4.,5.,6.,7.) for t in (-.30,-.35,-.40,-.45,-.50,-.55)]
    for c in cfgs:
        eq,w,tr,_=simulate(p,c); r={'strategy':c.name,'activate_mult':c.a,'trail_dd':c.t}; r.update(perf(eq)); r.update(period_metrics(eq)); am=perf(eq.loc[actual:]); r['ActualEra_CAGR']=am['CAGR']; r['ActualEra_MDD']=am['MDD']; r['TradeCount']=len(tr); r.update(cycle_2022_metrics(p,eq,w,tr)); rows.append(r)
    df=pd.DataFrame(rows); df['score']=3*df.CAGR+1.3*df.ActualEra_CAGR+.4*df.Cycle22_Capture+.7*df.MDD+.2*df.DotCom_MDD+.1*df.GFC_MDD; df=df.sort_values('score',ascending=False); df.to_csv(OUT/'v12_ranked.csv',index=False)
    fronts=[]
    for floor in (-.60,-.55,-.50,-.45):
        e=df[(df.MDD>=floor)&(df.ActualEra_CAGR>=.30)]
        if len(e): z=e.sort_values('CAGR',ascending=False).iloc[0].copy(); z['risk_floor']=floor; fronts.append(z)
    pd.DataFrame(fronts).to_csv(OUT/'v12_frontier.csv',index=False)
    for _,r in df.head(10).iterrows():
        c=Cfg(r.activate_mult,r.trail_dd); eq,w,tr,path=simulate(p,c,True); path.to_csv(OUT/f'path_{c.name}.csv',index=False); tr.to_csv(OUT/f'trades_{c.name}.csv',index=False)
    cols=['strategy','CAGR','MDD','DotCom_MDD','GFC_MDD','COVID_MDD','2022_Bear_MDD','ActualEra_CAGR','ActualEra_MDD','TradeCount','Cycle22_PortfolioMult','Cycle22_Capture','Cycle22_ExitCountBeforePeak','score']; print('=== V12 ==='); print(df[cols].head(20).to_string(index=False,float_format=lambda x:f'{x:.4f}')); print('\nFRONTIER'); print(pd.DataFrame(fronts).to_string(index=False) if fronts else 'none')
if __name__=='__main__': main()
