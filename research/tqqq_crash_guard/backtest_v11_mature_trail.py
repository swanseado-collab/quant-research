from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from backtest_v1 import load_panel, execute_target, perf, period_metrics
from backtest_v2 import add_features
from backtest_v3 import rolling_metrics
from backtest_v6_bull_hold import cycle_2022_metrics

OUT=Path(__file__).resolve().parent/'results_v11_mature_trail'; OUT.mkdir(parents=True,exist_ok=True)

class Cfg:
    def __init__(self,activate_mult,trail_dd,reenter_level):
        self.arm_dd=-.25; self.entry_ma=80; self.full_ma=110; self.starter=.35; self.lb=20
        self.slow_ma=110; self.slow_slope=50; self.slow_dd=-.18
        self.activate_mult=float(activate_mult); self.trail_dd=float(trail_dd); self.reenter_level=float(reenter_level)
    @property
    def name(self): return f'V11_A{self.activate_mult:.1f}_T{abs(int(self.trail_dd*100))}_R{self.reenter_level:.3f}'.replace('.','p')

def features(p):
    p=p.copy()
    for ma in (80,110):
        p[f'ma{ma}']=p.qqq.rolling(ma,min_periods=ma).mean()
        p[f'above_ma{ma}_3d']=(p.qqq>p[f'ma{ma}']).rolling(3,min_periods=3).sum().eq(3)
    if 'hi252' not in p: p['hi252']=p.qqq.rolling(252,min_periods=252).max()
    if 'dd252' not in p: p['dd252']=p.qqq/p.hi252-1
    return p

def simulate(p,cfg,save=False):
    q=p.qqq.to_numpy(float); t=p.tqqq.to_numpy(float); dd=p.dd252.to_numpy(float); dates=pd.DatetimeIndex(p.date); cy=p.cash_yield_pct.to_numpy(float)
    ma80=p.ma80.to_numpy(float); ma110=p.ma110.to_numpy(float); e3=p.above_ma80_3d.to_numpy(bool)
    cash=1.; sh=0.; avg=np.nan; pending=None; reason=None; armed=False; mode='CASH'
    t_peak=np.nan; q_peak=np.nan; trail_active=False; profit_exit_peak=np.nan
    eqs=[]; ws=[]; states=[]; trades=[]
    for i in range(len(p)):
        if i>0: cash*=1+max(0.,cy[i-1]/100)/252
        if pending is not None:
            before=sh; cash,sh,avg,notional,fee=execute_target(cash,sh,avg,t[i],pending)
            if abs(notional)>1e-12: trades.append((dates[i],reason,pending,t[i],q[i],notional,fee,cash,sh,avg))
            if before>0 and sh==0:
                if reason=='MATURE_TRAIL_EXIT': mode='PROFIT_WAIT'
                else: mode='RED_WAIT'
                t_peak=np.nan; trail_active=False
            elif sh>0:
                mode='BULL' if pending>=.999 else 'STARTER'
                t_peak=t[i] if not np.isfinite(t_peak) else max(t_peak,t[i]); q_peak=q[i] if not np.isfinite(q_peak) else max(q_peak,q[i])
        pending=None; reason=None
        eq=cash+sh*t[i]; w=sh*t[i]/eq if eq>0 else 0.; eqs.append(eq); ws.append(w); states.append(mode)
        if np.isfinite(dd[i]) and dd[i]<=cfg.arm_dd: armed=True

        if sh>0:
            t_peak=t[i] if not np.isfinite(t_peak) else max(t_peak,t[i]); q_peak=q[i] if not np.isfinite(q_peak) else max(q_peak,q[i])
            if np.isfinite(avg) and t_peak>=avg*cfg.activate_mult: trail_active=True
            # Slow secular-bear exit has priority over profit trailing classification.
            slow=False
            if i>=cfg.slow_slope and np.isfinite(ma110[i]) and np.isfinite(ma110[i-cfg.slow_slope]):
                slow=(q[i]<ma110[i] and ma110[i]<ma110[i-cfg.slow_slope] and np.isfinite(dd[i]) and dd[i]<=cfg.slow_dd)
            if slow:
                pending=0.; reason='SLOW_BEAR_EXIT'; continue
            if trail_active and t[i] <= t_peak*(1+cfg.trail_dd):
                profit_exit_peak=q_peak; pending=0.; reason='MATURE_TRAIL_EXIT'; continue
            if mode=='STARTER' and i>=cfg.lb and np.isfinite(ma110[i]) and np.isfinite(ma110[i-cfg.lb]) and q[i]>ma110[i] and ma110[i]>ma110[i-cfg.lb]:
                pending=1.; reason='BULL_FULL'; continue
            continue

        if mode=='PROFIT_WAIT':
            # Rejoin an interrupted secular bull only after recovering the PRIOR QQQ peak.
            level_ok=np.isfinite(profit_exit_peak) and q[i]>=profit_exit_peak*cfg.reenter_level
            trend_ok=i>=cfg.lb and e3[i] and np.isfinite(ma80[i-cfg.lb]) and ma80[i]>ma80[i-cfg.lb]
            if level_ok and trend_ok:
                pending=cfg.starter; reason='NEW_HIGH_REENTRY'; q_peak=q[i]; continue
            # If instead a deep crash develops, use the normal reversal protocol.
            if np.isfinite(dd[i]) and dd[i]<=cfg.arm_dd:
                mode='RED_WAIT'; armed=True

        if mode in ('CASH','RED_WAIT') and armed and i>=cfg.lb and e3[i] and np.isfinite(ma80[i-cfg.lb]) and ma80[i]>ma80[i-cfg.lb]:
            pending=cfg.starter; reason='CRASH_REVERSAL_ENTRY'; armed=False; q_peak=q[i]; continue

    eq=pd.Series(eqs,index=dates); w=pd.Series(ws,index=dates)
    tr=pd.DataFrame(trades,columns=['date','reason','target_weight','tqqq_price','qqq_price','notional','fee','cash_after','shares_after','avg_cost_after'])
    path=None
    if save: path=pd.DataFrame({'date':dates,'qqq':q,'tqqq':t,'equity':eqs,'weight':ws,'state':states,'dd252':dd})
    return eq,w,tr,path

def main():
    p=features(add_features(load_panel())); actual=p.loc[p.price_source.eq('actual'),'date'].min()
    cfgs=[Cfg(a,t,r) for a in (2.0,3.0,4.0,5.0) for t in (-.30,-.35,-.40,-.45,-.50,-.55) for r in (.98,1.00,1.02)]
    rows=[]
    for cfg in cfgs:
        eq,w,tr,_=simulate(p,cfg); rr={'strategy':cfg.name,'activate_mult':cfg.activate_mult,'trail_dd':cfg.trail_dd,'reenter_level':cfg.reenter_level}; rr.update(perf(eq)); rr.update(period_metrics(eq)); am=perf(eq.loc[actual:]); rr['ActualEra_CAGR']=am['CAGR']; rr['ActualEra_MDD']=am['MDD']; rr['TradeCount']=len(tr); rr.update(cycle_2022_metrics(p,eq,w,tr.rename(columns={'tqqq_price':'price'}))); rows.append(rr)
    df=pd.DataFrame(rows); df.to_csv(OUT/'v11_sweep.csv',index=False)
    fronts=[]
    for floor in (-.60,-.55,-.50,-.45,-.40):
      for af in (.30,.35,.40):
        e=df[(df.MDD>=floor)&(df.DotCom_MDD>=floor)&(df.ActualEra_CAGR>=af)]
        if len(e):
            a=e.sort_values(['CAGR','Cycle22_Capture'],ascending=[False,False]).iloc[0].copy(); a['risk_floor']=floor; a['actual_floor']=af; fronts.append(a)
    front=pd.DataFrame(fronts); front.to_csv(OUT/'v11_frontier.csv',index=False)
    z=df.copy(); z['score']=2.8*z.CAGR+1.2*z.ActualEra_CAGR+.35*z.Cycle22_Capture+.65*z.MDD+.2*z.DotCom_MDD+.1*z.GFC_MDD; z=z.sort_values('score',ascending=False); z.to_csv(OUT/'v11_ranked.csv',index=False)
    names=list(dict.fromkeys(list(z.head(12).strategy)+(list(front.strategy) if len(front) else []))); enr=[]
    for name in names:
        r=df.loc[df.strategy.eq(name)].iloc[0].to_dict(); cfg=Cfg(r['activate_mult'],r['trail_dd'],r['reenter_level']); eq,w,tr,path=simulate(p,cfg,True); r.update(rolling_metrics(eq,3)); r.update(rolling_metrics(eq,5)); enr.append(r); path.to_csv(OUT/f'path_{name}.csv',index=False); tr.to_csv(OUT/f'trades_{name}.csv',index=False)
    pd.DataFrame(enr).to_csv(OUT/'v11_candidates_rolling.csv',index=False)
    cols=['strategy','CAGR','MDD','DotCom_MDD','GFC_MDD','COVID_MDD','2022_Bear_MDD','ActualEra_CAGR','ActualEra_MDD','TradeCount','Cycle22_PortfolioMult','Cycle22_Capture','Cycle22_ExitCountBeforePeak']
    print('=== V11 TOP ==='); print(z[cols+['score']].head(20).to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\n=== V11 FRONTIER ==='); print(front[['risk_floor','actual_floor']+cols].to_string(index=False,float_format=lambda x:f'{x:.4f}') if len(front) else 'none')
if __name__=='__main__': main()
