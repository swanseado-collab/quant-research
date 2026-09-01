from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from backtest_v1 import load_panel, execute_target, perf, period_metrics
from backtest_v2 import add_features
from backtest_v3 import rolling_metrics
from backtest_v6_bull_hold import cycle_2022_metrics

OUT=Path(__file__).resolve().parent/'results_v10_exit_level_resume'; OUT.mkdir(parents=True,exist_ok=True)

class Cfg:
    def __init__(self,xma,xs,xdd,rma,level_mult):
        self.arm_dd=-.25; self.entry_ma=80; self.full_ma=110; self.starter=.35; self.lb=20
        self.xma=int(xma); self.xs=int(xs); self.xdd=float(xdd); self.rma=int(rma); self.level_mult=float(level_mult)
    @property
    def name(self): return f'V10_X{self.xma}S{self.xs}D{abs(int(self.xdd*100))}_R{self.rma}L{self.level_mult:.2f}'.replace('.','p')

def features(p):
    p=p.copy()
    for ma in (60,80,100,110,120):
        p[f'ma{ma}']=p.qqq.rolling(ma,min_periods=ma).mean()
        p[f'above_ma{ma}_3d']=(p.qqq>p[f'ma{ma}']).rolling(3,min_periods=3).sum().eq(3)
    if 'hi252' not in p: p['hi252']=p.qqq.rolling(252,min_periods=252).max()
    if 'dd252' not in p: p['dd252']=p.qqq/p.hi252-1
    return p

def simulate(p,cfg,save=False):
    q=p.qqq.to_numpy(float); t=p.tqqq.to_numpy(float); dd=p.dd252.to_numpy(float); dates=pd.DatetimeIndex(p.date); cy=p.cash_yield_pct.to_numpy(float)
    ema=p.ma80.to_numpy(float); fma=p.ma110.to_numpy(float); xma=p[f'ma{cfg.xma}'].to_numpy(float); rma=p[f'ma{cfg.rma}'].to_numpy(float)
    e3=p.above_ma80_3d.to_numpy(bool); r3=p[f'above_ma{cfg.rma}_3d'].to_numpy(bool)
    cash=1.; sh=0.; avg=np.nan; pending=None; reason=None; armed=False; mode='CASH'; exit_level=np.nan
    eqs=[]; ws=[]; states=[]; trades=[]
    for i in range(len(p)):
        if i>0: cash*=1+max(0.,cy[i-1]/100)/252
        if pending is not None:
            before=sh; cash,sh,avg,notional,fee=execute_target(cash,sh,avg,t[i],pending)
            if abs(notional)>1e-12: trades.append((dates[i],reason,pending,t[i],q[i],notional,fee,cash,sh,avg))
            if before>0 and sh==0: mode='RESUME_WAIT'
            elif sh>0: mode='BULL' if pending>=.999 else 'STARTER'
        pending=None; reason=None
        eq=cash+sh*t[i]; w=sh*t[i]/eq if eq>0 else 0.; eqs.append(eq); ws.append(w); states.append(mode)
        if np.isfinite(dd[i]) and dd[i]<=cfg.arm_dd: armed=True

        if sh>0:
            ex=False
            if i>=cfg.xs and np.isfinite(xma[i]) and np.isfinite(xma[i-cfg.xs]):
                ex=(q[i]<xma[i] and xma[i]<xma[i-cfg.xs] and np.isfinite(dd[i]) and dd[i]<=cfg.xdd)
            if ex:
                exit_level=q[i]; pending=0.; reason='DEFENSIVE_EXIT'; continue
            if mode=='STARTER' and i>=cfg.lb and np.isfinite(fma[i]) and np.isfinite(fma[i-cfg.lb]) and q[i]>fma[i] and fma[i]>fma[i-cfg.lb]:
                pending=1.; reason='BULL_FULL'; continue
            continue

        if mode=='RESUME_WAIT':
            # If decline becomes a genuine crash, switch to the normal post-crash reversal protocol.
            if np.isfinite(dd[i]) and dd[i]<=cfg.arm_dd:
                armed=True; mode='RED_WAIT'
            else:
                level_ok=np.isfinite(exit_level) and q[i] >= exit_level*cfg.level_mult
                trend_ok=i>=cfg.lb and r3[i] and np.isfinite(rma[i-cfg.lb]) and rma[i]>rma[i-cfg.lb]
                if level_ok and trend_ok:
                    pending=cfg.starter; reason='LEVEL_RESUME_ENTRY'; continue
        # Initial / deep-bear re-entry.
        if mode in ('CASH','RED_WAIT') and armed and i>=cfg.lb and e3[i] and np.isfinite(ema[i-cfg.lb]) and ema[i]>ema[i-cfg.lb]:
            pending=cfg.starter; reason='CRASH_REVERSAL_ENTRY'; armed=False; continue

    eq=pd.Series(eqs,index=dates); w=pd.Series(ws,index=dates)
    tr=pd.DataFrame(trades,columns=['date','reason','target_weight','tqqq_price','qqq_price','notional','fee','cash_after','shares_after','avg_cost_after'])
    path=None
    if save: path=pd.DataFrame({'date':dates,'qqq':q,'tqqq':t,'equity':eqs,'weight':ws,'state':states,'dd252':dd})
    return eq,w,tr,path

def main():
    p=features(add_features(load_panel())); actual=p.loc[p.price_source.eq('actual'),'date'].min()
    cfgs=[Cfg(x,s,d,r,l) for x in (80,100,110,120) for s in (20,30,40,50) for d in (-.10,-.12,-.15,-.18) for r in (60,80,100) for l in (.98,1.00,1.02)]
    rows=[]
    for k,cfg in enumerate(cfgs,1):
        eq,w,tr,_=simulate(p,cfg); rr={'strategy':cfg.name,'exit_ma':cfg.xma,'exit_slope':cfg.xs,'exit_dd':cfg.xdd,'resume_ma':cfg.rma,'level_mult':cfg.level_mult}
        rr.update(perf(eq)); rr.update(period_metrics(eq)); am=perf(eq.loc[actual:]); rr['ActualEra_CAGR']=am['CAGR']; rr['ActualEra_MDD']=am['MDD']; rr['TradeCount']=len(tr); rr.update(cycle_2022_metrics(p,eq,w,tr.rename(columns={'tqqq_price':'price'})))
        rows.append(rr)
    df=pd.DataFrame(rows); df.to_csv(OUT/'v10_sweep.csv',index=False)
    fronts=[]
    for floor in (-.60,-.55,-.50,-.45,-.40):
      for af in (.30,.35,.40):
        e=df[(df.MDD>=floor)&(df.DotCom_MDD>=floor)&(df.ActualEra_CAGR>=af)]
        if len(e):
            a=e.sort_values(['CAGR','Cycle22_Capture'],ascending=[False,False]).iloc[0].copy(); a['risk_floor']=floor; a['actual_floor']=af; fronts.append(a)
    front=pd.DataFrame(fronts); front.to_csv(OUT/'v10_frontier.csv',index=False)
    z=df.copy(); z['score']=2.7*z.CAGR+1.2*z.ActualEra_CAGR+.4*z.Cycle22_Capture+.6*z.MDD+.2*z.DotCom_MDD+.1*z.GFC_MDD; z=z.sort_values('score',ascending=False); z.head(150).to_csv(OUT/'v10_top150.csv',index=False)
    names=list(dict.fromkeys(list(z.head(12).strategy)+(list(front.strategy) if len(front) else [])))
    enr=[]
    for name in names:
        r=df.loc[df.strategy.eq(name)].iloc[0].to_dict(); cfg=Cfg(r['exit_ma'],r['exit_slope'],r['exit_dd'],r['resume_ma'],r['level_mult']); eq,w,tr,path=simulate(p,cfg,True); r.update(rolling_metrics(eq,3)); r.update(rolling_metrics(eq,5)); enr.append(r); path.to_csv(OUT/f'path_{name}.csv',index=False); tr.to_csv(OUT/f'trades_{name}.csv',index=False)
    pd.DataFrame(enr).to_csv(OUT/'v10_candidates_rolling.csv',index=False)
    cols=['strategy','CAGR','MDD','DotCom_MDD','GFC_MDD','COVID_MDD','2022_Bear_MDD','ActualEra_CAGR','ActualEra_MDD','TradeCount','Cycle22_PortfolioMult','Cycle22_Capture','Cycle22_ExitCountBeforePeak']
    print('=== V10 TOP ==='); print(z[cols+['score']].head(20).to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\n=== V10 FRONTIER ==='); print(front[['risk_floor','actual_floor']+cols].to_string(index=False,float_format=lambda x:f'{x:.4f}') if len(front) else 'none')
if __name__=='__main__': main()
