from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from backtest_v1 import load_panel, execute_target, perf, period_metrics
from backtest_v2 import add_features
from backtest_v3 import rolling_metrics
from backtest_v6_bull_hold import cycle_2022_metrics

OUT=Path(__file__).resolve().parent/'results_v9_three_state'
OUT.mkdir(parents=True,exist_ok=True)

class Cfg:
    def __init__(self, soft_ma, soft_slope, soft_dd, soft_w, hard_ma, hard_slope, hard_dd, recover_ma):
        self.arm_dd=-.25
        self.entry_ma=80
        self.full_ma=110
        self.starter_w=.35
        self.soft_ma=int(soft_ma); self.soft_slope=int(soft_slope); self.soft_dd=float(soft_dd); self.soft_w=float(soft_w)
        self.hard_ma=int(hard_ma); self.hard_slope=int(hard_slope); self.hard_dd=float(hard_dd)
        self.recover_ma=int(recover_ma); self.lb=20
    @property
    def name(self):
        return (f'V9_SM{self.soft_ma}S{self.soft_slope}D{abs(int(self.soft_dd*100))}W{int(self.soft_w*100)}_'
                f'HM{self.hard_ma}S{self.hard_slope}D{abs(int(self.hard_dd*100))}_R{self.recover_ma}')

def features(p):
    p=p.copy()
    for ma in (60,80,100,110,120,150):
        p[f'ma{ma}']=p.qqq.rolling(ma,min_periods=ma).mean()
        p[f'above_ma{ma}_3d']=(p.qqq>p[f'ma{ma}']).rolling(3,min_periods=3).sum().eq(3)
    if 'hi252' not in p: p['hi252']=p.qqq.rolling(252,min_periods=252).max()
    if 'dd252' not in p: p['dd252']=p.qqq/p.hi252-1
    return p

def simulate(p,cfg,save=False):
    q=p.qqq.to_numpy(float); t=p.tqqq.to_numpy(float); dd=p.dd252.to_numpy(float)
    dates=pd.DatetimeIndex(p.date); cy=p.cash_yield_pct.to_numpy(float)
    ema=p[f'ma{cfg.entry_ma}'].to_numpy(float); fma=p[f'ma{cfg.full_ma}'].to_numpy(float)
    sma=p[f'ma{cfg.soft_ma}'].to_numpy(float); hma=p[f'ma{cfg.hard_ma}'].to_numpy(float); rma=p[f'ma{cfg.recover_ma}'].to_numpy(float)
    e3=p[f'above_ma{cfg.entry_ma}_3d'].to_numpy(bool); r3=p[f'above_ma{cfg.recover_ma}_3d'].to_numpy(bool)

    cash=1.; sh=0.; avg=np.nan; pending=None; reason=None
    armed=False; state='CASH'  # CASH, STARTER, BULL, CAUTION, RED
    eqs=[]; ws=[]; states=[]; trades=[]

    for i in range(len(p)):
        if i>0: cash*=1+max(0.,cy[i-1]/100)/252
        if pending is not None:
            cash,sh,avg,notional,fee=execute_target(cash,sh,avg,t[i],pending)
            if abs(notional)>1e-12: trades.append((dates[i],reason,pending,t[i],notional,fee,cash,sh,avg))
            if pending<=1e-10: state='RED'
            elif pending>=.999: state='BULL'
            elif pending<=cfg.soft_w+.01 and reason=='SOFT_CUT': state='CAUTION'
            else: state='STARTER'
        pending=None; reason=None

        eq=cash+sh*t[i]; w=sh*t[i]/eq if eq>0 else 0.; eqs.append(eq); ws.append(w); states.append(state)
        if np.isfinite(dd[i]) and dd[i]<=cfg.arm_dd: armed=True

        # Hard RED has absolute priority. It prevents repeated bear-rally exposure in secular declines.
        hard=False
        if i>=cfg.hard_slope and np.isfinite(hma[i]) and np.isfinite(hma[i-cfg.hard_slope]):
            hard=(q[i]<hma[i] and hma[i]<hma[i-cfg.hard_slope] and np.isfinite(dd[i]) and dd[i]<=cfg.hard_dd)
        if sh>0 and hard:
            pending=0.; reason='HARD_RED'; continue

        if state=='RED' or sh<=0:
            # Re-entry from true RED still requires a genuine post-crash reversal.
            if armed and i>=cfg.lb and e3[i] and np.isfinite(ema[i-cfg.lb]) and ema[i]>ema[i-cfg.lb]:
                pending=cfg.starter_w; reason='RED_REVERSAL_ENTRY'; armed=False
            continue

        # In CAUTION, regain full risk only on renewed medium-term uptrend.
        if state=='CAUTION':
            if i>=cfg.lb and r3[i] and np.isfinite(rma[i-cfg.lb]) and rma[i]>rma[i-cfg.lb]:
                pending=1.; reason='CAUTION_RECOVER_FULL'; continue
            # Otherwise remain at reduced weight until hard RED or recovery.
            continue

        # Starter after a crash: only promote when medium trend heals.
        if state=='STARTER':
            if i>=cfg.lb and np.isfinite(fma[i]) and np.isfinite(fma[i-cfg.lb]) and q[i]>fma[i] and fma[i]>fma[i-cfg.lb]:
                pending=1.; reason='BULL_FULL'; continue
            continue

        # BULL soft de-risk: reduce, don't exit, on a shallower trend deterioration.
        soft=False
        if i>=cfg.soft_slope and np.isfinite(sma[i]) and np.isfinite(sma[i-cfg.soft_slope]):
            soft=(q[i]<sma[i] and sma[i]<sma[i-cfg.soft_slope] and np.isfinite(dd[i]) and dd[i]<=cfg.soft_dd)
        if state=='BULL' and soft and w>cfg.soft_w+.02:
            pending=cfg.soft_w; reason='SOFT_CUT'; continue

    eq=pd.Series(eqs,index=dates); w=pd.Series(ws,index=dates)
    tr=pd.DataFrame(trades,columns=['date','reason','target_weight','price','notional','fee','cash_after','shares_after','avg_cost_after'])
    path=None
    if save: path=pd.DataFrame({'date':dates,'qqq':q,'tqqq':t,'equity':eqs,'weight':ws,'state':states,'dd252':dd})
    return eq,w,tr,path

def main():
    p=features(add_features(load_panel())); actual=p.loc[p.price_source.eq('actual'),'date'].min()
    cfgs=[]
    for sm in (80,100,110):
      for ss in (20,30,40):
       for sd in (-.10,-.12,-.15):
        for sw in (.25,.35,.50):
         for hm in (110,120,150):
          for hs in (20,40):
           for hd in (-.22,-.25,-.30):
            for rm in (80,100,110):
             cfgs.append(Cfg(sm,ss,sd,sw,hm,hs,hd,rm))
    # deterministic downsample of redundant Cartesian product to keep runtime reasonable while preserving neighborhoods
    cfgs=cfgs[::3]
    rows=[]
    for k,cfg in enumerate(cfgs,1):
        eq,w,tr,_=simulate(p,cfg)
        r={'strategy':cfg.name,'soft_ma':cfg.soft_ma,'soft_slope':cfg.soft_slope,'soft_dd':cfg.soft_dd,'soft_w':cfg.soft_w,
           'hard_ma':cfg.hard_ma,'hard_slope':cfg.hard_slope,'hard_dd':cfg.hard_dd,'recover_ma':cfg.recover_ma}
        r.update(perf(eq)); r.update(period_metrics(eq)); am=perf(eq.loc[actual:])
        r['ActualEra_CAGR']=am['CAGR']; r['ActualEra_MDD']=am['MDD']; r['TradeCount']=len(tr); r['InvestedDayPct']=float((w>.01).mean())
        r.update(cycle_2022_metrics(p,eq,w,tr)); rows.append(r)
        if k%500==0: print(f'tested {k}/{len(cfgs)}')
    df=pd.DataFrame(rows); df.to_csv(OUT/'v9_sweep.csv',index=False)

    fronts=[]
    for floor in (-.60,-.55,-.50,-.45):
      for af in (.30,.35,.40):
        e=df[(df.MDD>=floor)&(df.DotCom_MDD>=floor)&(df.ActualEra_CAGR>=af)]
        if len(e):
            a=e.sort_values(['CAGR','Cycle22_Capture'],ascending=[False,False]).iloc[0].copy(); a['risk_floor']=floor; a['actual_floor']=af; fronts.append(a)
    front=pd.DataFrame(fronts); front.to_csv(OUT/'v9_frontier.csv',index=False)

    z=df.copy(); z['score']=2.7*z.CAGR+1.2*z.ActualEra_CAGR+.4*z.Cycle22_Capture+.55*z.MDD+.2*z.DotCom_MDD+.1*z.GFC_MDD
    z=z.sort_values('score',ascending=False); z.head(150).to_csv(OUT/'v9_top150.csv',index=False)
    names=list(dict.fromkeys(list(z.head(12).strategy)+(list(front.strategy) if len(front) else [])))
    enr=[]
    for name in names:
        r=df.loc[df.strategy.eq(name)].iloc[0].to_dict(); cfg=Cfg(r['soft_ma'],r['soft_slope'],r['soft_dd'],r['soft_w'],r['hard_ma'],r['hard_slope'],r['hard_dd'],r['recover_ma'])
        eq,w,tr,path=simulate(p,cfg,True); r.update(rolling_metrics(eq,3)); r.update(rolling_metrics(eq,5)); enr.append(r)
        path.to_csv(OUT/f'path_{name}.csv',index=False); tr.to_csv(OUT/f'trades_{name}.csv',index=False)
    pd.DataFrame(enr).to_csv(OUT/'v9_candidates_rolling.csv',index=False)

    cols=['strategy','CAGR','MDD','DotCom_MDD','GFC_MDD','COVID_MDD','2022_Bear_MDD','ActualEra_CAGR','ActualEra_MDD','TradeCount','Cycle22_PortfolioMult','Cycle22_Capture','Cycle22_ExitCountBeforePeak']
    print('=== V9 TOP ==='); print(z[cols+['score']].head(20).to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\n=== V9 FRONTIER ===')
    if len(front): print(front[['risk_floor','actual_floor']+cols].to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    else: print('none')

if __name__=='__main__': main()
