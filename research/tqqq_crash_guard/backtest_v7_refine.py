from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from backtest_v1 import load_panel, perf, period_metrics
from backtest_v2 import add_features
from backtest_v3 import rolling_metrics
from backtest_v6_bull_hold import Cfg, simulate, cycle_2022_metrics

OUT=Path(__file__).resolve().parent/'results_v7_refine'
OUT.mkdir(parents=True,exist_ok=True)


def add_custom_features(p):
    p=p.copy()
    for ma in (50,60,70,80,100,110,120):
        p[f'ma{ma}']=p.qqq.rolling(ma,min_periods=ma).mean()
        p[f'above_ma{ma}_3d']=(p.qqq>p[f'ma{ma}']).rolling(3,min_periods=3).sum().eq(3)
    return p


def bh_metrics(p, start):
    z=p[p.date>=start].copy()
    def calc(col):
        s=pd.Series(z[col].to_numpy(float),index=pd.DatetimeIndex(z.date))
        yrs=(s.index[-1]-s.index[0]).days/365.25
        c=(s.iloc[-1]/s.iloc[0])**(1/yrs)-1
        m=(s/s.cummax()-1).min()
        return c,m,s.iloc[-1]/s.iloc[0]
    return calc('tqqq'),calc('qqq')


def main():
    p=add_custom_features(add_features(load_panel()))
    actual=p.loc[p.price_source.eq('actual'),'date'].min()
    cfgs=[]
    for fast in (60,70,80):
      for full in (100,110,120):
       for starter in (.35,.50,.65,.80,1.0):
        for xma in (100,110,120):
         for sl in (30,40,50):
          for xdd in (-.18,-.20,-.22):
           cfgs.append(Cfg(-.25,fast,full,starter,xma,sl,xdd))

    rows=[]
    for k,cfg in enumerate(cfgs,1):
        eq,w,tr,_=simulate(p,cfg)
        r={'strategy':cfg.name,'fast_ma':cfg.fast_ma,'full_ma':cfg.full_ma,'starter_w':cfg.starter_w,
           'exit_ma':cfg.exit_ma,'exit_slope_lb':cfg.exit_slope_lb,'exit_dd':cfg.exit_dd}
        r.update(perf(eq)); r.update(period_metrics(eq))
        am=perf(eq.loc[actual:]); r['ActualEra_CAGR']=am['CAGR']; r['ActualEra_MDD']=am['MDD']
        r['InvestedDayPct']=float((w>.01).mean()); r['AvgWeight']=float(w.mean()); r['TradeCount']=len(tr)
        r.update(cycle_2022_metrics(p,eq,w,tr)); rows.append(r)
        if k%400==0: print(f'tested {k}/{len(cfgs)}')
    df=pd.DataFrame(rows); df.to_csv(OUT/'v7_sweep.csv',index=False)

    # Primary feasible set: preserve strong actual-era compounding while keeping total MDD near 60%.
    feasible=df[(df.MDD>=-.62)&(df.DotCom_MDD>=-.62)&(df.ActualEra_CAGR>=.35)].copy()
    feasible['blend']=2.5*feasible.CAGR+1.2*feasible.ActualEra_CAGR+0.45*feasible.Cycle22_Capture+0.45*feasible.MDD+0.2*feasible.DotCom_MDD
    feasible=feasible.sort_values('blend',ascending=False)
    feasible.head(200).to_csv(OUT/'v7_feasible_top200.csv',index=False)

    # Capture frontier at several risk caps.
    fronts=[]
    for floor in (-.65,-.62,-.60,-.58,-.55):
      e=df[(df.MDD>=floor)&(df.DotCom_MDD>=floor)&(df.ActualEra_CAGR>=.30)]
      if len(e):
        # Report both best CAGR and best 2022-cycle capture; do not hide the tradeoff.
        a=e.sort_values(['CAGR','Cycle22_Capture'],ascending=[False,False]).iloc[0].copy(); a['risk_floor']=floor; a['selection']='best_CAGR'; fronts.append(a)
        b=e.sort_values(['Cycle22_Capture','CAGR'],ascending=[False,False]).iloc[0].copy(); b['risk_floor']=floor; b['selection']='best_capture'; fronts.append(b)
    front=pd.DataFrame(fronts); front.to_csv(OUT/'v7_frontier.csv',index=False)

    # Parameter neighborhood medians are more important than a single optimum.
    sens=(df.groupby(['fast_ma','full_ma','starter_w','exit_ma','exit_slope_lb','exit_dd'],as_index=False)
          .agg(CAGR=('CAGR','median'),MDD=('MDD','median'),DotCom_MDD=('DotCom_MDD','median'),
               ActualEra_CAGR=('ActualEra_CAGR','median'),Cycle22_Capture=('Cycle22_Capture','median'),TradeCount=('TradeCount','median')))
    sens.to_csv(OUT/'v7_sensitivity.csv',index=False)

    names=list(dict.fromkeys(list(feasible.head(12).strategy)+(list(front.strategy) if len(front) else [])))
    enr=[]
    for name in names:
        r=df.loc[df.strategy.eq(name)].iloc[0].to_dict()
        cfg=Cfg(-.25,r['fast_ma'],r['full_ma'],r['starter_w'],r['exit_ma'],r['exit_slope_lb'],r['exit_dd'])
        eq,w,tr,path=simulate(p,cfg,True); r.update(rolling_metrics(eq,3)); r.update(rolling_metrics(eq,5)); enr.append(r)
        path.to_csv(OUT/f'path_{name}.csv',index=False); tr.to_csv(OUT/f'trades_{name}.csv',index=False)
    en=pd.DataFrame(enr); en.to_csv(OUT/'v7_candidates_rolling.csv',index=False)

    tb, qb=bh_metrics(p,actual)
    print(f'BENCHMARK actual-era TQQQ_BH CAGR={tb[0]:.4f} MDD={tb[1]:.4f} multiple={tb[2]:.2f}')
    print(f'BENCHMARK actual-era QQQ_BH  CAGR={qb[0]:.4f} MDD={qb[1]:.4f} multiple={qb[2]:.2f}')
    cols=['strategy','CAGR','MDD','DotCom_MDD','GFC_MDD','COVID_MDD','2022_Bear_MDD','ActualEra_CAGR','ActualEra_MDD','TradeCount','Cycle22_PortfolioMult','Cycle22_Capture','Cycle22_First25Date','Cycle22_First75Date','Cycle22_ExitCountBeforePeak']
    print('\n=== V7 FEASIBLE TOP ===')
    print(feasible[cols+['blend']].head(20).to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\n=== V7 FRONTIER ===')
    if len(front): print(front[['risk_floor','selection']+cols].to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\n=== V7 ROLLING ===')
    if len(en):
        rc=['strategy','CAGR','MDD','ActualEra_CAGR','Cycle22_Capture','Roll3y_CAGR_P10','Roll3y_CAGR_Median','Roll3y_WorstMDD','Roll5y_CAGR_P10','Roll5y_CAGR_Median','Roll5y_WorstMDD']
        print(en.sort_values('CAGR',ascending=False)[rc].head(25).to_string(index=False,float_format=lambda x:f'{x:.4f}'))

if __name__=='__main__': main()
