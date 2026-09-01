from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from backtest_v1 import load_panel, execute_target, perf, period_metrics
from backtest_v2 import add_features
from backtest_v3 import rolling_metrics

OUT = Path(__file__).resolve().parent / "results_v6_bull_hold"
OUT.mkdir(parents=True, exist_ok=True)


class Cfg:
    def __init__(self, arm_dd, fast_ma, full_ma, starter_w, exit_ma, exit_slope_lb, exit_dd):
        self.arm_dd=float(arm_dd)
        self.fast_ma=int(fast_ma)
        self.full_ma=int(full_ma)
        self.starter_w=float(starter_w)
        self.exit_ma=int(exit_ma)
        self.exit_slope_lb=int(exit_slope_lb)
        self.exit_dd=float(exit_dd)
        self.confirm_days=3
        self.entry_slope_lb=20

    @property
    def name(self):
        return (f"BH_A{abs(int(self.arm_dd*100))}_F{self.fast_ma}_M{self.full_ma}_"
                f"W{int(self.starter_w*100)}_X{self.exit_ma}S{self.exit_slope_lb}D{abs(int(self.exit_dd*100))}")


def ensure_features(p):
    p=p.copy()
    for ma in (40,60,80,100,120,150,180,200,250):
        if f"ma{ma}" not in p:
            p[f"ma{ma}"]=p.qqq.rolling(ma,min_periods=ma).mean()
        above=p.qqq>p[f"ma{ma}"]
        if f"above_ma{ma}_3d" not in p:
            p[f"above_ma{ma}_3d"]=above.rolling(3,min_periods=3).sum().eq(3)
    if "hi252" not in p:
        p["hi252"]=p.qqq.rolling(252,min_periods=252).max()
    if "dd252" not in p:
        p["dd252"]=p.qqq/p.hi252-1
    return p


def simulate(p: pd.DataFrame, cfg: Cfg, save_path=False):
    q=p.qqq.to_numpy(float); t=p.tqqq.to_numpy(float)
    dates=pd.DatetimeIndex(p.date); cy=p.cash_yield_pct.to_numpy(float)
    dd=p.dd252.to_numpy(float)
    fma=p[f"ma{cfg.fast_ma}"].to_numpy(float)
    fullma=p[f"ma{cfg.full_ma}"].to_numpy(float)
    xma=p[f"ma{cfg.exit_ma}"].to_numpy(float)
    fast3=p[f"above_ma{cfg.fast_ma}_3d"].to_numpy(bool)

    cash=1.0; shares=0.0; avg=np.nan
    pending=None; reason=None
    armed=False; stage=0  # 0 cash/wait, 1 starter, 2 full bull
    eqs=[]; ws=[]; states=[]; trades=[]

    for i in range(len(p)):
        if i>0:
            cash*=1+max(0.0,cy[i-1]/100)/252

        if pending is not None:
            before=shares
            cash,shares,avg,notional,fee=execute_target(cash,shares,avg,t[i],pending)
            if abs(notional)>1e-12:
                trades.append((dates[i],reason,pending,t[i],notional,fee,cash,shares,avg))
            if before>0 and shares==0:
                stage=0
                # after a defensive exit, only re-arm immediately if the market is still deeply depressed
                armed=bool(np.isfinite(dd[i]) and dd[i] <= cfg.arm_dd)
            elif shares>0:
                stage=2 if pending>=.999 else 1
        pending=None; reason=None

        eq=cash+shares*t[i]
        w=shares*t[i]/eq if eq>0 else 0.0
        eqs.append(eq); ws.append(w)

        if np.isfinite(dd[i]) and dd[i] <= cfg.arm_dd:
            armed=True

        # Exit has priority once invested. No profit target: bull runs are allowed to compound.
        exit_signal=False
        if shares>0 and i>=cfg.exit_slope_lb and np.isfinite(xma[i]) and np.isfinite(xma[i-cfg.exit_slope_lb]):
            exit_signal=(q[i] < xma[i] and xma[i] < xma[i-cfg.exit_slope_lb]
                         and np.isfinite(dd[i]) and dd[i] <= cfg.exit_dd)
        if exit_signal:
            pending=0.0; reason="BULL_EXIT"
            states.append("EXIT_SIGNAL")
            continue

        # Cash state: wait for a genuine post-crash reversal, not merely a falling knife.
        if shares<=0:
            states.append("ARMED" if armed else "CASH")
            if armed and i>=cfg.entry_slope_lb and fast3[i] and np.isfinite(fma[i-cfg.entry_slope_lb]):
                fast_rising = fma[i] > fma[i-cfg.entry_slope_lb]
                if fast_rising:
                    pending=cfg.starter_w
                    reason="REVERSAL_ENTRY"
            continue

        # Starter state: promote to 100% only when the medium trend also heals.
        states.append("STARTER" if stage==1 else "BULL")
        if stage==1 and i>=cfg.entry_slope_lb and np.isfinite(fullma[i]) and np.isfinite(fullma[i-cfg.entry_slope_lb]):
            full_ok=(q[i] > fullma[i] and fullma[i] > fullma[i-cfg.entry_slope_lb])
            if full_ok:
                pending=1.0; reason="BULL_FULL"

    eq=pd.Series(eqs,index=dates); w=pd.Series(ws,index=dates)
    tr=pd.DataFrame(trades,columns=["date","reason","target_weight","price","notional","fee","cash_after","shares_after","avg_cost_after"])
    path=None
    if save_path:
        path=pd.DataFrame({"date":dates,"qqq":q,"tqqq":t,"equity":eqs,"weight":ws,"state":states,"dd252":dd})
    return eq,w,tr,path


def cycle_2022_metrics(p,eq,w,tr):
    # Define the observed 2022 crash trough ex ante only for REPORTING, never for signal generation.
    z=p[(p.date>=pd.Timestamp("2021-11-01"))&(p.date<=pd.Timestamp("2023-03-31"))]
    if z.empty: return {}
    trough_idx=z.tqqq.idxmin(); trough_date=p.loc[trough_idx,"date"]; trough_px=float(p.loc[trough_idx,"tqqq"])
    post=p[p.date>=trough_date]
    peak_idx=post.tqqq.idxmax(); peak_date=p.loc[peak_idx,"date"]; peak_px=float(p.loc[peak_idx,"tqqq"])
    e0=float(eq.loc[trough_date]); e1=float(eq.loc[peak_date])
    theoretical_mult=peak_px/trough_px
    portfolio_mult=e1/e0
    capture=(portfolio_mult-1)/(theoretical_mult-1) if theoretical_mult>1 else np.nan
    ww=w.loc[trough_date:peak_date]
    first25=ww[ww>=.25].index.min() if (ww>=.25).any() else pd.NaT
    first75=ww[ww>=.75].index.min() if (ww>=.75).any() else pd.NaT
    exits=tr[(tr.reason=="BULL_EXIT")&(tr.date>=trough_date)&(tr.date<=peak_date)]
    return {
        "Cycle22_TroughDate":trough_date,"Cycle22_TroughPx":trough_px,
        "Cycle22_PeakDate":peak_date,"Cycle22_PeakPx":peak_px,
        "Cycle22_TheoreticalMult":theoretical_mult,"Cycle22_PortfolioMult":portfolio_mult,
        "Cycle22_Capture":capture,"Cycle22_First25Date":first25,"Cycle22_First75Date":first75,
        "Cycle22_WeightAtPeak":float(w.loc[peak_date]),"Cycle22_MaxWeight":float(ww.max()),
        "Cycle22_ExitCountBeforePeak":int(len(exits)),
    }


def summarize(p,cfg,eq,w,tr,actual_start):
    r={"strategy":cfg.name,"arm_dd":cfg.arm_dd,"fast_ma":cfg.fast_ma,"full_ma":cfg.full_ma,
       "starter_w":cfg.starter_w,"exit_ma":cfg.exit_ma,"exit_slope_lb":cfg.exit_slope_lb,"exit_dd":cfg.exit_dd}
    r.update(perf(eq)); r.update(period_metrics(eq))
    am=perf(eq.loc[actual_start:]); r["ActualEra_CAGR"]=am["CAGR"]; r["ActualEra_MDD"]=am["MDD"]
    r["InvestedDayPct"]=float((w>.01).mean()); r["AvgWeight"]=float(w.mean()); r["TradeCount"]=len(tr)
    r.update(cycle_2022_metrics(p,eq,w,tr))
    return r


def main():
    p=ensure_features(add_features(load_panel()))
    actual=p.loc[p.price_source.eq("actual"),"date"].min()

    cfgs=[]
    for arm_dd in (-.20,-.25,-.30):
      for fast_ma in (40,60,80):
       for full_ma in (100,120,150):
        for starter_w in (.35,1.0):
         for exit_ma in (100,120,150,200):
          for exit_slope in (10,20,40):
           for exit_dd in (-.10,-.15,-.20):
            cfgs.append(Cfg(arm_dd,fast_ma,full_ma,starter_w,exit_ma,exit_slope,exit_dd))

    rows=[]
    total=len(cfgs)
    for k,cfg in enumerate(cfgs,1):
        eq,w,tr,_=simulate(p,cfg)
        rows.append(summarize(p,cfg,eq,w,tr,actual))
        if k%500==0: print(f"tested {k}/{total}")
    df=pd.DataFrame(rows)
    df.to_csv(OUT/"v6_sweep.csv",index=False)

    # Efficient frontier: protect against secular crashes while preserving actual-era compounding.
    fronts=[]
    for mdd_floor in (-.75,-.70,-.65,-.60,-.55,-.50):
      for actual_floor in (.18,.20,.22,.24,.26):
        e=df[(df.MDD>=mdd_floor)&(df.DotCom_MDD>=mdd_floor)&(df.ActualEra_CAGR>=actual_floor)]
        if len(e):
            z=e.sort_values(["CAGR","ActualEra_CAGR","Cycle22_Capture"],ascending=[False,False,False]).iloc[0].copy()
            z["MDD_floor"]=mdd_floor; z["Actual_floor"]=actual_floor; fronts.append(z)
    front=pd.DataFrame(fronts)
    front.to_csv(OUT/"v6_frontier.csv",index=False)

    # Highest cycle capture subject to non-catastrophic historical drawdown and decent long-run return.
    cap=df[(df.MDD>=-.70)&(df.CAGR>=.12)&(df.ActualEra_CAGR>=.18)].copy()
    cap=cap.sort_values(["Cycle22_Capture","CAGR"],ascending=[False,False]).head(100)
    cap.to_csv(OUT/"v6_cycle_capture_top100.csv",index=False)

    # Broad robust ranking. 2022 capture is deliberately low weight to avoid fitting that single cycle.
    z=df.copy()
    z["score"]=3.0*z.CAGR + 1.5*z.ActualEra_CAGR + .45*z.MDD + .25*z.DotCom_MDD + .10*z.GFC_MDD + .20*z.Cycle22_Capture
    z=z.sort_values("score",ascending=False)
    z.head(200).to_csv(OUT/"v6_top200_score.csv",index=False)

    # Rolling metrics and full paths for a compact candidate set.
    names=list(dict.fromkeys(list(df.sort_values("CAGR",ascending=False).head(10).strategy)
                             +(list(front.strategy) if len(front) else [])
                             +list(cap.head(10).strategy)+list(z.head(10).strategy)))
    enriched=[]
    for name in names:
        r=df.loc[df.strategy.eq(name)].iloc[0].to_dict()
        cfg=Cfg(r["arm_dd"],r["fast_ma"],r["full_ma"],r["starter_w"],r["exit_ma"],r["exit_slope_lb"],r["exit_dd"])
        eq,w,tr,path=simulate(p,cfg,True)
        r.update(rolling_metrics(eq,3)); r.update(rolling_metrics(eq,5)); enriched.append(r)
        path.to_csv(OUT/f"path_{name}.csv",index=False)
        tr.to_csv(OUT/f"trades_{name}.csv",index=False)
    en=pd.DataFrame(enriched); en.to_csv(OUT/"v6_candidates_rolling.csv",index=False)

    cols=["strategy","CAGR","MDD","DotCom_MDD","GFC_MDD","COVID_MDD","2022_Bear_MDD","ActualEra_CAGR","ActualEra_MDD",
          "InvestedDayPct","TradeCount","Cycle22_TheoreticalMult","Cycle22_PortfolioMult","Cycle22_Capture","Cycle22_First25Date","Cycle22_First75Date","Cycle22_WeightAtPeak"]
    print("\n=== V6 TOP CAGR ===")
    print(df.sort_values("CAGR",ascending=False)[cols].head(20).to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\n=== V6 FRONTIER ===")
    if len(front):
        print(front[["MDD_floor","Actual_floor"]+cols].to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    else: print("none")
    print("\n=== V6 TOP 2022->2026 CYCLE CAPTURE (with guardrails) ===")
    if len(cap): print(cap[cols].head(20).to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    else: print("none")
    print("\n=== V6 ROBUST SCORE ===")
    print(z[cols+["score"]].head(20).to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\n=== V6 ROLLING ===")
    if len(en):
        rc=["strategy","CAGR","MDD","ActualEra_CAGR","Cycle22_Capture","Roll3y_CAGR_P10","Roll3y_CAGR_Median","Roll3y_WorstMDD","Roll5y_CAGR_P10","Roll5y_CAGR_Median","Roll5y_WorstMDD"]
        print(en.sort_values("score" if "score" in en else "CAGR",ascending=False)[rc].head(30).to_string(index=False,float_format=lambda x:f"{x:.4f}"))

if __name__=="__main__":
    main()
