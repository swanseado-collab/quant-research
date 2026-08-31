from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from backtest_v1 import load_panel, perf, period_metrics, ENTRY_DD, TARGET_W, TP, ONE_WAY_COST
from backtest_v2 import add_features
from backtest_v3 import rolling_metrics

OUT = Path(__file__).resolve().parent / "results_v5"
OUT.mkdir(parents=True, exist_ok=True)


class V5:
    def __init__(self, trend_ma, slope_lb, yellow_buy_cap, red_dd, red_tqqq, red_qqq, rec_fast_ma=60):
        self.trend_ma=int(trend_ma); self.slope_lb=int(slope_lb)
        self.yellow_buy_cap=float(yellow_buy_cap); self.red_dd=float(red_dd)
        self.red_tqqq=float(red_tqqq); self.red_qqq=float(red_qqq)
        self.rec_fast_ma=int(rec_fast_ma); self.rec_fast_days=3; self.rec_mid_ma=120
        if self.red_tqqq + self.red_qqq > 1.000001:
            raise ValueError("red targets exceed 100%")

    @property
    def name(self):
        return (f"V5_M{self.trend_ma}_S{self.slope_lb}_Y{int(self.yellow_buy_cap*100)}_"
                f"RD{abs(int(self.red_dd*100))}_T{int(self.red_tqqq*100)}Q{int(self.red_qqq*100)}")


def rebalance(cash, tsh, tavg, qsh, tpx, qpx, target_t, target_q):
    target_t=float(np.clip(target_t,0,1)); target_q=float(np.clip(target_q,0,1-target_t))
    eq=cash+tsh*tpx+qsh*qpx
    desired_t=target_t*eq; desired_q=target_q*eq
    cur_t=tsh*tpx; cur_q=qsh*qpx
    fees=0.0; notionals=[]

    # Sell first.
    if cur_t > desired_t + 1e-12:
        sell=cur_t-desired_t; qty=sell/tpx; fee=sell*ONE_WAY_COST
        tsh-=qty; cash+=sell-fee; fees+=fee; notionals.append(("TQQQ",-sell))
        if tsh <= 1e-12:
            tsh=0.0; tavg=np.nan
    if cur_q > desired_q + 1e-12:
        sell=cur_q-desired_q; qty=sell/qpx; fee=sell*ONE_WAY_COST
        qsh-=qty; cash+=sell-fee; fees+=fee; notionals.append(("QQQ",-sell))
        if qsh <= 1e-12: qsh=0.0

    # Recompute desired notionals using post-sell equity approximately; target drift from fees is tiny.
    eq=cash+tsh*tpx+qsh*qpx
    desired_t=target_t*eq; desired_q=target_q*eq
    cur_t=tsh*tpx; cur_q=qsh*qpx

    if cur_t < desired_t - 1e-12:
        buy=min(desired_t-cur_t, cash/(1+ONE_WAY_COST))
        if buy>0:
            qty=buy/tpx; fee=buy*ONE_WAY_COST
            old_basis=tsh*tavg if tsh>0 and np.isfinite(tavg) else 0.0
            tsh+=qty; cash-=buy+fee; fees+=fee
            tavg=(old_basis+buy+fee)/tsh
            notionals.append(("TQQQ",buy))
    if cur_q < desired_q - 1e-12:
        buy=min(desired_q-cur_q, cash/(1+ONE_WAY_COST))
        if buy>0:
            qty=buy/qpx; fee=buy*ONE_WAY_COST
            qsh+=qty; cash-=buy+fee; fees+=fee
            notionals.append(("QQQ",buy))
    return cash,tsh,tavg,qsh,fees,notionals


def simulate(p: pd.DataFrame, cfg: V5, save_path=False):
    q=p["qqq"].to_numpy(float); t=p["tqqq"].to_numpy(float)
    dates=pd.DatetimeIndex(p.date); cy=p.cash_yield_pct.to_numpy(float)
    trend=p[f"ma{cfg.trend_ma}"].to_numpy(float)
    rec_mid=p[f"ma{cfg.rec_mid_ma}"].to_numpy(float)
    rec_fast=p[f"above_ma{cfg.rec_fast_ma}_{cfg.rec_fast_days}d"].to_numpy(bool)
    dd252=p.dd252.to_numpy(float)

    cash=1.0; tsh=0.0; tavg=np.nan; qsh=0.0
    pending=None; reason=None
    cycle_peak=np.nan; entry_started=False; tier=[False]*len(ENTRY_DD); red=False
    eqs=[]; tw=[]; qw=[]; states=[]; trades=[]

    for i in range(len(p)):
        if i>0: cash*=1+max(0.0,cy[i-1]/100)/252

        if pending is not None:
            bt,bq=pending; before_t=tsh
            cash,tsh,tavg,qsh,fee,nts=rebalance(cash,tsh,tavg,qsh,t[i],q[i],bt,bq)
            if nts:
                trades.append((dates[i],reason,bt,bq,t[i],q[i],fee,";".join(f"{a}:{n:.8f}" for a,n in nts)))
            if before_t>0 and tsh==0 and reason=="TP_50" and not red:
                cycle_peak=q[i]; entry_started=False; tier=[False]*len(ENTRY_DD)
        pending=None; reason=None

        eq=cash+tsh*t[i]+qsh*q[i]
        tweight=tsh*t[i]/eq if eq>0 else 0; qweight=qsh*q[i]/eq if eq>0 else 0
        eqs.append(eq); tw.append(tweight); qw.append(qweight)

        trend_down=False
        if i>=cfg.slope_lb and np.isfinite(trend[i]) and np.isfinite(trend[i-cfg.slope_lb]):
            trend_down=q[i]<trend[i] and trend[i]<trend[i-cfg.slope_lb]
        red_signal=trend_down and np.isfinite(dd252[i]) and dd252[i] <= cfg.red_dd
        if red_signal: red=True

        if red:
            states.append("RED")
            if red_signal and (abs(tweight-cfg.red_tqqq)>.015 or abs(qweight-cfg.red_qqq)>.015):
                pending=(cfg.red_tqqq,cfg.red_qqq); reason="RED_rotate"; continue

            if np.isfinite(trend[i]) and q[i]>trend[i]:
                red=False
                # Exit defensive QQQ, but keep recovery TQQQ if any.
                if qweight>.01:
                    pending=(tweight,0.0); reason="RED_release_QQQ"
                cycle_peak=q[i]; entry_started=False; tier=[False]*len(ENTRY_DD)
                continue

            # staged re-risk: replace QQQ with TQQQ as trend heals
            if np.isfinite(rec_mid[i]) and q[i]>rec_mid[i]:
                if abs(tweight-.60)>.02 or qweight>.02:
                    pending=(.60,0.0); reason="RECOVERY_T60" 
                continue
            if rec_fast[i]:
                if tweight<.35-.02 or qweight>.02:
                    pending=(.35,0.0); reason="RECOVERY_T35"
                continue
            continue

        states.append("YELLOW" if trend_down else "GREEN")

        # No QQQ outside RED/recovery states.
        if qweight>.01:
            pending=(tweight,0.0); reason="NORMAL_clear_QQQ"; continue

        if tsh>0 and np.isfinite(tavg) and t[i]>=tavg*(1+TP):
            pending=(0.0,0.0); reason="TP_50"; continue

        if not entry_started:
            cycle_peak=q[i] if not np.isfinite(cycle_peak) else max(cycle_peak,q[i])
        dd=q[i]/cycle_peak-1 if np.isfinite(cycle_peak) and cycle_peak>0 else 0
        fired=None
        for j,th in enumerate(ENTRY_DD):
            if dd<=th and not tier[j]: fired=j
        if fired is not None:
            for j in range(fired+1): tier[j]=True
            entry_started=True
            target=TARGET_W[fired]
            if trend_down: target=min(target,cfg.yellow_buy_cap)
            if target>tweight+1e-6:
                pending=(target,0.0); reason=f"dip_{ENTRY_DD[fired]:.0%}_T{target:.0%}"

    eq=pd.Series(eqs,index=dates); tws=pd.Series(tw,index=dates); qws=pd.Series(qw,index=dates)
    tr=pd.DataFrame(trades,columns=["date","reason","target_tqqq","target_qqq","tqqq_px","qqq_px","fee","notionals"])
    path=None
    if save_path:
        path=pd.DataFrame({"date":dates,"qqq":q,"tqqq":t,"equity":eqs,"tqqq_weight":tw,"qqq_weight":qw,"state":states})
    return eq,tws,qws,tr,path


def main():
    p=add_features(load_panel())
    for ma in (100,120,150):
        if f"ma{ma}" not in p: p[f"ma{ma}"]=p.qqq.rolling(ma,min_periods=ma).mean()
    actual=p.loc[p.price_source.eq("actual"),"date"].min()

    cfgs=[]
    for ma in (100,120,150):
      for slope in (5,10,20):
       for ycap in (.65,.80):
        for rdd in (-.18,-.20,-.22,-.25):
         for rt in (0.0,.10):
          for rq in (.25,.50,.75,1.0):
           if rt+rq<=1.0: cfgs.append(V5(ma,slope,ycap,rdd,rt,rq,60))

    rows=[]
    for cfg in cfgs:
        eq,tw,qw,tr,_=simulate(p,cfg)
        r={"strategy":cfg.name,"trend_ma":cfg.trend_ma,"slope_lb":cfg.slope_lb,
           "yellow_buy_cap":cfg.yellow_buy_cap,"red_dd":cfg.red_dd,"red_tqqq":cfg.red_tqqq,"red_qqq":cfg.red_qqq}
        r.update(perf(eq)); r.update(period_metrics(eq)); am=perf(eq.loc[actual:])
        r["ActualEra_CAGR"]=am["CAGR"]; r["ActualEra_MDD"]=am["MDD"]
        r["RiskDayPct"]=float(((tw+qw)>.01).mean()); r["AvgTQQQWeight"]=float(tw.mean()); r["AvgQQQWeight"]=float(qw.mean()); r["TradeCount"]=len(tr)
        rows.append(r)
    df=pd.DataFrame(rows); df.to_csv(OUT/"v5_sweep.csv",index=False)

    fronts=[]
    for afloor in (.18,.20,.22,.24):
      for cap in (-.60,-.55,-.50,-.45,-.40):
        e=df[(df.ActualEra_CAGR>=afloor)&(df.DotCom_MDD>=cap)&(df.MDD>=cap-.03)]
        if len(e):
            z=e.sort_values(["CAGR","ActualEra_CAGR","GFC_MDD"],ascending=[False,False,False]).iloc[0].copy()
            z["actual_floor"]=afloor; z["dotcom_cap"]=cap; fronts.append(z)
    front=pd.DataFrame(fronts); front.to_csv(OUT/"v5_frontier.csv",index=False)

    # Neighborhood sensitivity and rolling robustness for candidates.
    sens=(df.groupby(["trend_ma","slope_lb","red_dd","red_qqq"],as_index=False)
          .agg(best_CAGR=("CAGR","max"),median_CAGR=("CAGR","median"),best_MDD=("MDD","max"),
               best_ActualEra_CAGR=("ActualEra_CAGR","max")))
    sens.to_csv(OUT/"v5_sensitivity.csv",index=False)
    names=list(dict.fromkeys(list(df.sort_values("CAGR",ascending=False).head(15).strategy)+(list(front.strategy) if len(front) else [])))
    enr=[]
    for name in names:
        r=df.loc[df.strategy.eq(name)].iloc[0].to_dict()
        cfg=V5(r["trend_ma"],r["slope_lb"],r["yellow_buy_cap"],r["red_dd"],r["red_tqqq"],r["red_qqq"],60)
        eq,tw,qw,tr,path=simulate(p,cfg,True); r.update(rolling_metrics(eq,3)); r.update(rolling_metrics(eq,5)); enr.append(r)
        if name in set((front.strategy.tolist() if len(front) else [])[:8]):
            path.to_csv(OUT/f"path_{name}.csv",index=False); tr.to_csv(OUT/f"trades_{name}.csv",index=False)
    en=pd.DataFrame(enr); en.to_csv(OUT/"v5_candidates_rolling.csv",index=False)

    cols=["strategy","CAGR","MDD","DotCom_MDD","GFC_MDD","COVID_MDD","2022_Bear_MDD","ActualEra_CAGR","ActualEra_MDD","RiskDayPct","AvgTQQQWeight","AvgQQQWeight","TradeCount"]
    print("=== V5 TOP CAGR ===")
    print(df.sort_values("CAGR",ascending=False)[cols].head(20).to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\n=== V5 FRONTIER ===")
    if len(front): print(front[["actual_floor","dotcom_cap"]+cols].to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    else: print("none")
    print("\n=== V5 SENSITIVITY TOP ===")
    print(sens.sort_values(["best_CAGR","best_MDD"],ascending=[False,False]).head(30).to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\n=== V5 ROLLING ===")
    if len(en):
        c=["strategy","CAGR","MDD","ActualEra_CAGR","Roll3y_CAGR_P10","Roll3y_CAGR_Median","Roll5y_CAGR_P10","Roll5y_CAGR_Median"]
        print(en.sort_values("CAGR",ascending=False)[c].head(30).to_string(index=False,float_format=lambda x:f"{x:.4f}"))

if __name__=="__main__": main()
