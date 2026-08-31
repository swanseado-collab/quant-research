from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from backtest_v1 import load_panel, execute_target, perf, period_metrics, ENTRY_DD, TARGET_W, TP
from backtest_v2 import add_features

OUT = Path(__file__).resolve().parent / "results_v3"
OUT.mkdir(parents=True, exist_ok=True)


class V3:
    def __init__(self, trend_ma, slope_lb, yellow_buy_cap, red_dd, red_target, rec_fast_ma):
        self.trend_ma = int(trend_ma)
        self.slope_lb = int(slope_lb)
        self.yellow_buy_cap = float(yellow_buy_cap)
        self.red_dd = float(red_dd)
        self.red_target = float(red_target)
        self.rec_fast_ma = int(rec_fast_ma)
        self.rec_fast_days = 3
        self.rec_mid_ma = 120
        self.release_ma = int(trend_ma)

    @property
    def name(self):
        return (
            f"V3_M{self.trend_ma}_S{self.slope_lb}_YBC{int(self.yellow_buy_cap*100)}_"
            f"RD{abs(int(self.red_dd*100))}_RT{int(self.red_target*100)}_RF{self.rec_fast_ma}"
        )


def simulate_v3(p: pd.DataFrame, cfg: V3, save_path=False):
    q = p["qqq"].to_numpy(float)
    lev = p["tqqq"].to_numpy(float)
    dates = pd.DatetimeIndex(p["date"])
    cash_y = p["cash_yield_pct"].to_numpy(float)
    trend = p[f"ma{cfg.trend_ma}"].to_numpy(float)
    rec_mid = p[f"ma{cfg.rec_mid_ma}"].to_numpy(float)
    rec_fast_ok = p[f"above_ma{cfg.rec_fast_ma}_{cfg.rec_fast_days}d"].to_numpy(bool)
    dd252 = p["dd252"].to_numpy(float)

    cash, shares, avg_cost = 1.0, 0.0, np.nan
    pending_target = None
    pending_reason = None
    cycle_peak = np.nan
    entry_started = False
    tier_used = [False] * len(ENTRY_DD)
    red_mode = False

    eq_arr = np.empty(len(p), float)
    w_arr = np.empty(len(p), float)
    state_arr = np.empty(len(p), object)
    trades = []

    for i in range(len(p)):
        if i > 0:
            cash *= 1.0 + max(0.0, cash_y[i-1] / 100.0) / 252.0

        if pending_target is not None:
            exec_reason = pending_reason
            before_shares = shares
            cash, shares, avg_cost, notional, fee = execute_target(cash, shares, avg_cost, lev[i], pending_target)
            if abs(notional) > 1e-12:
                trades.append((dates[i], exec_reason, pending_target, lev[i], notional, fee, cash, shares, avg_cost))
            # A completed +50% take-profit ends the current dip-buy cycle.
            # Without this reset, later re-entry can be accidentally disabled because
            # tier_used remains latched from the old cycle.
            if before_shares > 0 and shares == 0 and exec_reason == "TP_50" and not red_mode:
                cycle_peak = q[i]
                entry_started = False
                tier_used = [False] * len(ENTRY_DD)
        pending_target = None
        pending_reason = None

        equity = cash + shares * lev[i]
        weight = shares * lev[i] / equity if equity > 0 else 0.0
        eq_arr[i], w_arr[i] = equity, weight

        trend_down = False
        if i >= cfg.slope_lb and np.isfinite(trend[i]) and np.isfinite(trend[i-cfg.slope_lb]):
            trend_down = q[i] < trend[i] and trend[i] < trend[i-cfg.slope_lb]
        red_signal = trend_down and np.isfinite(dd252[i]) and dd252[i] <= cfg.red_dd
        if red_signal:
            red_mode = True

        if red_mode:
            state_arr[i] = "RED"
            if red_signal and weight > cfg.red_target + 1e-6:
                pending_target = cfg.red_target
                pending_reason = "RED_cut"
                continue

            if np.isfinite(trend[i]) and q[i] > trend[i]:
                red_mode = False
                cycle_peak = q[i]
                entry_started = False
                tier_used = [False] * len(ENTRY_DD)
                continue

            if np.isfinite(rec_mid[i]) and q[i] > rec_mid[i]:
                target = 0.60
                if weight < target - 1e-6:
                    pending_target = target
                    pending_reason = "RED_recovery_60"
                continue
            if rec_fast_ok[i]:
                target = 0.35
                if weight < target - 1e-6:
                    pending_target = target
                    pending_reason = "RED_recovery_35"
                continue
            continue

        state_arr[i] = "YELLOW" if trend_down else "GREEN"

        if shares > 0 and np.isfinite(avg_cost) and lev[i] >= avg_cost * (1.0 + TP):
            pending_target = 0.0
            pending_reason = "TP_50"
            continue

        # YELLOW does not force-sell existing holdings. It only caps NEW dip buys.
        if not entry_started:
            cycle_peak = q[i] if not np.isfinite(cycle_peak) else max(cycle_peak, q[i])

        dd = q[i] / cycle_peak - 1.0 if np.isfinite(cycle_peak) and cycle_peak > 0 else 0.0
        fired = None
        for j, threshold in enumerate(ENTRY_DD):
            if dd <= threshold and not tier_used[j]:
                fired = j
        if fired is not None:
            for j in range(fired + 1):
                tier_used[j] = True
            entry_started = True
            target = TARGET_W[fired]
            if trend_down:
                target = min(target, cfg.yellow_buy_cap)
            if target > weight + 1e-6:
                pending_target = target
                pending_reason = f"dip_{ENTRY_DD[fired]:.0%}_to_{target:.0%}"

    eq = pd.Series(eq_arr, index=dates)
    w = pd.Series(w_arr, index=dates)
    tr = pd.DataFrame(trades, columns=["date","reason","target_weight","price","notional","fee","cash_after","shares_after","avg_cost_after"])
    path = None
    if save_path:
        path = pd.DataFrame({"date":dates,"qqq":q,"tqqq":lev,"equity":eq_arr,"weight":w_arr,"state":state_arr})
    return eq, w, tr, path


def rolling_metrics(eq: pd.Series, years: int) -> dict:
    rows=[]
    starts = pd.date_range(eq.index.min(), eq.index.max()-pd.DateOffset(years=years), freq="QS")
    for s in starts:
        e=s+pd.DateOffset(years=years)
        z=eq.loc[s:e]
        if len(z)<years*200:
            continue
        m=perf(z)
        rows.append((s,m["CAGR"],m["MDD"]))
    if not rows:
        return {f"Roll{years}y_CAGR_P10":np.nan,f"Roll{years}y_CAGR_Median":np.nan,f"Roll{years}y_WorstMDD":np.nan}
    d=pd.DataFrame(rows,columns=["start","cagr","mdd"])
    return {
        f"Roll{years}y_CAGR_P10":float(d.cagr.quantile(.10)),
        f"Roll{years}y_CAGR_Median":float(d.cagr.median()),
        f"Roll{years}y_WorstMDD":float(d.mdd.min()),
    }


def summarize(cfg,eq,w,tr,actual_start):
    r={
        "strategy":cfg.name,"trend_ma":cfg.trend_ma,"slope_lb":cfg.slope_lb,
        "yellow_buy_cap":cfg.yellow_buy_cap,"red_dd":cfg.red_dd,"red_target":cfg.red_target,
        "rec_fast_ma":cfg.rec_fast_ma,
    }
    r.update(perf(eq)); r.update(period_metrics(eq))
    am=perf(eq.loc[actual_start:]); r["ActualEra_CAGR"]=am["CAGR"]; r["ActualEra_MDD"]=am["MDD"]
    r["InvestedDayPct"]=float((w>.01).mean()); r["AvgWeight"]=float(w.mean()); r["TradeCount"]=len(tr)
    return r


def main():
    p=add_features(load_panel())
    actual_start=p.loc[p.price_source.eq("actual"),"date"].min()
    rows=[]; cfgs=[]
    for trend_ma in (150,180,200):
        for slope_lb in (20,40):
            for yellow_buy_cap in (.50,.60,.70,.80):
                for red_dd in (-.20,-.25,-.30):
                    for red_target in (0.0,.15,.25):
                        for rec_fast_ma in (40,60):
                            cfgs.append(V3(trend_ma,slope_lb,yellow_buy_cap,red_dd,red_target,rec_fast_ma))
    for cfg in cfgs:
        eq,w,tr,_=simulate_v3(p,cfg)
        rows.append(summarize(cfg,eq,w,tr,actual_start))
    df=pd.DataFrame(rows)

    tier_rows=[]
    for cap in (-.65,-.60,-.55,-.50,-.45,-.40):
        elig=df[(df.DotCom_MDD>=cap)&(df.MDD>=cap-.03)&(df.ActualEra_CAGR>=.18)]
        if len(elig):
            z=elig.sort_values(["CAGR","ActualEra_CAGR","GFC_MDD"],ascending=[False,False,False]).iloc[0].copy()
            z["dotcom_cap"]=cap; tier_rows.append(z)
    tiers=pd.DataFrame(tier_rows)

    z=df.copy()
    z["score"]=4*z.CAGR + 2*z.ActualEra_CAGR + .8*z.DotCom_MDD + .4*z.GFC_MDD + .2*z["2022_Bear_MDD"]
    z=z.sort_values("score",ascending=False)

    candidate_names=list(dict.fromkeys(
        list(df.sort_values("CAGR",ascending=False).head(15).strategy)+
        (list(tiers.strategy) if len(tiers) else [])+
        list(z.head(15).strategy)
    ))
    enriched=[]
    for name in candidate_names:
        r=df.loc[df.strategy.eq(name)].iloc[0].to_dict()
        cfg=V3(r["trend_ma"],r["slope_lb"],r["yellow_buy_cap"],r["red_dd"],r["red_target"],r["rec_fast_ma"])
        eq,w,tr,path=simulate_v3(p,cfg,save_path=True)
        r.update(rolling_metrics(eq,3)); r.update(rolling_metrics(eq,5)); enriched.append(r)
        path.to_csv(OUT/f"path_{name}.csv",index=False)
        tr.to_csv(OUT/f"trades_{name}.csv",index=False)
    en=pd.DataFrame(enriched)

    df.to_csv(OUT/"v3_sweep.csv",index=False)
    tiers.to_csv(OUT/"v3_risk_tiers.csv",index=False)
    z.head(100).to_csv(OUT/"v3_top100_score.csv",index=False)
    en.to_csv(OUT/"v3_candidates_rolling.csv",index=False)

    cols=["strategy","CAGR","MDD","DotCom_MDD","GFC_MDD","COVID_MDD","2022_Bear_MDD","ActualEra_CAGR","ActualEra_MDD","InvestedDayPct","AvgWeight","TradeCount"]
    print("=== V3 TOP CAGR ===")
    print(df.sort_values("CAGR",ascending=False)[cols].head(15).to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\n=== V3 RISK TIERS (actual-era CAGR >=18%) ===")
    if len(tiers): print(tiers[["dotcom_cap"]+cols].to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    else: print("none")
    print("\n=== V3 TOP SCORE ===")
    print(z[cols+["score"]].head(15).to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    if len(en):
        print("\n=== V3 ROLLING ROBUSTNESS ===")
        c2=["strategy","CAGR","MDD","ActualEra_CAGR","Roll3y_CAGR_P10","Roll3y_CAGR_Median","Roll3y_WorstMDD","Roll5y_CAGR_P10","Roll5y_CAGR_Median","Roll5y_WorstMDD"]
        print(en.sort_values(["CAGR"],ascending=False)[c2].head(20).to_string(index=False,float_format=lambda x:f"{x:.4f}"))

if __name__=="__main__":
    main()
