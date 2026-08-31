from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from backtest_v1 import load_panel, perf, period_metrics
from backtest_v2 import add_features
from backtest_v3 import V3, simulate_v3, rolling_metrics

OUT = Path(__file__).resolve().parent / "results_v4"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    p = add_features(load_panel())
    # Add faster MA features for V4.
    for ma in (100, 120):
        if f"ma{ma}" not in p:
            p[f"ma{ma}"] = p["qqq"].rolling(ma, min_periods=ma).mean()
    actual_start = p.loc[p.price_source.eq("actual"), "date"].min()

    rows=[]
    cfgs=[]
    for trend_ma in (100,120,150,180):
        for slope_lb in (5,10,20):
            for yellow_buy_cap in (.50,.65,.80):
                for red_dd in (-.12,-.15,-.18,-.20,-.22,-.25):
                    for red_target in (0.0,.15,.25):
                        for rec_fast_ma in (40,60):
                            cfgs.append(V3(trend_ma,slope_lb,yellow_buy_cap,red_dd,red_target,rec_fast_ma))

    for cfg in cfgs:
        eq,w,tr,_ = simulate_v3(p,cfg)
        r={
            "strategy":cfg.name.replace("V3_","V4_"),
            "trend_ma":cfg.trend_ma,"slope_lb":cfg.slope_lb,
            "yellow_buy_cap":cfg.yellow_buy_cap,"red_dd":cfg.red_dd,
            "red_target":cfg.red_target,"rec_fast_ma":cfg.rec_fast_ma,
        }
        r.update(perf(eq)); r.update(period_metrics(eq))
        am=perf(eq.loc[actual_start:])
        r["ActualEra_CAGR"]=am["CAGR"]; r["ActualEra_MDD"]=am["MDD"]
        r["InvestedDayPct"]=float((w>.01).mean()); r["AvgWeight"]=float(w.mean()); r["TradeCount"]=len(tr)
        rows.append(r)

    df=pd.DataFrame(rows)
    df.to_csv(OUT/"v4_sweep.csv",index=False)

    # Explicit efficient frontier by crash constraint, with actual-era return floors.
    fronts=[]
    for actual_floor in (.18,.20,.22,.24):
        for cap in (-.65,-.60,-.55,-.50,-.45,-.40):
            elig=df[(df.ActualEra_CAGR>=actual_floor)&(df.DotCom_MDD>=cap)&(df.MDD>=cap-.03)]
            if len(elig):
                z=elig.sort_values(["CAGR","ActualEra_CAGR","GFC_MDD"],ascending=[False,False,False]).iloc[0].copy()
                z["actual_floor"]=actual_floor; z["dotcom_cap"]=cap
                fronts.append(z)
    front=pd.DataFrame(fronts)
    front.to_csv(OUT/"v4_frontier.csv",index=False)

    # Parameter sensitivity summary: best CAGR for each MA/slope/red threshold neighborhood.
    sens=(df.groupby(["trend_ma","slope_lb","red_dd"],as_index=False)
          .agg(best_CAGR=("CAGR","max"),median_CAGR=("CAGR","median"),
               best_DotCom_MDD=("DotCom_MDD","max"),median_DotCom_MDD=("DotCom_MDD","median"),
               best_ActualEra_CAGR=("ActualEra_CAGR","max")))
    sens.to_csv(OUT/"v4_sensitivity.csv",index=False)

    # Focused candidate set: top CAGR and frontier winners, then rolling robustness.
    names=list(dict.fromkeys(list(df.sort_values("CAGR",ascending=False).head(15).strategy)+(list(front.strategy) if len(front) else [])))
    enriched=[]
    for name in names:
        r=df.loc[df.strategy.eq(name)].iloc[0].to_dict()
        cfg=V3(r["trend_ma"],r["slope_lb"],r["yellow_buy_cap"],r["red_dd"],r["red_target"],r["rec_fast_ma"])
        eq,w,tr,path=simulate_v3(p,cfg,save_path=True)
        r.update(rolling_metrics(eq,3)); r.update(rolling_metrics(eq,5))
        enriched.append(r)
    en=pd.DataFrame(enriched)
    en.to_csv(OUT/"v4_candidates_rolling.csv",index=False)

    cols=["strategy","CAGR","MDD","DotCom_MDD","GFC_MDD","COVID_MDD","2022_Bear_MDD","ActualEra_CAGR","ActualEra_MDD","InvestedDayPct","AvgWeight","TradeCount"]
    print("=== V4 TOP CAGR ===")
    print(df.sort_values("CAGR",ascending=False)[cols].head(20).to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\n=== V4 FRONTIER ===")
    if len(front):
        print(front[["actual_floor","dotcom_cap"]+cols].to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    else: print("none")
    print("\n=== V4 SENSITIVITY TOP ===")
    print(sens.sort_values(["best_CAGR","best_DotCom_MDD"],ascending=[False,False]).head(30).to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\n=== V4 ROLLING ===")
    if len(en):
        c2=["strategy","CAGR","MDD","ActualEra_CAGR","Roll3y_CAGR_P10","Roll3y_CAGR_Median","Roll3y_WorstMDD","Roll5y_CAGR_P10","Roll5y_CAGR_Median","Roll5y_WorstMDD"]
        print(en.sort_values("CAGR",ascending=False)[c2].head(30).to_string(index=False,float_format=lambda x:f"{x:.4f}"))

if __name__=="__main__":
    main()
