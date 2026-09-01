from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from backtest_v1 import perf, period_metrics
from backtest_v6_bull_hold import Cfg
from backtest_v13_validation import RunSpec, simulate_spec
from backtest_v15_cycle_and_synth_stress import build_live_panel
from backtest_v16_sleeve_sizing import buy_with_cost, sell_with_cost

OUT = Path(__file__).resolve().parent / "results_v17_ratchet_harvest"
OUT.mkdir(parents=True, exist_ok=True)
BASE = Cfg(-0.25, 80, 110, 0.35, 110, 50, -0.18)

RATCHETS = {
    "NONE": [],
    "MILD": [(0.60,0.50),(0.75,0.65),(0.90,0.80)],
    "BAL":  [(0.40,0.30),(0.60,0.45),(0.80,0.60)],
    "DEF":  [(0.30,0.20),(0.50,0.35),(0.70,0.50)],
}


def simulate_ratchet(p, base_tr, budget_pct, ratchet_name):
    ratchets = RATCHETS[ratchet_name]
    dates = pd.DatetimeIndex(p.date)
    px = p.tqqq.to_numpy(float)
    cy = p.cash_yield_pct.to_numpy(float)
    trade_map = {pd.Timestamp(r.date): r for _, r in base_tr.iterrows()}

    core_cash=1.0; tact_cash=0.0; shares=0.0; in_campaign=False
    fired=[False]*len(ratchets); pending_idx=None
    rows=[]; trades=[]

    for i,d in enumerate(dates):
        if i>0:
            rate=max(0.0,cy[i-1]/100.0)/252.0
            core_cash*=1+rate; tact_cash*=1+rate

        if d in trade_map:
            r=trade_map[d]; reason=str(r.reason); target=float(r.target_weight)
            if reason=="REVERSAL_ENTRY":
                total=core_cash+tact_cash+shares*px[i]
                if not in_campaign:
                    budget=min(core_cash,budget_pct*total)
                    core_cash-=budget; tact_cash+=budget
                    in_campaign=True; fired=[False]*len(ratchets); pending_idx=None
                camp=tact_cash+shares*px[i]; desired=target*camp; current=shares*px[i]
                if desired>current:
                    tact_cash,shares,fee=buy_with_cost(tact_cash,shares,px[i],desired-current)
                    trades.append((d,reason,target,fee))
            elif reason=="BULL_FULL" and in_campaign:
                camp=tact_cash+shares*px[i]; desired=target*camp; current=shares*px[i]
                if desired>current:
                    tact_cash,shares,fee=buy_with_cost(tact_cash,shares,px[i],desired-current)
                    trades.append((d,reason,target,fee))
            elif reason=="BULL_EXIT" and in_campaign:
                shares,proceeds,fee=sell_with_cost(shares,px[i],shares*px[i])
                tact_cash+=proceeds; core_cash+=tact_cash; tact_cash=0.0
                in_campaign=False; pending_idx=None
                trades.append((d,reason,0.0,fee))

        # Execute one scheduled ratchet trim at today's close, after any V7 trade.
        if pending_idx is not None and in_campaign and shares>0:
            trigger,target=ratchets[pending_idx]
            total=core_cash+tact_cash+shares*px[i]
            desired=target*total; current=shares*px[i]
            if current>desired:
                shares,proceeds,fee=sell_with_cost(shares,px[i],current-desired)
                core_cash+=proceeds
                trades.append((d,f"RATCHET_{int(trigger*100)}_TO_{int(target*100)}",target,fee))
            fired[pending_idx]=True; pending_idx=None

        total=core_cash+tact_cash+shares*px[i]
        tw=shares*px[i]/total if total>0 else 0.0
        rows.append((d,total,tw,core_cash,tact_cash,shares*px[i],in_campaign))

        # Schedule the lowest not-yet-fired threshold crossed at this close.
        if in_campaign and pending_idx is None:
            for j,(trigger,target) in enumerate(ratchets):
                if not fired[j] and tw>=trigger:
                    pending_idx=j
                    break

    path=pd.DataFrame(rows,columns=["date","equity","tqqq_weight","core_cash","tactical_cash","tqqq_value","in_campaign"])
    eq=pd.Series(path.equity.to_numpy(),index=pd.DatetimeIndex(path.date))
    tr=pd.DataFrame(trades,columns=["date","reason","target","fee"])
    return eq,path,tr


def main():
    p=build_live_panel()
    _,_,base_tr,_=simulate_spec(p,RunSpec(BASE))
    actual_start=p.loc[p.price_source.eq("actual"),"date"].min()
    rows=[]
    for budget in (0.10,0.15,0.20,0.25,0.30):
        for rn in RATCHETS:
            eq,path,tr=simulate_ratchet(p,base_tr,budget,rn)
            r={"budget_pct":budget,"ratchet":rn,**perf(eq),**period_metrics(eq),
               "Max_TQQQ_Weight":float(path.tqqq_weight.max()),
               "Avg_TQQQ_Weight":float(path.tqqq_weight.mean()),
               "RatchetTrades":int(tr.reason.str.startswith("RATCHET").sum()) if len(tr) else 0,
               "TotalTradeEvents":len(tr),"FinalMultiple":float(eq.iloc[-1]/eq.iloc[0])}
            am=perf(eq.loc[actual_start:]); r["ActualEra_CAGR"]=am["CAGR"]; r["ActualEra_MDD"]=am["MDD"]
            rows.append(r)
            if budget in (0.15,0.20,0.25) and rn in ("NONE","MILD","BAL"):
                tag=f"B{int(budget*100)}_{rn}"
                path.to_csv(OUT/f"path_{tag}.csv",index=False); tr.to_csv(OUT/f"trades_{tag}.csv",index=False)
    df=pd.DataFrame(rows)
    df.to_csv(OUT/"v17_grid.csv",index=False)

    # Return-first frontier: for each tolerated MDD, show maximum CAGR.
    fronts=[]
    for floor in (-0.55,-0.50,-0.45,-0.40,-0.35,-0.30,-0.25,-0.20):
        z=df[df.MDD>=floor]
        if len(z):
            a=z.sort_values(["CAGR","ActualEra_CAGR"],ascending=False).iloc[0].copy(); a["MDD_floor"]=floor; fronts.append(a)
    front=pd.DataFrame(fronts); front.to_csv(OUT/"v17_frontier.csv",index=False)

    print("=== V17 GRID ===")
    cols=["budget_pct","ratchet","CAGR","MDD","ActualEra_CAGR","ActualEra_MDD","Max_TQQQ_Weight","RatchetTrades","FinalMultiple"]
    print(df[cols].sort_values(["budget_pct","ratchet"]).to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\n=== V17 RETURN-FIRST FRONTIER ===")
    if len(front): print(front[["MDD_floor"]+cols].to_string(index=False,float_format=lambda x:f"{x:.4f}"))

if __name__=="__main__": main()
