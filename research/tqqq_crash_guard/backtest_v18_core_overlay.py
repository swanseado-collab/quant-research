from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from backtest_v1 import perf, period_metrics
from backtest_v6_bull_hold import Cfg
from backtest_v13_validation import RunSpec, simulate_spec
from backtest_v14_live_extension import yf_series, END
from backtest_v15_cycle_and_synth_stress import build_live_panel
from backtest_v16_sleeve_sizing import buy_with_cost, sell_with_cost
from backtest_v17_ratchet_harvest import RATCHETS

OUT = Path(__file__).resolve().parent / "results_v18_core_overlay"
OUT.mkdir(parents=True, exist_ok=True)
BASE = Cfg(-0.25, 80, 110, 0.35, 110, 50, -0.18)
COST = 0.0005


def build_core_indices(p: pd.DataFrame):
    dates = pd.DatetimeIndex(p.date)
    spy = yf_series("SPY", start="1999-01-01", end=END).reindex(dates).ffill().bfill()
    qqq = pd.Series(p.qqq.to_numpy(float), index=dates)
    spy = spy / spy.iloc[0]
    qqq = qqq / qqq.iloc[0]
    # Daily-rebalanced 50/50 return basket: a clean proxy for mixed US equity core.
    sr = spy.pct_change().fillna(0.0)
    qr = qqq.pct_change().fillna(0.0)
    mix = (1.0 + 0.5*sr + 0.5*qr).cumprod()
    return {"SPY": spy, "QQQ": qqq, "SPY_QQQ_50_50": mix}


def core_buy(core_cash, core_units, core_px, dollars):
    buy = min(max(0.0, dollars), core_cash / (1.0 + COST))
    if buy <= 0: return core_cash, core_units, 0.0
    fee = buy*COST
    core_units += buy/core_px
    core_cash -= buy+fee
    return core_cash, core_units, fee


def core_sell(core_units, core_px, dollars):
    current = core_units*core_px
    sell = min(max(0.0,dollars), current)
    if sell<=0: return core_units, 0.0, 0.0
    fee=sell*COST
    core_units -= sell/core_px
    if core_units<1e-12: core_units=0.0
    return core_units, sell-fee, fee


def simulate_overlay(p, base_tr, core_idx: pd.Series, budget_pct: float, ratchet_name: str):
    ratchets=RATCHETS[ratchet_name]
    dates=pd.DatetimeIndex(p.date); tpx=p.tqqq.to_numpy(float); cy=p.cash_yield_pct.to_numpy(float)
    cpx=core_idx.reindex(dates).to_numpy(float)
    trade_map={pd.Timestamp(r.date):r for _,r in base_tr.iterrows()}

    core_units=1.0/cpx[0]; core_cash=0.0
    tact_cash=0.0; shares=0.0; in_campaign=False
    fired=[False]*len(ratchets); pending_idx=None
    rows=[]; trades=[]

    for i,d in enumerate(dates):
        if i>0:
            tact_cash *= 1.0 + max(0.0,cy[i-1]/100.0)/252.0
            # core_cash is only transient intra-close; normally zero. If nonzero, let it earn T-bill too.
            core_cash *= 1.0 + max(0.0,cy[i-1]/100.0)/252.0

        if d in trade_map:
            r=trade_map[d]; reason=str(r.reason); target=float(r.target_weight)
            if reason=="REVERSAL_ENTRY":
                total=core_units*cpx[i]+core_cash+tact_cash+shares*tpx[i]
                if not in_campaign:
                    budget=budget_pct*total
                    core_units, proceeds, fee1=core_sell(core_units,cpx[i],budget)
                    tact_cash += proceeds
                    in_campaign=True; fired=[False]*len(ratchets); pending_idx=None
                else: fee1=0.0
                camp=tact_cash+shares*tpx[i]; desired=target*camp; current=shares*tpx[i]
                tact_cash,shares,fee2=buy_with_cost(tact_cash,shares,tpx[i],max(0,desired-current))
                trades.append((d,reason,target,fee1+fee2))
            elif reason=="BULL_FULL" and in_campaign:
                camp=tact_cash+shares*tpx[i]; desired=target*camp; current=shares*tpx[i]
                tact_cash,shares,fee=buy_with_cost(tact_cash,shares,tpx[i],max(0,desired-current))
                trades.append((d,reason,target,fee))
            elif reason=="BULL_EXIT" and in_campaign:
                shares,proceeds,fee1=sell_with_cost(shares,tpx[i],shares*tpx[i]); tact_cash+=proceeds
                core_cash += tact_cash; tact_cash=0.0
                core_cash,core_units,fee2=core_buy(core_cash,core_units,cpx[i],core_cash/(1.0+COST))
                in_campaign=False; pending_idx=None
                trades.append((d,reason,0.0,fee1+fee2))

        if pending_idx is not None and in_campaign and shares>0:
            trigger,target=ratchets[pending_idx]
            total=core_units*cpx[i]+core_cash+tact_cash+shares*tpx[i]
            desired=target*total; current=shares*tpx[i]
            if current>desired:
                shares,proceeds,fee1=sell_with_cost(shares,tpx[i],current-desired)
                core_cash += proceeds
                core_cash,core_units,fee2=core_buy(core_cash,core_units,cpx[i],core_cash/(1.0+COST))
                trades.append((d,f"RATCHET_{int(trigger*100)}_TO_{int(target*100)}",target,fee1+fee2))
            fired[pending_idx]=True; pending_idx=None

        total=core_units*cpx[i]+core_cash+tact_cash+shares*tpx[i]
        tw=shares*tpx[i]/total if total>0 else 0.0
        rows.append((d,total,tw,core_units*cpx[i]+core_cash,tact_cash,shares*tpx[i],in_campaign))
        if in_campaign and pending_idx is None:
            for j,(trigger,target) in enumerate(ratchets):
                if not fired[j] and tw>=trigger:
                    pending_idx=j; break

    path=pd.DataFrame(rows,columns=["date","equity","tqqq_weight","core_value","tactical_cash","tqqq_value","in_campaign"])
    eq=pd.Series(path.equity.to_numpy(),index=dates)
    tr=pd.DataFrame(trades,columns=["date","reason","target","fee"])
    return eq,path,tr


def main():
    p=build_live_panel(); dates=pd.DatetimeIndex(p.date)
    cores=build_core_indices(p)
    _,_,base_tr,_=simulate_spec(p,RunSpec(BASE))
    actual_start=p.loc[p.price_source.eq("actual"),"date"].min()

    bench=[]; rows=[]
    for cn,idx in cores.items():
        beq=pd.Series(idx.to_numpy(),index=dates)
        bm={"core":cn,**perf(beq),**period_metrics(beq)}
        am=perf(beq.loc[actual_start:]); bm["ActualEra_CAGR"]=am["CAGR"]; bm["ActualEra_MDD"]=am["MDD"]
        bench.append(bm)
        for budget in (0.10,0.15,0.20,0.25):
            for rn in ("NONE","MILD","BAL"):
                eq,path,tr=simulate_overlay(p,base_tr,idx,budget,rn)
                r={"core":cn,"budget_pct":budget,"ratchet":rn,**perf(eq),**period_metrics(eq),
                   "Max_TQQQ_Weight":float(path.tqqq_weight.max()),"Avg_TQQQ_Weight":float(path.tqqq_weight.mean()),
                   "RatchetTrades":int(tr.reason.str.startswith("RATCHET").sum()) if len(tr) else 0,
                   "FinalMultiple":float(eq.iloc[-1]/eq.iloc[0]),
                   "CoreOnly_CAGR":bm["CAGR"],"CoreOnly_MDD":bm["MDD"],
                   "CAGR_Increment":perf(eq)["CAGR"]-bm["CAGR"],"MDD_Change":perf(eq)["MDD"]-bm["MDD"]}
                am2=perf(eq.loc[actual_start:]); r["ActualEra_CAGR"]=am2["CAGR"]; r["ActualEra_MDD"]=am2["MDD"]
                rows.append(r)
                if budget in (0.15,0.20) and rn in ("NONE","MILD"):
                    tag=f"{cn}_B{int(budget*100)}_{rn}".replace("/","_")
                    path.to_csv(OUT/f"path_{tag}.csv",index=False); tr.to_csv(OUT/f"trades_{tag}.csv",index=False)

    bdf=pd.DataFrame(bench); df=pd.DataFrame(rows)
    bdf.to_csv(OUT/"v18_core_benchmarks.csv",index=False); df.to_csv(OUT/"v18_overlay_grid.csv",index=False)

    fronts=[]
    for cn in cores:
        z=df[df.core==cn]
        for floor in (-0.60,-0.55,-0.50,-0.45,-0.40,-0.35,-0.30):
            zz=z[z.MDD>=floor]
            if len(zz):
                a=zz.sort_values(["CAGR","ActualEra_CAGR"],ascending=False).iloc[0].copy(); a["MDD_floor"]=floor; fronts.append(a)
    fdf=pd.DataFrame(fronts); fdf.to_csv(OUT/"v18_frontier.csv",index=False)

    print("=== CORE BENCHMARKS ===")
    print(bdf[["core","CAGR","MDD","ActualEra_CAGR","ActualEra_MDD"]].to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\n=== OVERLAY GRID ===")
    cols=["core","budget_pct","ratchet","CAGR","MDD","ActualEra_CAGR","ActualEra_MDD","Max_TQQQ_Weight","CAGR_Increment","MDD_Change","RatchetTrades"]
    print(df[cols].to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\n=== FRONTIER ===")
    if len(fdf): print(fdf[["MDD_floor"]+cols].to_string(index=False,float_format=lambda x:f"{x:.4f}"))

if __name__=="__main__": main()
