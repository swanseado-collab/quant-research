from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from backtest_v1 import perf, period_metrics
from backtest_v6_bull_hold import Cfg
from backtest_v13_validation import RunSpec, simulate_spec
from backtest_v15_cycle_and_synth_stress import build_live_panel
from backtest_v16_sleeve_sizing import buy_with_cost, sell_with_cost
from backtest_v17_ratchet_harvest import RATCHETS
from backtest_v18_core_overlay import build_core_indices

OUT = Path(__file__).resolve().parent / "results_v19_cash_funded_overlay"
OUT.mkdir(parents=True, exist_ok=True)
BASE = Cfg(-0.25,80,110,0.35,110,50,-0.18)


def simulate_baseline(p, core_idx, core_pct):
    dates=pd.DatetimeIndex(p.date); cy=p.cash_yield_pct.to_numpy(float)
    cpx=core_idx.reindex(dates).to_numpy(float)
    core_units=core_pct/cpx[0]; reserve=1.0-core_pct
    rows=[]
    for i,d in enumerate(dates):
        if i>0: reserve*=1+max(0.0,cy[i-1]/100.0)/252.0
        rows.append((d,core_units*cpx[i]+reserve,core_units*cpx[i],reserve))
    path=pd.DataFrame(rows,columns=["date","equity","core_value","reserve_cash"])
    eq=pd.Series(path.equity.to_numpy(),index=dates)
    return eq,path


def simulate_overlay(p,base_tr,core_idx,core_pct,budget_pct,ratchet_name):
    ratchets=RATCHETS[ratchet_name]
    dates=pd.DatetimeIndex(p.date); tpx=p.tqqq.to_numpy(float); cy=p.cash_yield_pct.to_numpy(float)
    cpx=core_idx.reindex(dates).to_numpy(float)
    trade_map={pd.Timestamp(r.date):r for _,r in base_tr.iterrows()}

    core_units=core_pct/cpx[0]
    reserve=1.0-core_pct; tact_cash=0.0; shares=0.0; in_campaign=False
    fired=[False]*len(ratchets); pending_idx=None
    rows=[]; trades=[]; fundings=[]

    for i,d in enumerate(dates):
        if i>0:
            rate=max(0.0,cy[i-1]/100.0)/252.0
            reserve*=1+rate; tact_cash*=1+rate

        if d in trade_map:
            r=trade_map[d]; reason=str(r.reason); target=float(r.target_weight)
            if reason=="REVERSAL_ENTRY":
                total=core_units*cpx[i]+reserve+tact_cash+shares*tpx[i]
                if not in_campaign:
                    desired_budget=budget_pct*total
                    funded=min(reserve,desired_budget)
                    reserve-=funded; tact_cash+=funded
                    fundings.append((d,desired_budget,funded,funded/total if total>0 else np.nan))
                    in_campaign=funded>0; fired=[False]*len(ratchets); pending_idx=None
                if in_campaign:
                    camp=tact_cash+shares*tpx[i]; desired=target*camp; current=shares*tpx[i]
                    tact_cash,shares,fee=buy_with_cost(tact_cash,shares,tpx[i],max(0,desired-current))
                    trades.append((d,reason,target,fee))
            elif reason=="BULL_FULL" and in_campaign:
                camp=tact_cash+shares*tpx[i]; desired=target*camp; current=shares*tpx[i]
                tact_cash,shares,fee=buy_with_cost(tact_cash,shares,tpx[i],max(0,desired-current))
                trades.append((d,reason,target,fee))
            elif reason=="BULL_EXIT" and in_campaign:
                shares,proceeds,fee=sell_with_cost(shares,tpx[i],shares*tpx[i]); tact_cash+=proceeds
                reserve+=tact_cash; tact_cash=0.0; in_campaign=False; pending_idx=None
                trades.append((d,reason,0.0,fee))

        if pending_idx is not None and in_campaign and shares>0:
            trigger,target=ratchets[pending_idx]
            total=core_units*cpx[i]+reserve+tact_cash+shares*tpx[i]
            current=shares*tpx[i]; desired=target*total
            if current>desired:
                shares,proceeds,fee=sell_with_cost(shares,tpx[i],current-desired)
                reserve+=proceeds
                trades.append((d,f"RATCHET_{int(trigger*100)}_TO_{int(target*100)}",target,fee))
            fired[pending_idx]=True; pending_idx=None

        total=core_units*cpx[i]+reserve+tact_cash+shares*tpx[i]
        tw=shares*tpx[i]/total if total>0 else 0.0
        rows.append((d,total,tw,core_units*cpx[i],reserve,tact_cash,shares*tpx[i],in_campaign))
        if in_campaign and pending_idx is None:
            for j,(trigger,target) in enumerate(ratchets):
                if not fired[j] and tw>=trigger:
                    pending_idx=j; break

    path=pd.DataFrame(rows,columns=["date","equity","tqqq_weight","core_value","reserve_cash","tactical_cash","tqqq_value","in_campaign"])
    eq=pd.Series(path.equity.to_numpy(),index=dates)
    tr=pd.DataFrame(trades,columns=["date","reason","target","fee"])
    fd=pd.DataFrame(fundings,columns=["date","desired_budget","funded_budget","actual_budget_pct"])
    return eq,path,tr,fd


def main():
    p=build_live_panel(); dates=pd.DatetimeIndex(p.date); cores=build_core_indices(p)
    _,_,base_tr,_=simulate_spec(p,RunSpec(BASE)); actual_start=p.loc[p.price_source.eq("actual"),"date"].min()
    bench=[]; rows=[]
    for cn,idx in cores.items():
        for cp in (0.40,0.60,0.80):
            beq,bpath=simulate_baseline(p,idx,cp)
            bm={"core":cn,"core_pct":cp,**perf(beq),**period_metrics(beq),"FinalMultiple":float(beq.iloc[-1]/beq.iloc[0])}
            am=perf(beq.loc[actual_start:]); bm["ActualEra_CAGR"]=am["CAGR"]; bm["ActualEra_MDD"]=am["MDD"]
            bench.append(bm)
            max_budget=1.0-cp+1e-12
            for bp in (0.10,0.15,0.20,0.25):
                if bp>max_budget: continue
                for rn in ("NONE","MILD","BAL"):
                    eq,path,tr,fd=simulate_overlay(p,base_tr,idx,cp,bp,rn)
                    pm=perf(eq); am2=perf(eq.loc[actual_start:])
                    r={"core":cn,"core_pct":cp,"initial_cash_pct":1-cp,"budget_pct":bp,"ratchet":rn,
                       **pm,**period_metrics(eq),"FinalMultiple":float(eq.iloc[-1]/eq.iloc[0]),
                       "ActualEra_CAGR":am2["CAGR"],"ActualEra_MDD":am2["MDD"],
                       "Max_TQQQ_Weight":float(path.tqqq_weight.max()),"Avg_TQQQ_Weight":float(path.tqqq_weight.mean()),
                       "Min_Reserve_Cash_Pct":float((path.reserve_cash/path.equity).min()),
                       "RatchetTrades":int(tr.reason.str.startswith("RATCHET").sum()) if len(tr) else 0,
                       "Avg_Actual_Funding_Pct":float(fd.actual_budget_pct.mean()) if len(fd) else np.nan,
                       "Min_Actual_Funding_Pct":float(fd.actual_budget_pct.min()) if len(fd) else np.nan,
                       "Baseline_CAGR":bm["CAGR"],"Baseline_MDD":bm["MDD"],
                       "CAGR_Increment":pm["CAGR"]-bm["CAGR"],"MDD_Change":pm["MDD"]-bm["MDD"]}
                    rows.append(r)
                    if cn=="SPY_QQQ_50_50" and cp in (0.60,0.80) and bp in (0.10,0.15,0.20) and rn in ("NONE","MILD"):
                        tag=f"MIX_C{int(cp*100)}_B{int(bp*100)}_{rn}"
                        path.to_csv(OUT/f"path_{tag}.csv",index=False); tr.to_csv(OUT/f"trades_{tag}.csv",index=False); fd.to_csv(OUT/f"funding_{tag}.csv",index=False)
    bdf=pd.DataFrame(bench); df=pd.DataFrame(rows)
    bdf.to_csv(OUT/"v19_baselines.csv",index=False); df.to_csv(OUT/"v19_grid.csv",index=False)

    fronts=[]
    for cn in cores:
        for cp in (0.40,0.60,0.80):
            z=df[(df.core==cn)&(df.core_pct==cp)]
            for floor in (-0.60,-0.55,-0.50,-0.45,-0.40,-0.35,-0.30,-0.25):
                zz=z[z.MDD>=floor]
                if len(zz):
                    a=zz.sort_values(["CAGR","ActualEra_CAGR"],ascending=False).iloc[0].copy(); a["MDD_floor"]=floor; fronts.append(a)
    fdf=pd.DataFrame(fronts); fdf.to_csv(OUT/"v19_frontier.csv",index=False)

    print("=== V19 BASELINES ===")
    print(bdf[["core","core_pct","CAGR","MDD","ActualEra_CAGR","ActualEra_MDD"]].to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\n=== V19 MIXED-CORE SELECTED ===")
    cols=["core","core_pct","budget_pct","ratchet","CAGR","MDD","ActualEra_CAGR","ActualEra_MDD","Max_TQQQ_Weight","CAGR_Increment","MDD_Change","Min_Actual_Funding_Pct"]
    print(df[(df.core=="SPY_QQQ_50_50")][cols].to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\n=== V19 FRONTIER ===")
    if len(fdf): print(fdf[["MDD_floor"]+cols].to_string(index=False,float_format=lambda x:f"{x:.4f}"))

if __name__=="__main__": main()
