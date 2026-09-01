from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from backtest_v1 import perf
from backtest_v13_validation import RunSpec, simulate_spec
from backtest_v6_bull_hold import Cfg
from backtest_v23_ndx_history_extension import build_extended_panel, campaign_table
from backtest_v27_late_entry_ensemble_exit import ma_map, run_policy

OUT=Path(__file__).resolve().parent/"results_v28_late_entry_portfolio_budget"
OUT.mkdir(parents=True,exist_ok=True)
BASE=Cfg(-0.25,80,110,0.35,110,50,-0.18)

# Policy tuples must match V27 run_policy args: iw, add target, pb, family, k.
POLICIES={
    "I25_N3_K2":(0.25,None,None,"N3",2),
    "I50_N3_K2":(0.50,None,None,"N3",2),
    "L25_50_PB5_N3_K2":(0.25,0.50,-0.05,"N3",2),
    "HOLD100":(1.00,None,None,"NONE",0),
}
BUDGETS=(0.10,0.20,0.30,0.40)


def cash_index(p,si,ei):
    cy=p.cash_yield_pct.to_numpy(float)
    vals=[1.0]
    for i in range(si+1,ei+1):
        vals.append(vals[-1]*(1+max(0.0,cy[i-1]/100.0)/252.0))
    return np.asarray(vals,float)


def combine_portfolio(tactical_path,budget,cash_idx):
    te=tactical_path.equity.to_numpy(float)
    # Both components start at 1, with strategic reserve never touched.
    pe=(1-budget)*cash_idx+budget*te
    idx=pd.DatetimeIndex(tactical_path.date)
    return pd.Series(pe,index=idx)


def main():
    p,_,_=build_extended_panel(); dates=pd.DatetimeIndex(p.date); mm=ma_map(p)
    _,_,tr,_=simulate_spec(p,RunSpec(BASE)); camps=campaign_table(p,tr)
    rows=[]
    for _,c in camps.iterrows():
        if bool(c.ongoing):continue
        fd=pd.Timestamp(c.full_date); ex=pd.Timestamp(c.exit_or_latest); fi=dates.get_loc(fd); ei=dates.get_loc(ex)
        for age in (505,756,879,1009):
            si=fi+age
            if si>=ei:continue
            ci=cash_index(p,si,ei)
            cash_perf=perf(pd.Series(ci,index=dates[si:ei+1]))
            for pol,args in POLICIES.items():
                iw,at,pb,fam,k=args
                m,tpath,ev=run_policy(p,si,ei,ex,iw,at,pb,fam,k,mm,save_path=True)
                for b in BUDGETS:
                    pe=combine_portfolio(tpath,b,ci); pm=perf(pe)
                    rows.append({
                        "campaign_full_date":fd,"age_td":age,"policy":pol,"budget_pct":b,
                        "initial_total_tqqq_target":b*iw,
                        "post_add_total_tqqq_target_if_triggered":b*at if at is not None else np.nan,
                        "CAGR":pm["CAGR"],"MDD":pm["MDD"],"Return":pm["Final"]-1,
                        "cash_CAGR":cash_perf["CAGR"],"CAGR_increment_vs_cash":pm["CAGR"]-cash_perf["CAGR"],
                        "outperformed_cash":pm["Final"]>cash_perf["Final"],
                        "tactical_CAGR":m["CAGR"],"tactical_MDD":m["MDD"],
                        "max_tactical_tqqq_weight":float(tpath.weight.max()),
                        "max_total_tqqq_weight_approx":float((b*tpath.weight*tpath.equity/((1-b)*ci+b*tpath.equity)).max()),
                        "trade_events":len(ev),
                    })
    rdf=pd.DataFrame(rows); rdf.to_csv(OUT/"v28_raw.csv",index=False)

    summ=[]
    for (age,pol,b),z in rdf.groupby(["age_td","policy","budget_pct"]):
        summ.append({"age_td":age,"policy":pol,"budget_pct":b,"campaigns":z.campaign_full_date.nunique(),
                     "initial_total_tqqq_target":z.initial_total_tqqq_target.iloc[0],
                     "CAGR_Median":z.CAGR.median(),"CAGR_Min":z.CAGR.min(),"MDD_Median":z.MDD.median(),"MDD_Worst":z.MDD.min(),
                     "CAGR_Increment_Median":z.CAGR_increment_vs_cash.median(),"CAGR_Increment_Min":z.CAGR_increment_vs_cash.min(),
                     "WinRate_vs_cash":z.outperformed_cash.mean(),"Max_Total_TQQQ_Weight_Worst":z.max_total_tqqq_weight_approx.max()})
    sdf=pd.DataFrame(summ); sdf.to_csv(OUT/"v28_summary.csv",index=False)

    overall=[]
    for (pol,b),z in rdf.groupby(["policy","budget_pct"]):
        overall.append({"policy":pol,"budget_pct":b,"n":len(z),"campaigns":z.campaign_full_date.nunique(),
                        "initial_total_tqqq_target":z.initial_total_tqqq_target.iloc[0],
                        "CAGR_Median":z.CAGR.median(),"CAGR_Min":z.CAGR.min(),"MDD_Median":z.MDD.median(),"MDD_Worst":z.MDD.min(),
                        "CAGR_Increment_Median":z.CAGR_increment_vs_cash.median(),"CAGR_Increment_Min":z.CAGR_increment_vs_cash.min(),
                        "WinRate_vs_cash":z.outperformed_cash.mean(),"Max_Total_TQQQ_Weight_Worst":z.max_total_tqqq_weight_approx.max()})
    odf=pd.DataFrame(overall); odf.to_csv(OUT/"v28_overall.csv",index=False)

    cur=sdf[sdf.age_td.eq(879)].copy()
    # Decision table: prefer budget/rule combos with <=10% whole-portfolio worst MDD at current age and <=15% across all checkpoints.
    merged=cur.merge(odf,on=["policy","budget_pct"],suffixes=("_cur","_all"))
    good=merged[(merged.MDD_Worst_cur>=-0.10)&(merged.MDD_Worst_all>=-0.15)].copy()
    good=good.sort_values(["CAGR_Increment_Median_cur","CAGR_Increment_Median_all"],ascending=False)
    good.to_csv(OUT/"v28_low_risk_shortlist.csv",index=False)

    print("=== V28 CURRENT AGE 879 ===")
    cols=["policy","budget_pct","initial_total_tqqq_target","CAGR_Median","CAGR_Min","MDD_Worst","CAGR_Increment_Median","CAGR_Increment_Min","WinRate_vs_cash","Max_Total_TQQQ_Weight_Worst"]
    print(cur[cols].to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\n=== V28 OVERALL ===")
    print(odf[cols].sort_values(["policy","budget_pct"]).to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\n=== LOW-RISK SHORTLIST ===")
    print(good.to_string(index=False,float_format=lambda x:f"{x:.4f}") if len(good) else "none")

if __name__=="__main__":main()
