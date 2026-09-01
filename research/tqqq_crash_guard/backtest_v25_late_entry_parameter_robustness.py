from __future__ import annotations

from pathlib import Path
import itertools
import numpy as np
import pandas as pd

from backtest_v1 import perf
from backtest_v13_validation import execute_target_var, RunSpec, simulate_spec
from backtest_v6_bull_hold import Cfg
from backtest_v23_ndx_history_extension import build_extended_panel, campaign_table

OUT=Path(__file__).resolve().parent/"results_v25_late_entry_robustness"
OUT.mkdir(parents=True,exist_ok=True)
BASE=Cfg(-0.25,80,110,0.35,110,50,-0.18)
COST=0.0005
MAS=(90,100,110,120,130)
SLOPES=(10,20,30,50)
FORMS=(
    ("I25",0.25,None,None),
    ("I50",0.50,None,None),
    ("L25_50_PB3",0.25,0.50,-0.03),
    ("L25_50_PB5",0.25,0.50,-0.05),
    ("L25_50_PB75",0.25,0.50,-0.075),
)


def ma_arrays(p):
    q=p.qqq.astype(float)
    return {ma:q.rolling(ma,min_periods=ma).mean().to_numpy(float) for ma in MAS}


def run_one(p,start_idx,end_idx,base_exit_date,init_w,add_target,pb,exit_ma,slope_lb,ma_map):
    dates=pd.DatetimeIndex(p.date); q=p.qqq.to_numpy(float); t=p.tqqq.to_numpy(float); cy=p.cash_yield_pct.to_numpy(float)
    ma=ma_map[exit_ma]
    cash=1.0; shares=0.0; avg=np.nan; pending=(start_idx+1,init_w,"INITIAL")
    running_high=q[start_idx]; add_fired=False; early_exited=False
    vals=[]; events=[]
    for i in range(start_idx,end_idx+1):
        if i>start_idx: cash*=1+max(0.0,cy[i-1]/100.0)/252.0
        if pending is not None and i>=pending[0]:
            _,target,reason=pending
            cash,shares,avg,notional,fee=execute_target_var(cash,shares,avg,t[i],target,COST)
            if abs(notional)>1e-12: events.append((dates[i],reason,target))
            if target<=1e-12 and reason=="EARLY_EXIT": early_exited=True
            pending=None
        if dates[i]==base_exit_date and shares>0:
            cash,shares,avg,notional,fee=execute_target_var(cash,shares,avg,t[i],0.0,COST)
            if abs(notional)>1e-12: events.append((dates[i],"BASE_EXIT",0.0))
        vals.append(cash+shares*t[i])
        running_high=max(running_high,q[i])
        if i>=end_idx or shares<=0 or early_exited or pending is not None: continue
        # Close signal -> next trading-day close execution.
        if i>=slope_lb and np.isfinite(ma[i]) and np.isfinite(ma[i-slope_lb]) and q[i]<ma[i] and ma[i]<ma[i-slope_lb]:
            pending=(min(i+1,end_idx),0.0,"EARLY_EXIT")
            continue
        if add_target is not None and not add_fired and q[i]/running_high-1.0<=pb:
            pending=(min(i+1,end_idx),add_target,"ADD")
            add_fired=True
    s=pd.Series(vals,index=dates[start_idx:end_idx+1]); m=perf(s)
    return m,events


def wait_perf(p,si,ei):
    dates=pd.DatetimeIndex(p.date); cy=p.cash_yield_pct.to_numpy(float); cash=1.0; vals=[]
    for i in range(si,ei+1):
        if i>si: cash*=1+max(0.0,cy[i-1]/100.0)/252.0
        vals.append(cash)
    return perf(pd.Series(vals,index=dates[si:ei+1]))


def main():
    p,_,_=build_extended_panel(); dates=pd.DatetimeIndex(p.date); ma_map=ma_arrays(p)
    _,_,tr,_=simulate_spec(p,RunSpec(BASE)); camps=campaign_table(p,tr)
    cfgs=[]
    for ma,sl,(form,iw,at,pb) in itertools.product(MAS,SLOPES,FORMS):
        cfgs.append((f"{form}_XMA{ma}_S{sl}",form,iw,at,pb,ma,sl))

    rows=[]
    for _,c in camps.iterrows():
        if bool(c.ongoing): continue
        fd=pd.Timestamp(c.full_date); ex=pd.Timestamp(c.exit_or_latest); fi=dates.get_loc(fd); ei=dates.get_loc(ex)
        for age in (505,756,879,1009):
            si=fi+age
            if si>=ei: continue
            wm=wait_perf(p,si,ei)
            for name,form,iw,at,pb,ma,sl in cfgs:
                m,ev=run_one(p,si,ei,ex,iw,at,pb,ma,sl,ma_map)
                rows.append({"campaign_full_date":fd,"age_td":age,"policy":name,"form":form,"exit_ma":ma,"slope_lb":sl,
                             "CAGR":m["CAGR"],"MDD":m["MDD"],"Return":m["Final"]-1.0,"WinWait":m["Final"]>wm["Final"],
                             "DiffCAGR":m["CAGR"]-wm["CAGR"],"event_count":len(ev)})
    raw=pd.DataFrame(rows); raw.to_csv(OUT/"v25_raw.csv",index=False)

    # Aggregate by exact current-like age and by all campaign-age checkpoints.
    cur=[]; overall=[]
    for name,z in raw[raw.age_td.eq(879)].groupby("policy"):
        r=z.iloc[0]
        cur.append({"policy":name,"form":r.form,"exit_ma":r.exit_ma,"slope_lb":r.slope_lb,"campaigns":z.campaign_full_date.nunique(),
                    "CAGR_Median":z.CAGR.median(),"CAGR_Min":z.CAGR.min(),"MDD_Median":z.MDD.median(),"MDD_Worst":z.MDD.min(),
                    "WinRate":z.WinWait.mean(),"DiffCAGR_Median":z.DiffCAGR.median(),"DiffCAGR_Min":z.DiffCAGR.min()})
    for name,z in raw.groupby("policy"):
        r=z.iloc[0]
        overall.append({"policy":name,"form":r.form,"exit_ma":r.exit_ma,"slope_lb":r.slope_lb,"n":len(z),"campaigns":z.campaign_full_date.nunique(),
                        "CAGR_Median":z.CAGR.median(),"CAGR_Min":z.CAGR.min(),"MDD_Median":z.MDD.median(),"MDD_Worst":z.MDD.min(),
                        "WinRate":z.WinWait.mean(),"DiffCAGR_Median":z.DiffCAGR.median(),"DiffCAGR_Min":z.DiffCAGR.min()})
    cdf=pd.DataFrame(cur); odf=pd.DataFrame(overall)
    cdf.to_csv(OUT/"v25_current_age_grid.csv",index=False); odf.to_csv(OUT/"v25_overall_grid.csv",index=False)

    # Predeclared selection constraints: no current-age loss in either completed analog, current-age MDD >= -35%,
    # and across all campaign-age checkpoints worst MDD >= -40% and minimum CAGR >= -10%.
    merged=cdf.merge(odf,on=["policy","form","exit_ma","slope_lb"],suffixes=("_cur","_all"))
    good=merged[(merged.CAGR_Min_cur>=0)&(merged.MDD_Worst_cur>=-0.35)&(merged.MDD_Worst_all>=-0.40)&(merged.CAGR_Min_all>=-0.10)].copy()
    good=good.sort_values(["CAGR_Median_cur","CAGR_Median_all","CAGR_Min_cur"],ascending=False)
    good.to_csv(OUT/"v25_robust_shortlist.csv",index=False)

    # Parameter plateau: among qualifying rules, summarize how often each MA/slope appears near the top.
    top=good.head(min(30,len(good))).copy()
    plateau=[]
    if len(top):
        for (ma,sl),z in top.groupby(["exit_ma","slope_lb"]):
            plateau.append({"exit_ma":ma,"slope_lb":sl,"top30_count":len(z),"mean_current_CAGR":z.CAGR_Median_cur.mean(),
                            "worst_current_MDD":z.MDD_Worst_cur.min(),"mean_all_CAGR":z.CAGR_Median_all.mean()})
    pdf=pd.DataFrame(plateau).sort_values(["top30_count","mean_current_CAGR"],ascending=False) if plateau else pd.DataFrame()
    pdf.to_csv(OUT/"v25_parameter_plateau.csv",index=False)

    # Current rule state for all MA/slope pairs.
    q=p.qqq.to_numpy(float); latest_i=len(p)-1
    state=[]
    for ma in MAS:
        for sl in SLOPES:
            m=ma_map[ma]
            sig=bool(latest_i>=sl and q[-1]<m[-1] and m[-1]<m[-1-sl])
            state.append({"latest_date":dates[-1],"exit_ma":ma,"slope_lb":sl,"exit_signal_today":sig,
                          "qqq_vs_ma":q[-1]/m[-1]-1,"ma_slope":m[-1]/m[-1-sl]-1})
    pd.DataFrame(state).to_csv(OUT/"v25_current_exit_state.csv",index=False)

    print(f"configs={len(cfgs)} observations={len(raw)} robust={len(good)}")
    print("=== ROBUST SHORTLIST TOP 25 ===")
    print(good.head(25).to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\n=== PARAMETER PLATEAU ===")
    print(pdf.to_string(index=False,float_format=lambda x:f"{x:.4f}") if len(pdf) else "none")
    print("\n=== CURRENT EXIT STATE ===")
    print(pd.DataFrame(state).to_string(index=False,float_format=lambda x:f"{x:.4f}"))

if __name__=="__main__":main()
