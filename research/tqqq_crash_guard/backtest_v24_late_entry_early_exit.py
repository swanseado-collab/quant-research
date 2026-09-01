from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from backtest_v1 import perf
from backtest_v13_validation import execute_target_var, RunSpec, simulate_spec
from backtest_v6_bull_hold import Cfg
from backtest_v23_ndx_history_extension import build_extended_panel, campaign_table

OUT=Path(__file__).resolve().parent/"results_v24_late_entry_early_exit"
OUT.mkdir(parents=True,exist_ok=True)
BASE=Cfg(-0.25,80,110,0.35,110,50,-0.18)
COST=0.0005

# name, initial weight, add target on 5% post-adoption QQQ pullback, early-exit rule
POLICIES=[
    ("HOLD25",0.25,None,"NONE"),
    ("HOLD50",0.50,None,"NONE"),
    ("HOLD100",1.00,None,"NONE"),
    ("I25_X60_3D",0.25,None,"MA60_3D"),
    ("I25_X80_3D",0.25,None,"MA80_3D"),
    ("I25_X100_3D",0.25,None,"MA100_3D"),
    ("I25_X110NEG20",0.25,None,"MA110_NEG20"),
    ("I25_XDD63_10",0.25,None,"DD63_10"),
    ("I50_X60_3D",0.50,None,"MA60_3D"),
    ("I50_X80_3D",0.50,None,"MA80_3D"),
    ("I50_X100_3D",0.50,None,"MA100_3D"),
    ("I50_X110NEG20",0.50,None,"MA110_NEG20"),
    ("L25_50_PB5_X60",0.25,0.50,"MA60_3D"),
    ("L25_50_PB5_X80",0.25,0.50,"MA80_3D"),
    ("L25_50_PB5_X100",0.25,0.50,"MA100_3D"),
    ("L25_50_PB5_X110NEG20",0.25,0.50,"MA110_NEG20"),
    ("WAIT_NEXT",0.0,None,"NONE"),
]


def feature_arrays(p):
    q=p.qqq.astype(float)
    f={}
    for ma in (60,80,100,110):
        m=q.rolling(ma,min_periods=ma).mean()
        f[f"ma{ma}"]=m.to_numpy(float)
        below=(q<m).rolling(3,min_periods=3).sum().eq(3)
        f[f"below{ma}_3d"]=below.to_numpy(bool)
    f["hi63"]=q.rolling(63,min_periods=63).max().to_numpy(float)
    return f


def early_exit_signal(rule,i,q,f):
    if rule=="NONE": return False
    if rule=="MA60_3D": return bool(f["below60_3d"][i])
    if rule=="MA80_3D": return bool(f["below80_3d"][i])
    if rule=="MA100_3D": return bool(f["below100_3d"][i])
    if rule=="MA110_NEG20":
        if i<20 or not np.isfinite(f["ma110"][i]) or not np.isfinite(f["ma110"][i-20]): return False
        return bool(q[i]<f["ma110"][i] and f["ma110"][i]<f["ma110"][i-20])
    if rule=="DD63_10":
        return bool(np.isfinite(f["hi63"][i]) and q[i]/f["hi63"][i]-1.0<=-0.10)
    raise ValueError(rule)


def run_late_policy(p,start_idx,end_idx,base_exit_date,init_w,add_target,exit_rule):
    dates=pd.DatetimeIndex(p.date); q=p.qqq.to_numpy(float); t=p.tqqq.to_numpy(float); cy=p.cash_yield_pct.to_numpy(float)
    f=feature_arrays(p)
    cash=1.0; shares=0.0; avg=np.nan
    pending=None  # (exec_idx,target,reason); every signal executes at t+1 close
    running_high=q[start_idx]
    add_fired=False; early_exited=False
    events=[]; vals=[]

    if init_w>0 and start_idx+1<=end_idx:
        pending=(start_idx+1,init_w,"INITIAL")

    for i in range(start_idx,end_idx+1):
        if i>start_idx:
            cash*=1+max(0.0,cy[i-1]/100.0)/252.0

        # Execute previously scheduled order before using today's close for a new signal.
        if pending is not None and i>=pending[0]:
            _,target,reason=pending
            cash,shares,avg,notional,fee=execute_target_var(cash,shares,avg,t[i],target,COST)
            if abs(notional)>1e-12: events.append((dates[i],reason,target,t[i],notional,fee))
            if target<=1e-12 and reason.startswith("EARLY_EXIT"):
                early_exited=True
            pending=None

        # Existing V7 campaign's scheduled exit is an already-defined external execution date.
        if dates[i]==base_exit_date and shares>0:
            cash,shares,avg,notional,fee=execute_target_var(cash,shares,avg,t[i],0.0,COST)
            if abs(notional)>1e-12: events.append((dates[i],"BASE_V7_EXIT",0.0,t[i],notional,fee))

        eq=cash+shares*t[i]
        vals.append(eq)
        running_high=max(running_high,q[i])

        if i>=end_idx or shares<=0 or early_exited or pending is not None:
            continue

        # Early exit has priority over adding risk. Signal at today's close, execute next close.
        if early_exit_signal(exit_rule,i,q,f):
            pending=(min(i+1,end_idx),0.0,f"EARLY_EXIT_{exit_rule}")
            continue

        if add_target is not None and not add_fired:
            dd_post=q[i]/running_high-1.0
            if dd_post<=-0.05:
                pending=(min(i+1,end_idx),add_target,"ADD_PB5")
                add_fired=True

    s=pd.Series(vals,index=dates[start_idx:end_idx+1]); m=perf(s)
    ev=pd.DataFrame(events,columns=["date","reason","target","price","notional","fee"])
    return m,ev


def run_wait(p,start_idx,end_idx):
    dates=pd.DatetimeIndex(p.date); cy=p.cash_yield_pct.to_numpy(float)
    cash=1.0; vals=[]
    for i in range(start_idx,end_idx+1):
        if i>start_idx: cash*=1+max(0.0,cy[i-1]/100.0)/252.0
        vals.append(cash)
    return perf(pd.Series(vals,index=dates[start_idx:end_idx+1]))


def main():
    p,_,_=build_extended_panel(); dates=pd.DatetimeIndex(p.date)
    _,_,tr,_=simulate_spec(p,RunSpec(BASE))
    camps=campaign_table(p,tr)
    rows=[]; event_rows=[]
    for _,c in camps.iterrows():
        if bool(c.ongoing): continue
        fd=pd.Timestamp(c.full_date); ex=pd.Timestamp(c.exit_or_latest); fi=dates.get_loc(fd); ei=dates.get_loc(ex)
        for age in (505,756,879,1009):
            si=fi+age
            if si>=ei: continue
            wm=run_wait(p,si,ei)
            for name,iw,at,xr in POLICIES:
                if name=="WAIT_NEXT":
                    m=wm; ev=pd.DataFrame()
                else:
                    m,ev=run_late_policy(p,si,ei,ex,iw,at,xr)
                rows.append({"campaign_full_date":fd,"start_date":dates[si],"exit_date":ex,"age_td":age,"policy":name,
                             "CAGR":m["CAGR"],"MDD":m["MDD"],"Return":m["Final"]-1.0,
                             "DiffCAGR_vs_Wait":m["CAGR"]-wm["CAGR"],"Win_vs_Wait":m["Final"]>wm["Final"],
                             "event_count":len(ev)})
                if len(ev):
                    tmp=ev.copy(); tmp["campaign_full_date"]=fd; tmp["start_date"]=dates[si]; tmp["age_td"]=age; tmp["policy"]=name
                    event_rows.append(tmp)
    raw=pd.DataFrame(rows); raw.to_csv(OUT/"v24_raw.csv",index=False)
    events=pd.concat(event_rows,ignore_index=True) if event_rows else pd.DataFrame(); events.to_csv(OUT/"v24_events.csv",index=False)

    summ=[]
    for (age,pol),z in raw.groupby(["age_td","policy"]):
        summ.append({"age_td":age,"policy":pol,"campaigns":z.campaign_full_date.nunique(),"CAGR_Median":z.CAGR.median(),"CAGR_Min":z.CAGR.min(),
                     "MDD_Median":z.MDD.median(),"MDD_Worst":z.MDD.min(),"Return_Median":z.Return.median(),"Return_Min":z.Return.min(),
                     "WinRate_vs_Wait":z.Win_vs_Wait.mean(),"DiffCAGR_Median":z.DiffCAGR_vs_Wait.median(),"DiffCAGR_Min":z.DiffCAGR_vs_Wait.min()})
    sdf=pd.DataFrame(summ); sdf.to_csv(OUT/"v24_summary.csv",index=False)

    # Across all independent campaign-age checkpoints: avoid choosing a rule that only works at age 879.
    overall=[]
    for pol,z in raw.groupby("policy"):
        overall.append({"policy":pol,"n_campaign_age":len(z),"campaigns":z.campaign_full_date.nunique(),"CAGR_Median":z.CAGR.median(),"CAGR_Min":z.CAGR.min(),
                        "MDD_Median":z.MDD.median(),"MDD_Worst":z.MDD.min(),"WinRate_vs_Wait":z.Win_vs_Wait.mean(),
                        "DiffCAGR_Median":z.DiffCAGR_vs_Wait.median(),"DiffCAGR_Min":z.DiffCAGR_vs_Wait.min()})
    odf=pd.DataFrame(overall).sort_values(["CAGR_Median","MDD_Worst"],ascending=[False,False]); odf.to_csv(OUT/"v24_overall.csv",index=False)

    # Current-state readout: signals only, no trade execution.
    latest_i=len(p)-1; q=p.qqq.to_numpy(float); f=feature_arrays(p)
    latest=dates[-1]; current_full=max(pd.to_datetime(tr.loc[tr.reason.eq("BULL_FULL"),"date"])); current_age=latest_i-dates.get_loc(current_full)
    state=[]
    for name,iw,at,xr in POLICIES:
        state.append({"policy":name,"latest_date":latest,"current_full_date":current_full,"current_age_td":current_age,
                      "initial_weight_if_adopted":iw,"add_target":at,"early_exit_rule":xr,
                      "early_exit_signal_today":early_exit_signal(xr,latest_i,q,f) if xr!="NONE" else False,
                      "qqq_dd63":q[-1]/f["hi63"][-1]-1 if np.isfinite(f["hi63"][-1]) else np.nan,
                      "qqq_vs_ma60":q[-1]/f["ma60"][-1]-1,"qqq_vs_ma80":q[-1]/f["ma80"][-1]-1,
                      "qqq_vs_ma100":q[-1]/f["ma100"][-1]-1,"qqq_vs_ma110":q[-1]/f["ma110"][-1]-1,
                      "ma110_slope20":f["ma110"][-1]/f["ma110"][-21]-1 if np.isfinite(f["ma110"][-21]) else np.nan})
    pd.DataFrame(state).to_csv(OUT/"v24_current_state.csv",index=False)

    # Predeclared robust shortlist: current-age rule must have positive minimum CAGR and worst MDD no worse than -35%.
    z=sdf[sdf.age_td.eq(879)].copy(); good=z[(z.CAGR_Min>=0)&(z.MDD_Worst>=-0.35)].sort_values(["CAGR_Median","CAGR_Min"],ascending=False)
    good.to_csv(OUT/"v24_current_age_shortlist.csv",index=False)

    print("=== V24 AGE 879 ===")
    print(z.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\n=== V24 OVERALL ===")
    print(odf.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\n=== V24 CURRENT-AGE ROBUST SHORTLIST ===")
    print(good.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\n=== V24 CURRENT STATE ===")
    print(pd.DataFrame(state).head(12).to_string(index=False,float_format=lambda x:f"{x:.4f}"))

if __name__=="__main__": main()
