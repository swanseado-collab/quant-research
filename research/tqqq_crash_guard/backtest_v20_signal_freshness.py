from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from backtest_v1 import perf
from backtest_v6_bull_hold import Cfg
from backtest_v13_validation import RunSpec, simulate_spec, execute_target_var
from backtest_v15_cycle_and_synth_stress import build_live_panel

OUT=Path(__file__).resolve().parent/"results_v20_signal_freshness"
OUT.mkdir(parents=True,exist_ok=True)
BASE=Cfg(-0.25,80,110,0.35,110,50,-0.18)
COST=0.0005


def mini_run(p, base_tr, start_idx, end_idx, policy):
    dates=pd.DatetimeIndex(p.date); px=p.tqqq.to_numpy(float); cy=p.cash_yield_pct.to_numpy(float)
    trade_map={pd.Timestamp(r.date):r for _,r in base_tr.iterrows()}
    cash=1.0; shares=0.0; avg=np.nan; active=False
    jump_idx=start_idx+1 if policy=="JUMP" and start_idx+1<=end_idx else None
    eq=[]
    for i in range(start_idx,end_idx+1):
        if i>start_idx:
            cash*=1+max(0.0,cy[i-1]/100.0)/252.0
        d=dates[i]
        if jump_idx is not None and i==jump_idx:
            cash,shares,avg,_,_=execute_target_var(cash,shares,avg,px[i],1.0,COST)
            active=True
        if d in trade_map and i>start_idx:
            r=trade_map[d]; reason=str(r.reason); target=float(r.target_weight)
            if policy=="WAIT":
                if not active and reason=="REVERSAL_ENTRY":
                    cash,shares,avg,_,_=execute_target_var(cash,shares,avg,px[i],target,COST); active=True
                elif active and reason in ("BULL_FULL","BULL_EXIT"):
                    cash,shares,avg,_,_=execute_target_var(cash,shares,avg,px[i],target,COST)
                    if reason=="BULL_EXIT": active=False
            else: # JUMP: join current campaign, then follow all later campaigns normally
                if active and reason in ("BULL_FULL","BULL_EXIT"):
                    cash,shares,avg,_,_=execute_target_var(cash,shares,avg,px[i],target,COST)
                    if reason=="BULL_EXIT": active=False
                elif not active and reason=="REVERSAL_ENTRY":
                    cash,shares,avg,_,_=execute_target_var(cash,shares,avg,px[i],target,COST); active=True
        eq.append(cash+shares*px[i])
    s=pd.Series(eq,index=dates[start_idx:end_idx+1])
    return perf(s)


def age_bucket(age):
    if age<=126:return "000_126"
    if age<=252:return "127_252"
    if age<=504:return "253_504"
    if age<=756:return "505_756"
    if age<=1008:return "757_1008"
    return "1009_plus"


def main():
    p=build_live_panel(); dates=pd.DatetimeIndex(p.date)
    _,_,tr,path=simulate_spec(p,RunSpec(BASE),save_path=True)
    fulls=[pd.Timestamp(x) for x in tr.loc[tr.reason.eq("BULL_FULL"),"date"]]
    exits=[pd.Timestamp(x) for x in tr.loc[tr.reason.eq("BULL_EXIT"),"date"]]

    # Map each date to the most recent still-active BULL_FULL and trading-day age.
    full_idx={d:dates.get_loc(d) for d in fulls}
    current_full=None; current_full_i=None
    active_full=[]
    for i,d in enumerate(dates):
        if d in full_idx:
            current_full=d; current_full_i=i
        if d in exits:
            current_full=None; current_full_i=None
        state=str(path.iloc[i].state)
        if state=="BULL" and current_full is not None:
            active_full.append((i,d,current_full,i-current_full_i))

    # Monthly adoption dates only, to reduce extreme overlap.
    af=pd.DataFrame(active_full,columns=["idx","date","full_date","age_td"])
    af["month"]=af.date.dt.to_period("M")
    starts=af.groupby("month",as_index=False).first()
    rows=[]
    for _,r in starts.iterrows():
        si=int(r.idx); sd=pd.Timestamp(r.date); age=int(r.age_td)
        for yrs in (3,5):
            target=sd+pd.DateOffset(years=yrs)
            valid=np.flatnonzero(dates<=target)
            if len(valid)==0: continue
            ei=int(valid[-1])
            # Require almost the full requested horizon; otherwise this is censored and excluded.
            if (dates[ei]-sd).days < int(365.2425*yrs)-10: continue
            j=mini_run(p,tr,si,ei,"JUMP"); w=mini_run(p,tr,si,ei,"WAIT")
            rows.append({"start_date":sd,"full_date":pd.Timestamp(r.full_date),"age_td":age,"age_bucket":age_bucket(age),"horizon_y":yrs,
                         "Jump_CAGR":j["CAGR"],"Jump_MDD":j["MDD"],"Jump_Final":j["Final"],
                         "Wait_CAGR":w["CAGR"],"Wait_MDD":w["MDD"],"Wait_Final":w["Final"],
                         "CAGR_Diff":j["CAGR"]-w["CAGR"],"Jump_Outperformed":j["Final"]>w["Final"]})
    raw=pd.DataFrame(rows); raw.to_csv(OUT/"v20_monthly_adoption_raw.csv",index=False)

    summ=[]
    for (h,b),z in raw.groupby(["horizon_y","age_bucket"]):
        summ.append({"horizon_y":h,"age_bucket":b,"n_month_starts":len(z),"n_distinct_campaigns":z.full_date.nunique(),
                     "Jump_CAGR_Median":z.Jump_CAGR.median(),"Jump_CAGR_P10":z.Jump_CAGR.quantile(.10),
                     "Wait_CAGR_Median":z.Wait_CAGR.median(),"CAGR_Diff_Median":z.CAGR_Diff.median(),"CAGR_Diff_P10":z.CAGR_Diff.quantile(.10),
                     "Jump_WinRate":z.Jump_Outperformed.mean(),"Jump_MDD_Median":z.Jump_MDD.median(),"Jump_MDD_Worst":z.Jump_MDD.min()})
    sdf=pd.DataFrame(summ); sdf.to_csv(OUT/"v20_age_bucket_summary.csv",index=False)

    # Current signal age and historical analogs. Current row itself has no future outcome and is not in raw.
    latest_i=len(p)-1; latest_date=dates[-1]
    prior_full=[d for d in fulls if d<=latest_date]
    current_full=max(prior_full) if prior_full else pd.NaT
    cf_i=dates.get_loc(current_full) if pd.notna(current_full) else np.nan
    current_age=int(latest_i-cf_i) if pd.notna(current_full) else np.nan
    current_bucket=age_bucket(current_age) if pd.notna(current_age) else ""
    analog=raw[(raw.age_td>=current_age-126)&(raw.age_td<=current_age+126)].copy() if pd.notna(current_age) else raw.iloc[0:0]
    analog.to_csv(OUT/"v20_current_age_analogs.csv",index=False)
    current=pd.DataFrame([{"latest_date":latest_date,"current_full_date":current_full,"current_age_trading_days":current_age,"current_age_bucket":current_bucket,
                           "analog_window_age_min":current_age-126 if pd.notna(current_age) else np.nan,"analog_window_age_max":current_age+126 if pd.notna(current_age) else np.nan,
                           "analog_rows":len(analog),"analog_distinct_campaigns":analog.full_date.nunique() if len(analog) else 0}])
    current.to_csv(OUT/"v20_current_signal_age.csv",index=False)

    # Exact-age cross-cycle snapshots: enter at approximately today's bull age in each completed sufficiently-long campaign, hold to that campaign exit.
    snapshots=[]
    for fd in fulls:
        fi=dates.get_loc(fd); desired_i=fi+current_age if pd.notna(current_age) else fi
        later=[x for x in exits if x>fd]
        if not later: continue
        ex=later[0]; exi=dates.get_loc(ex)
        if desired_i>=exi: continue
        entry_i=desired_i; entry_d=dates[entry_i]
        z=pd.Series(p.tqqq.iloc[entry_i:exi+1].to_numpy(float),index=dates[entry_i:exi+1])
        ret=float(z.iloc[-1]/z.iloc[0]-1); mdd=float((z/z.cummax()-1).min())
        yrs=(z.index[-1]-z.index[0]).days/365.2425; cagr=(z.iloc[-1]/z.iloc[0])**(1/yrs)-1 if yrs>0 else np.nan
        snapshots.append({"campaign_full_date":fd,"late_entry_date":entry_d,"exit_date":ex,"matched_age_td":current_age,
                          "remaining_calendar_days":(ex-entry_d).days,"TQQQ_Remaining_Return":ret,"TQQQ_Remaining_CAGR":cagr,"TQQQ_Remaining_MDD":mdd})
    snap=pd.DataFrame(snapshots); snap.to_csv(OUT/"v20_exact_current_age_completed_cycles.csv",index=False)

    print("=== CURRENT SIGNAL AGE ===")
    print(current.to_string(index=False))
    print("\n=== AGE BUCKET SUMMARY ===")
    print(sdf.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\n=== EXACT CURRENT-AGE COMPLETED CYCLES ===")
    print(snap.to_string(index=False,float_format=lambda x:f"{x:.4f}") if len(snap) else "none")

if __name__=="__main__": main()
