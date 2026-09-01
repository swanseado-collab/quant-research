from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from backtest_v1 import perf
from backtest_v6_bull_hold import Cfg
from backtest_v13_validation import RunSpec, simulate_spec, execute_target_var
from backtest_v15_cycle_and_synth_stress import build_live_panel

OUT = Path(__file__).resolve().parent / "results_v22_late_join_ladder"
OUT.mkdir(parents=True, exist_ok=True)
BASE = Cfg(-0.25,80,110,0.35,110,50,-0.18)
COST = 0.0005

# (name, initial_weight, add_target, pullback_threshold, reference)
# reference='post' means drawdown from max QQQ observed after adoption.
# reference='r63' means drawdown from the observable rolling 63-trading-day QQQ high.
POLICIES = [
    ("IMM25",0.25,None,None,None),
    ("IMM50",0.50,None,None,None),
    ("IMM100",1.00,None,None,None),
    ("L25_TO50_PB5_POST",0.25,0.50,-0.05,"post"),
    ("L25_TO100_PB5_POST",0.25,1.00,-0.05,"post"),
    ("L50_TO100_PB5_POST",0.50,1.00,-0.05,"post"),
    ("L25_TO100_PB5_R63",0.25,1.00,-0.05,"r63"),
    ("L25_TO100_PB10_R63",0.25,1.00,-0.10,"r63"),
    ("WAIT_NEXT",0.0,None,None,None),
]


def bucket(age):
    if age<=756:return "505_756"
    if age<=1008:return "757_1008"
    return "1009_plus"


def run_policy(p,tr,start_idx,end_idx,init_w,add_target,pb_thresh,ref):
    dates=pd.DatetimeIndex(p.date); q=p.qqq.to_numpy(float); t=p.tqqq.to_numpy(float); cy=p.cash_yield_pct.to_numpy(float)
    tmap={pd.Timestamp(r.date):r for _,r in tr.iterrows()}
    cash=1.0; shares=0.0; avg=np.nan
    joined=False; added=False; future=False
    running_high=q[start_idx]
    rolling63=pd.Series(q).rolling(63,min_periods=1).max().to_numpy(float)
    init_date=pd.NaT; add_date=pd.NaT; add_dd=np.nan
    vals=[]

    for i in range(start_idx,end_idx+1):
        if i>start_idx: cash*=1+max(0.0,cy[i-1]/100.0)/252.0
        d=dates[i]; running_high=max(running_high,q[i])

        # Process a known V7 campaign exit before any stale-signal add on the same close.
        reason = str(tmap[d].reason) if d in tmap and i>start_idx else None
        target = float(tmap[d].target_weight) if d in tmap and i>start_idx else None
        if reason=="BULL_EXIT":
            if shares>0:
                cash,shares,avg,_,_=execute_target_var(cash,shares,avg,t[i],0.0,COST)
            joined=False; added=False; future=True

        if i==start_idx+1 and init_w>0 and not future:
            cash,shares,avg,_,_=execute_target_var(cash,shares,avg,t[i],init_w,COST)
            joined=shares>0; init_date=d

        if joined and add_target is not None and not added and not future and i>start_idx+1:
            dd=(q[i]/running_high-1.0) if ref=="post" else (q[i]/rolling63[i]-1.0)
            if dd<=pb_thresh:
                cash,shares,avg,_,_=execute_target_var(cash,shares,avg,t[i],add_target,COST)
                added=True; add_date=d; add_dd=dd

        # After stale campaign is over, all policies follow subsequent fresh V7 campaigns identically.
        if future and reason=="REVERSAL_ENTRY":
            cash,shares,avg,_,_=execute_target_var(cash,shares,avg,t[i],target,COST)
        elif future and reason=="BULL_FULL" and shares>0:
            cash,shares,avg,_,_=execute_target_var(cash,shares,avg,t[i],target,COST)
        elif future and reason=="BULL_EXIT" and shares>0:
            cash,shares,avg,_,_=execute_target_var(cash,shares,avg,t[i],0.0,COST)

        vals.append(cash+shares*t[i])

    s=pd.Series(vals,index=dates[start_idx:end_idx+1]); m=perf(s)
    return m,init_date,add_date,add_dd


def main():
    p=build_live_panel(); dates=pd.DatetimeIndex(p.date)
    _,_,tr,path=simulate_spec(p,RunSpec(BASE),save_path=True)
    fulls=[pd.Timestamp(x) for x in tr.loc[tr.reason.eq("BULL_FULL"),"date"]]
    exits=[pd.Timestamp(x) for x in tr.loc[tr.reason.eq("BULL_EXIT"),"date"]]
    fidx={d:dates.get_loc(d) for d in fulls}

    cur=None; ci=None; starts=[]
    for i,d in enumerate(dates):
        if d in fidx: cur=d; ci=i
        if d in exits: cur=None; ci=None
        if cur is not None and str(path.iloc[i].state)=="BULL":
            age=i-ci
            if age>=505: starts.append((i,d,cur,age))
    st=pd.DataFrame(starts,columns=["idx","date","full_date","age_td"]); st["month"]=st.date.dt.to_period("M"); st=st.groupby("month",as_index=False).first()

    raw=[]
    for _,r in st.iterrows():
        si=int(r.idx); sd=pd.Timestamp(r.date); age=int(r.age_td)
        for yrs in (3,5):
            target=sd+pd.DateOffset(years=yrs); valid=np.flatnonzero(dates<=target)
            if not len(valid):continue
            ei=int(valid[-1])
            if (dates[ei]-sd).days<int(365.2425*yrs)-10:continue
            for name,iw,at,pb,ref in POLICIES:
                m,idt,adt,adddd=run_policy(p,tr,si,ei,iw,at,pb,ref)
                raw.append({"start_date":sd,"full_date":pd.Timestamp(r.full_date),"age_td":age,"age_bucket":bucket(age),"horizon_y":yrs,"policy":name,
                            "CAGR":m["CAGR"],"MDD":m["MDD"],"Final":m["Final"],"initial_entry_date":idt,"add_date":adt,"add_qqq_dd":adddd})
    rdf=pd.DataFrame(raw); rdf.to_csv(OUT/"v22_raw.csv",index=False)
    wait=rdf[rdf.policy.eq("WAIT_NEXT")][["start_date","horizon_y","Final","CAGR"]].rename(columns={"Final":"WaitFinal","CAGR":"WaitCAGR"})
    c=rdf.merge(wait,on=["start_date","horizon_y"],how="left"); c["WinWait"]=c.Final>c.WaitFinal; c["DiffCAGR"]=c.CAGR-c.WaitCAGR
    summ=[]
    for (h,b,pol),z in c.groupby(["horizon_y","age_bucket","policy"]):
        summ.append({"horizon_y":h,"age_bucket":b,"policy":pol,"n":len(z),"campaigns":z.full_date.nunique(),
                     "CAGR_Median":z.CAGR.median(),"CAGR_P10":z.CAGR.quantile(.1),"MDD_Median":z.MDD.median(),"MDD_Worst":z.MDD.min(),
                     "WinRate_vs_Wait":z.WinWait.mean(),"DiffCAGR_Median":z.DiffCAGR.median(),"DiffCAGR_P10":z.DiffCAGR.quantile(.1),
                     "AddRate":z.add_date.notna().mean()})
    sdf=pd.DataFrame(summ); sdf.to_csv(OUT/"v22_summary.csv",index=False)

    latest_i=len(p)-1; latest=dates[-1]; current_full=max([d for d in fulls if d<=latest]); age=latest_i-dates.get_loc(current_full)
    snap=[]
    for fd in fulls:
        fi=dates.get_loc(fd); si=fi+age; later=[x for x in exits if x>fd]
        if not later:continue
        ex=later[0]; ei=dates.get_loc(ex)
        if si>=ei:continue
        for name,iw,at,pb,ref in POLICIES:
            m,idt,adt,adddd=run_policy(p,tr,si,ei,iw,at,pb,ref)
            snap.append({"campaign_full_date":fd,"matched_start_date":dates[si],"exit_date":ex,"age_td":age,"policy":name,
                         "Return_to_exit":m["Final"]-1,"CAGR_to_exit":m["CAGR"],"MDD_to_exit":m["MDD"],"initial_entry_date":idt,"add_date":adt,"add_qqq_dd":adddd})
    snap=pd.DataFrame(snap); snap.to_csv(OUT/"v22_exact_current_age.csv",index=False)

    # What would each ladder do TODAY? No execution; just state based on current observable QQQ drawdown.
    q=p.qqq.to_numpy(float); r63=pd.Series(q).rolling(63,min_periods=1).max().to_numpy(float); current_r63_dd=q[-1]/r63[-1]-1
    today=[]
    for name,iw,at,pb,ref in POLICIES:
        today.append({"policy":name,"initial_weight_next_close":iw,"add_target":at,"pullback_threshold":pb,"reference":ref,
                      "current_qqq_dd_from_63d_high":current_r63_dd,"r63_add_condition_already_true":bool(ref=="r63" and current_r63_dd<=pb) if pb is not None else False})
    pd.DataFrame(today).to_csv(OUT/"v22_current_policy_state.csv",index=False)

    print(f"CURRENT latest={latest.date()} full={current_full.date()} age={age} qqq_dd63={current_r63_dd:.4f}")
    print("\n=== CURRENT-AGE BUCKET ===")
    print(sdf[sdf.age_bucket.eq(bucket(age))].to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\n=== EXACT CURRENT AGE ===")
    print(snap.to_string(index=False,float_format=lambda x:f"{x:.4f}") if len(snap) else "none")

if __name__=="__main__":main()
