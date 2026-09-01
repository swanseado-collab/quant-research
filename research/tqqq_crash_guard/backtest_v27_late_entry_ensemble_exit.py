from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from backtest_v1 import perf
from backtest_v13_validation import execute_target_var, RunSpec, simulate_spec
from backtest_v6_bull_hold import Cfg
from backtest_v23_ndx_history_extension import build_extended_panel, campaign_table

OUT=Path(__file__).resolve().parent/"results_v27_late_entry_ensemble"
OUT.mkdir(parents=True,exist_ok=True)
BASE=Cfg(-0.25,80,110,0.35,110,50,-0.18)
COST=0.0005
MAS=(100,110,120)
SLOPES=(20,30,50)

# name, initial weight, add target, pullback, ensemble family, votes required
POLICIES=[
    ("I25_N3_K2",0.25,None,None,"N3",2),
    ("I50_N3_K2",0.50,None,None,"N3",2),
    ("L25_50_PB5_N3_K2",0.25,0.50,-0.05,"N3",2),
    ("I25_W9_K3",0.25,None,None,"W9",3),
    ("I25_W9_K5",0.25,None,None,"W9",5),
    ("I25_W9_K7",0.25,None,None,"W9",7),
    ("I50_W9_K3",0.50,None,None,"W9",3),
    ("I50_W9_K5",0.50,None,None,"W9",5),
    ("I50_W9_K7",0.50,None,None,"W9",7),
    ("L25_50_PB5_W9_K3",0.25,0.50,-0.05,"W9",3),
    ("L25_50_PB5_W9_K5",0.25,0.50,-0.05,"W9",5),
    ("L25_50_PB5_W9_K7",0.25,0.50,-0.05,"W9",7),
    ("WAIT_NEXT",0.0,None,None,"NONE",0),
]


def ma_map(p):
    q=p.qqq.astype(float)
    return {ma:q.rolling(ma,min_periods=ma).mean().to_numpy(float) for ma in MAS}


def votes(i,q,mm,family):
    rules=[]
    if family=="N3":
        for ma in MAS:
            m=mm[ma]; sl=30
            rules.append(i>=sl and np.isfinite(m[i]) and np.isfinite(m[i-sl]) and q[i]<m[i] and m[i]<m[i-sl])
    elif family=="W9":
        for ma in MAS:
            m=mm[ma]
            for sl in SLOPES:
                rules.append(i>=sl and np.isfinite(m[i]) and np.isfinite(m[i-sl]) and q[i]<m[i] and m[i]<m[i-sl])
    return int(sum(bool(x) for x in rules)),len(rules)


def run_policy(p,si,ei,base_exit,iw,at,pb,family,k,mm,save_path=False):
    dates=pd.DatetimeIndex(p.date); q=p.qqq.to_numpy(float); t=p.tqqq.to_numpy(float); cy=p.cash_yield_pct.to_numpy(float)
    cash=1.0; shares=0.0; avg=np.nan; pending=(si+1,iw,"INITIAL") if iw>0 else None
    running_high=q[si]; add_fired=False; early=False; rows=[]; ev=[]
    for i in range(si,ei+1):
        if i>si: cash*=1+max(0.0,cy[i-1]/100.0)/252.0
        if pending is not None and i>=pending[0]:
            _,target,reason=pending
            cash,shares,avg,notional,fee=execute_target_var(cash,shares,avg,t[i],target,COST)
            if abs(notional)>1e-12: ev.append((dates[i],reason,target,t[i],notional,fee))
            if target<=1e-12 and reason=="ENSEMBLE_EXIT": early=True
            pending=None
        if dates[i]==base_exit and shares>0:
            cash,shares,avg,notional,fee=execute_target_var(cash,shares,avg,t[i],0.0,COST)
            if abs(notional)>1e-12: ev.append((dates[i],"BASE_EXIT",0.0,t[i],notional,fee))
        eq=cash+shares*t[i]; v,n=votes(i,q,mm,family) if family!="NONE" else (0,0)
        rows.append((dates[i],eq,shares*t[i]/eq if eq>0 else 0.0,v,n,q[i]))
        running_high=max(running_high,q[i])
        if i>=ei or shares<=0 or early or pending is not None: continue
        # signal at close -> t+1 close
        if family!="NONE" and v>=k:
            pending=(min(i+1,ei),0.0,"ENSEMBLE_EXIT"); continue
        if at is not None and not add_fired and q[i]/running_high-1.0<=pb:
            pending=(min(i+1,ei),at,"ADD_PB"); add_fired=True
    path=pd.DataFrame(rows,columns=["date","equity","weight","votes","n_rules","qqq"])
    m=perf(pd.Series(path.equity.to_numpy(),index=pd.DatetimeIndex(path.date)))
    return m,path,pd.DataFrame(ev,columns=["date","reason","target","price","notional","fee"])


def wait_perf(p,si,ei):
    dates=pd.DatetimeIndex(p.date); cy=p.cash_yield_pct.to_numpy(float); c=1.0; vals=[]
    for i in range(si,ei+1):
        if i>si:c*=1+max(0.0,cy[i-1]/100.0)/252.0
        vals.append(c)
    return perf(pd.Series(vals,index=dates[si:ei+1]))


def main():
    p,_,_=build_extended_panel(); dates=pd.DatetimeIndex(p.date); mm=ma_map(p)
    _,_,tr,_=simulate_spec(p,RunSpec(BASE)); camps=campaign_table(p,tr)
    raw=[]
    for _,c in camps.iterrows():
        if bool(c.ongoing):continue
        fd=pd.Timestamp(c.full_date); ex=pd.Timestamp(c.exit_or_latest); fi=dates.get_loc(fd); ei=dates.get_loc(ex)
        for age in (505,756,879,1009):
            si=fi+age
            if si>=ei:continue
            wm=wait_perf(p,si,ei)
            for name,iw,at,pb,fam,k in POLICIES:
                if name=="WAIT_NEXT":m=wm
                else:m,_,_=run_policy(p,si,ei,ex,iw,at,pb,fam,k,mm)
                raw.append({"campaign_full_date":fd,"age_td":age,"policy":name,"family":fam,"votes_required":k,
                            "CAGR":m["CAGR"],"MDD":m["MDD"],"Return":m["Final"]-1.0,"WinWait":m["Final"]>wm["Final"],"DiffCAGR":m["CAGR"]-wm["CAGR"]})
    rdf=pd.DataFrame(raw); rdf.to_csv(OUT/"v27_raw.csv",index=False)
    summ=[]
    for (age,pol),z in rdf.groupby(["age_td","policy"]):
        summ.append({"age_td":age,"policy":pol,"campaigns":z.campaign_full_date.nunique(),"CAGR_Median":z.CAGR.median(),"CAGR_Min":z.CAGR.min(),
                     "MDD_Median":z.MDD.median(),"MDD_Worst":z.MDD.min(),"WinRate":z.WinWait.mean(),"DiffCAGR_Median":z.DiffCAGR.median(),"DiffCAGR_Min":z.DiffCAGR.min()})
    sdf=pd.DataFrame(summ); sdf.to_csv(OUT/"v27_summary.csv",index=False)
    overall=[]
    for pol,z in rdf.groupby("policy"):
        overall.append({"policy":pol,"n":len(z),"campaigns":z.campaign_full_date.nunique(),"CAGR_Median":z.CAGR.median(),"CAGR_Min":z.CAGR.min(),
                        "MDD_Median":z.MDD.median(),"MDD_Worst":z.MDD.min(),"WinRate":z.WinWait.mean(),"DiffCAGR_Median":z.DiffCAGR.median(),"DiffCAGR_Min":z.DiffCAGR.min()})
    odf=pd.DataFrame(overall); odf.to_csv(OUT/"v27_overall.csv",index=False)

    # predeclared robust set for current age: positive min CAGR, <=35% worst MDD; across all ages min CAGR >= -10%, <=40% worst MDD
    cur=sdf[sdf.age_td.eq(879)].copy(); merged=cur.merge(odf,on="policy",suffixes=("_cur","_all"))
    good=merged[(merged.CAGR_Min_cur>=0)&(merged.MDD_Worst_cur>=-0.35)&(merged.CAGR_Min_all>=-0.10)&(merged.MDD_Worst_all>=-0.40)].sort_values(["CAGR_Median_cur","CAGR_Median_all"],ascending=False)
    good.to_csv(OUT/"v27_robust_shortlist.csv",index=False)

    # Current vote state.
    q=p.qqq.to_numpy(float); i=len(p)-1; states=[]
    for fam in ("N3","W9"):
        v,n=votes(i,q,mm,fam); states.append({"latest_date":dates[-1],"family":fam,"votes":v,"n_rules":n,"vote_fraction":v/n})
    pd.DataFrame(states).to_csv(OUT/"v27_current_vote_state.csv",index=False)

    print("=== V27 CURRENT AGE ===")
    print(cur.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\n=== V27 OVERALL ===")
    print(odf.sort_values("CAGR_Median",ascending=False).to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\n=== V27 ROBUST ===")
    print(good.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\n=== CURRENT VOTES ===")
    print(pd.DataFrame(states).to_string(index=False,float_format=lambda x:f"{x:.4f}"))

if __name__=="__main__":main()
