from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/"results_v26_late_entry_walkforward"
OUT.mkdir(parents=True,exist_ok=True)
RAW=HERE/"results_v25_late_entry_robustness"/"v25_raw.csv"
FIXED="L25_50_PB5_XMA110_S30"
BENCH_FORMS=("I25","I50")


def aggregate(z):
    return pd.DataFrame([
        {"policy":p,
         "n":len(g),"campaigns":g.campaign_full_date.nunique(),
         "CAGR_Median":g.CAGR.median(),"CAGR_Min":g.CAGR.min(),
         "MDD_Worst":g.MDD.min(),"MDD_Median":g.MDD.median(),
         "DiffCAGR_Median":g.DiffCAGR.median(),"DiffCAGR_Min":g.DiffCAGR.min(),
         "WinRate":g.WinWait.mean(),
         "form":g.form.iloc[0],"exit_ma":int(g.exit_ma.iloc[0]),"slope_lb":int(g.slope_lb.iloc[0])}
        for p,g in z.groupby("policy")
    ])


def choose(train):
    a=aggregate(train)
    # Same broad risk discipline as V25, but only using information available in training campaigns.
    q=a[(a.MDD_Worst>=-0.40)&(a.CAGR_Min>=-0.10)].copy()
    if not len(q):
        q=a.copy()
    # Avoid selecting solely on median upside: minimum CAGR and worst MDD act as tie-breaks.
    q=q.sort_values(["CAGR_Median","CAGR_Min","MDD_Worst"],ascending=[False,False,False])
    return q.iloc[0],a


def main():
    raw=pd.read_csv(RAW,parse_dates=["campaign_full_date"])
    campaigns=sorted(raw.campaign_full_date.unique())
    rows=[]; selections=[]

    for test_date in campaigns:
        train=raw[raw.campaign_full_date<test_date].copy()
        test=raw[raw.campaign_full_date.eq(test_date)].copy()
        if train.campaign_full_date.nunique()<1:
            continue
        sel,agg=choose(train)
        policy=sel.policy
        selections.append({"test_campaign":test_date,"train_campaigns":train.campaign_full_date.nunique(),"train_obs":len(train),
                           "selected_policy":policy,"selected_form":sel.form,"selected_exit_ma":sel.exit_ma,"selected_slope_lb":sel.slope_lb,
                           "train_CAGR_Median":sel.CAGR_Median,"train_CAGR_Min":sel.CAGR_Min,"train_MDD_Worst":sel.MDD_Worst})
        for _,r in test[test.policy.eq(policy)].iterrows():
            rows.append({"test_campaign":test_date,"method":"WF_SELECTED","policy":policy,"age_td":r.age_td,"CAGR":r.CAGR,"MDD":r.MDD,"Return":r.Return,"WinWait":r.WinWait,"DiffCAGR":r.DiffCAGR})
        for _,r in test[test.policy.eq(FIXED)].iterrows():
            rows.append({"test_campaign":test_date,"method":"FIXED_V25","policy":FIXED,"age_td":r.age_td,"CAGR":r.CAGR,"MDD":r.MDD,"Return":r.Return,"WinWait":r.WinWait,"DiffCAGR":r.DiffCAGR})
        # Best simple no-ladder I25/I50 under the training sample, selected independently of the grid winner.
        simple=agg[agg.form.isin(BENCH_FORMS)].sort_values(["CAGR_Median","CAGR_Min","MDD_Worst"],ascending=[False,False,False]).iloc[0]
        for _,r in test[test.policy.eq(simple.policy)].iterrows():
            rows.append({"test_campaign":test_date,"method":"SIMPLE_SELECTED","policy":simple.policy,"age_td":r.age_td,"CAGR":r.CAGR,"MDD":r.MDD,"Return":r.Return,"WinWait":r.WinWait,"DiffCAGR":r.DiffCAGR})

    res=pd.DataFrame(rows); sel=pd.DataFrame(selections)
    res.to_csv(OUT/"v26_walkforward_tests.csv",index=False); sel.to_csv(OUT/"v26_selections.csv",index=False)

    # Summary by method across truly later campaigns; each campaign-age observation receives equal weight.
    summary=[]
    for method,z in res.groupby("method"):
        summary.append({"method":method,"n":len(z),"test_campaigns":z.test_campaign.nunique(),"CAGR_Median":z.CAGR.median(),"CAGR_Min":z.CAGR.min(),
                        "MDD_Median":z.MDD.median(),"MDD_Worst":z.MDD.min(),"WinRate_vs_Wait":z.WinWait.mean(),
                        "DiffCAGR_Median":z.DiffCAGR.median(),"DiffCAGR_Min":z.DiffCAGR.min()})
    sdf=pd.DataFrame(summary).sort_values("CAGR_Median",ascending=False); sdf.to_csv(OUT/"v26_summary.csv",index=False)

    # What would a policy selector trained only on all COMPLETED campaigns choose now?
    current_sel,allagg=choose(raw)
    near=allagg[(allagg.exit_ma.between(100,120))&(allagg.slope_lb.isin([20,30,50]))].sort_values(["CAGR_Median","CAGR_Min"],ascending=False).head(20)
    current=pd.DataFrame([{"selected_policy_now":current_sel.policy,"form":current_sel.form,"exit_ma":current_sel.exit_ma,"slope_lb":current_sel.slope_lb,
                           "train_campaigns":raw.campaign_full_date.nunique(),"train_obs":len(raw),"CAGR_Median":current_sel.CAGR_Median,
                           "CAGR_Min":current_sel.CAGR_Min,"MDD_Worst":current_sel.MDD_Worst,"fixed_candidate":FIXED}])
    current.to_csv(OUT/"v26_current_training_selection.csv",index=False); near.to_csv(OUT/"v26_nearby_completed_history_rank.csv",index=False)

    print("=== WALK-FORWARD SELECTIONS ===")
    print(sel.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\n=== WALK-FORWARD TESTS ===")
    print(res.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\n=== SUMMARY ===")
    print(sdf.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\n=== CURRENT COMPLETED-HISTORY SELECTION ===")
    print(current.to_string(index=False,float_format=lambda x:f"{x:.4f}"))

if __name__=="__main__":main()
