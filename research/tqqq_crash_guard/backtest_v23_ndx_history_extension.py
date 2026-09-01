from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

from backtest_v1 import perf
from backtest_v2 import add_features
from backtest_v6_bull_hold import Cfg
from backtest_v13_validation import RunSpec, ensure_features, simulate_spec
from backtest_v15_cycle_and_synth_stress import build_live_panel
from backtest_v22_late_join_ladder import POLICIES, run_policy

OUT = Path(__file__).resolve().parent / "results_v23_ndx_extension"
OUT.mkdir(parents=True, exist_ok=True)
BASE = Cfg(-0.25,80,110,0.35,110,50,-0.18)


def yf_close(ticker: str, start: str, end: str | None = None) -> pd.Series:
    x = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False, actions=False)
    if x is None or len(x) == 0:
        raise RuntimeError(f"No Yahoo data for {ticker}")
    if isinstance(x.columns, pd.MultiIndex):
        if ("Adj Close", ticker) in x.columns:
            s = x[("Adj Close", ticker)]
        elif ("Close", ticker) in x.columns:
            s = x[("Close", ticker)]
        else:
            s = x.xs("Close", axis=1, level=0).iloc[:,0]
    else:
        s = x["Adj Close"] if "Adj Close" in x.columns else x["Close"]
    s = pd.Series(s, dtype=float).dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def build_extended_panel(leverage=3.0, annual_drag=0.0135):
    live = build_live_panel().copy()
    live["date"] = pd.to_datetime(live.date)
    first = pd.Timestamp(live.date.min())
    anchor_q = float(live.loc[live.date.eq(first), "qqq"].iloc[0])
    anchor_t = float(live.loc[live.date.eq(first), "tqqq"].iloc[0])

    ndx = yf_close("^NDX", "1985-01-01", (first + pd.Timedelta(days=3)).strftime("%Y-%m-%d"))
    irx = yf_close("^IRX", "1985-01-01", (first + pd.Timedelta(days=3)).strftime("%Y-%m-%d"))
    if ndx.index.min() > pd.Timestamp("1986-01-01"):
        raise RuntimeError(f"NDX history unexpectedly short: {ndx.index.min()}")

    # Scale the index level to join continuously to QQQ. Returns are invariant to this scale.
    anchor_candidates = ndx[ndx.index <= first]
    if len(anchor_candidates) == 0:
        raise RuntimeError("No NDX observation at/before QQQ panel start")
    ndx_anchor_date = anchor_candidates.index[-1]
    scale = anchor_q / float(anchor_candidates.iloc[-1])
    q_pre = (ndx * scale).loc[ndx.index < first]

    y = irx.reindex(q_pre.index).ffill().bfill()
    # ^IRX is quoted as a percent yield. Cap only pathological data errors.
    y = y.clip(lower=0.0, upper=25.0)

    # Backcast a hypothetical daily-reset 3x product from the existing 1999 synthetic anchor.
    all_idx = q_pre.index.append(pd.DatetimeIndex([first]))
    q_anchor_series = pd.concat([q_pre, pd.Series([anchor_q], index=[first])])
    y_anchor = pd.concat([y, pd.Series([float(y.iloc[-1])], index=[first])])
    qr = q_anchor_series.pct_change().to_numpy(float)
    mr = np.zeros(len(all_idx), float)
    for i in range(1, len(all_idx)):
        financing = max(0.0, float(y_anchor.iloc[i-1])/100.0) * max(0.0, leverage-1.0) / 252.0
        mr[i] = max(leverage*qr[i] - financing - annual_drag/252.0, -0.99)
    t = np.empty(len(all_idx), float); t[-1] = anchor_t
    for i in range(len(all_idx)-1, 0, -1):
        t[i-1] = t[i] / (1.0 + mr[i])
    t_pre = pd.Series(t[:-1], index=q_pre.index)

    pre = pd.DataFrame({
        "date": q_pre.index,
        "qqq": q_pre.to_numpy(float),
        "tqqq": t_pre.to_numpy(float),
        "price_source": f"NDX_synth_L{leverage:.1f}_D{annual_drag:.4f}",
        "cash_yield_pct": y.to_numpy(float),
    })
    cols=["date","qqq","tqqq","price_source","cash_yield_pct"]
    p = pd.concat([pre[cols], live[cols]], ignore_index=True).sort_values("date").drop_duplicates("date",keep="last").reset_index(drop=True)
    return ensure_features(add_features(p)), ndx_anchor_date, scale


def campaign_table(p, tr):
    dates=pd.DatetimeIndex(p.date)
    fulls=[pd.Timestamp(x) for x in tr.loc[tr.reason.eq("BULL_FULL"),"date"]]
    exits=[pd.Timestamp(x) for x in tr.loc[tr.reason.eq("BULL_EXIT"),"date"]]
    rows=[]
    for fd in fulls:
        later=[x for x in exits if x>fd]
        ex=later[0] if later else pd.Timestamp(dates[-1])
        fi=dates.get_loc(fd); ei=dates.get_loc(ex)
        rows.append({"full_date":fd,"exit_or_latest":ex,"ongoing":len(later)==0,"bull_trading_days":ei-fi,"calendar_days":(ex-fd).days})
    return pd.DataFrame(rows)


def evaluate_exact_ages(p,tr,camps):
    dates=pd.DatetimeIndex(p.date)
    rows=[]
    for _,c in camps.iterrows():
        if bool(c.ongoing):
            continue
        fd=pd.Timestamp(c.full_date); ex=pd.Timestamp(c.exit_or_latest)
        fi=dates.get_loc(fd); ei=dates.get_loc(ex)
        for age in (505,756,879,1009):
            si=fi+age
            if si>=ei: continue
            # Evaluate to campaign exit, which is independent of arbitrary horizon censoring.
            for name,iw,at,pb,ref in POLICIES:
                m,idt,adt,adddd=run_policy(p,tr,si,ei,iw,at,pb,ref)
                rows.append({
                    "campaign_full_date":fd,"matched_start_date":dates[si],"exit_date":ex,
                    "age_td":age,"policy":name,"remaining_days":(ex-dates[si]).days,
                    "Return_to_exit":m["Final"]-1.0,"CAGR_to_exit":m["CAGR"],"MDD_to_exit":m["MDD"],
                    "initial_entry_date":idt,"add_date":adt,"add_qqq_dd":adddd,
                })
    return pd.DataFrame(rows)


def summarize_independent(raw):
    if len(raw)==0:return pd.DataFrame()
    wait=raw[raw.policy.eq("WAIT_NEXT")][["campaign_full_date","age_td","Return_to_exit","CAGR_to_exit"]].rename(columns={"Return_to_exit":"WaitReturn","CAGR_to_exit":"WaitCAGR"})
    x=raw.merge(wait,on=["campaign_full_date","age_td"],how="left")
    x["WinWait"]=x.Return_to_exit>x.WaitReturn
    x["DiffCAGR"]=x.CAGR_to_exit-x.WaitCAGR
    rows=[]
    for (age,pol),z in x.groupby(["age_td","policy"]):
        rows.append({
            "age_td":age,"policy":pol,"independent_campaigns":z.campaign_full_date.nunique(),
            "CAGR_Median":z.CAGR_to_exit.median(),"CAGR_Min":z.CAGR_to_exit.min(),
            "Return_Median":z.Return_to_exit.median(),"Return_Min":z.Return_to_exit.min(),
            "MDD_Median":z.MDD_to_exit.median(),"MDD_Worst":z.MDD_to_exit.min(),
            "WinRate_vs_Wait":z.WinWait.mean(),"DiffCAGR_Median":z.DiffCAGR.median(),"DiffCAGR_Min":z.DiffCAGR.min(),
            "AddRate":z.add_date.notna().mean(),
        })
    return pd.DataFrame(rows)


def main():
    p,anchor_date,scale=build_extended_panel()
    eq,w,tr,path=simulate_spec(p,RunSpec(BASE),save_path=True)
    camps=campaign_table(p,tr)
    raw=evaluate_exact_ages(p,tr,camps)
    summ=summarize_independent(raw)

    p[["date","qqq","tqqq","price_source","cash_yield_pct"]].to_csv(OUT/"v23_extended_panel_prices.csv",index=False)
    camps.to_csv(OUT/"v23_campaigns.csv",index=False)
    raw.to_csv(OUT/"v23_exact_age_raw.csv",index=False)
    summ.to_csv(OUT/"v23_independent_campaign_summary.csv",index=False)
    tr.to_csv(OUT/"v23_v7_trades.csv",index=False)

    meta=pd.DataFrame([{
        "start":p.date.min(),"end":p.date.max(),"rows":len(p),"ndx_anchor_date":anchor_date,"ndx_to_qqq_scale":scale,
        "completed_campaigns":int((~camps.ongoing).sum()),"long_505":int(((~camps.ongoing)&(camps.bull_trading_days>505)).sum()),
        "long_756":int(((~camps.ongoing)&(camps.bull_trading_days>756)).sum()),"long_879":int(((~camps.ongoing)&(camps.bull_trading_days>879)).sum()),
        **{f"V7_{k}":v for k,v in perf(eq).items()}
    }])
    meta.to_csv(OUT/"v23_meta.csv",index=False)

    # Robust shortlist: at each age, prefer non-negative minimum CAGR, worst MDD above -45%, then highest median CAGR.
    shortlist=[]
    for age,z in summ.groupby("age_td"):
        zz=z[(z.CAGR_Min>=0)&(z.MDD_Worst>=-0.45)].copy()
        if len(zz):
            a=zz.sort_values(["CAGR_Median","CAGR_Min"],ascending=False).iloc[0].copy(); a["rule"]="minCAGR>=0_and_worstMDD>=-45%"; shortlist.append(a)
        # Also record the highest median regardless of risk as an upside benchmark.
        a=z.sort_values("CAGR_Median",ascending=False).iloc[0].copy(); a["rule"]="max_median_CAGR_unconstrained"; shortlist.append(a)
    pd.DataFrame(shortlist).to_csv(OUT/"v23_shortlist.csv",index=False)

    print("=== V23 META ===")
    print(meta.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\n=== CAMPAIGNS ===")
    print(camps.to_string(index=False))
    print("\n=== INDEPENDENT CAMPAIGN SUMMARY ===")
    print(summ.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\n=== SHORTLIST ===")
    print(pd.DataFrame(shortlist).to_string(index=False,float_format=lambda x:f"{x:.4f}"))

if __name__=="__main__":
    main()
