#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

OUT = Path("results/eth_entry_grid")
OUT.mkdir(parents=True, exist_ok=True)

SYMBOL = "ETHUSDT"
INTERVAL = "1d"
START = "2017-08-17"
CAPITAL = 10_000.0
FEE = 0.0005
HORIZON_DAYS = 730

INITIAL_PCTS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
DCA_MONTHS = [1, 3, 6, 9, 12]
DIP_SETS = {
    "none": [],
    "10_20_30_40": [-0.10, -0.20, -0.30, -0.40],
    "15_25_35_45": [-0.15, -0.25, -0.35, -0.45],
    "10_15_20_30": [-0.10, -0.15, -0.20, -0.30],
}


def fetch_binance_daily() -> pd.DataFrame:
    bases = [
        "https://data-api.binance.vision/api/v3/klines",
        "https://api.binance.com/api/v3/klines",
    ]
    start_ms = int(pd.Timestamp(START, tz="UTC").timestamp() * 1000)
    last_err = None
    for base in bases:
        try:
            rows = []
            cur = start_ms
            while True:
                r = requests.get(
                    base,
                    params={"symbol": SYMBOL, "interval": INTERVAL, "startTime": cur, "limit": 1000},
                    timeout=30,
                    headers={"User-Agent": "quant-research/eth-entry-grid"},
                )
                r.raise_for_status()
                chunk = r.json()
                if not isinstance(chunk, list) or not chunk:
                    break
                rows.extend(chunk)
                nxt = int(chunk[-1][0]) + 86_400_000
                if nxt <= cur or len(chunk) < 1000:
                    break
                cur = nxt
                time.sleep(0.05)
            if len(rows) >= 2500:
                d = pd.DataFrame(rows, columns=[
                    "open_time","open","high","low","close","volume","close_time","quote_volume",
                    "trades","taker_base","taker_quote","ignore"
                ])
                d["date"] = pd.to_datetime(d.open_time, unit="ms", utc=True).dt.tz_localize(None)
                for c in ["open","high","low","close","volume"]:
                    d[c] = pd.to_numeric(d[c], errors="coerce")
                return d[["date","open","high","low","close","volume"]].dropna().drop_duplicates("date").sort_values("date").reset_index(drop=True)
        except Exception as e:
            last_err = repr(e)
    raise RuntimeError(f"Binance daily fetch failed: {last_err}")


def validate(d: pd.DataFrame) -> dict:
    bad = ((d.high < d[["open","close"]].max(axis=1)) |
           (d.low > d[["open","close"]].min(axis=1)) |
           (d.high < d.low))
    gaps = d.date.diff().dt.days
    info = {
        "rows": int(len(d)),
        "start": str(d.date.min().date()),
        "end": str(d.date.max().date()),
        "duplicate_dates": int(d.date.duplicated().sum()),
        "bad_ohlc": int(bad.sum()),
        "gaps_gt_1d": int((gaps > 1).sum()),
        "max_gap_days": int(gaps.max()) if len(gaps.dropna()) else 0,
    }
    if info["duplicate_dates"] or info["bad_ohlc"]:
        raise RuntimeError(f"Invalid OHLCV: {info}")
    return info


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    x["peak365"] = x.close.rolling(365, min_periods=365).max()
    x["dd365"] = x.close / x.peak365 - 1.0
    x["mom30"] = x.close.pct_change(30)
    x["mom7"] = x.close.pct_change(7)
    x["ma200"] = x.close.rolling(200, min_periods=200).mean()
    return x


def buy(cash: float, qty: float, amount_total: float, px: float):
    amount_total = max(0.0, min(amount_total, cash))
    if amount_total <= 1e-12:
        return cash, qty, 0.0, 0.0
    principal = amount_total / (1.0 + FEE)
    fee = amount_total - principal
    q = principal / px
    return cash - amount_total, qty + q, principal, fee


def max_drawdown(a: np.ndarray) -> float:
    p = np.maximum.accumulate(a)
    return float(np.min(a / p - 1.0))


def month_date_index(dates: pd.Series, target: pd.Timestamp, end_idx: int) -> int | None:
    arr = dates.values
    i = int(np.searchsorted(arr, np.datetime64(target), side="left"))
    return i if i <= end_idx and i < len(arr) else None


def simulate_grid_one(d: pd.DataFrame, start_idx: int, initial_pct: float, dca_months: int,
                      dip_name: str, dip_levels: list[float]) -> dict | None:
    start_date = pd.Timestamp(d.iloc[start_idx].date)
    target_end = start_date + pd.Timedelta(days=HORIZON_DAYS)
    end_idx = int(np.searchsorted(d.date.values, np.datetime64(target_end), side="right") - 1)
    if end_idx <= start_idx or end_idx >= len(d):
        return None
    if pd.Timestamp(d.iloc[end_idx].date) < target_end - pd.Timedelta(days=2):
        return None

    cash, qty = CAPITAL, 0.0
    total_principal = 0.0
    total_qty_bought = 0.0
    fees = 0.0
    buys = 0

    entry_px = float(d.iloc[start_idx].open)
    init_total = CAPITAL * initial_pct
    cash, qty, pr, fe = buy(cash, qty, init_total, entry_px)
    total_principal += pr; total_qty_bought += pr / entry_px; fees += fe; buys += int(pr > 0)

    remaining_total = CAPITAL - init_total
    tranche_total = remaining_total / dca_months if dca_months > 0 else 0.0
    consumed = [False] * dca_months
    schedule = []
    for k in range(1, dca_months + 1):
        idx = month_date_index(d.date, start_date + pd.DateOffset(months=k), end_idx)
        schedule.append(idx)
    dip_used = [False] * len(dip_levels)

    equity = []
    full_deploy_idx = start_idx if cash <= 1e-8 else None

    for i in range(start_idx, end_idx + 1):
        row = d.iloc[i]
        # Scheduled DCA executes at the day's OPEN. Each scheduled tranche can be
        # consumed earlier by a dip trigger, in which case this date is skipped.
        for j, si in enumerate(schedule):
            if si == i and not consumed[j] and cash > 1e-8:
                amt = min(tranche_total, cash)
                cash, qty, pr, fe = buy(cash, qty, amt, float(row.open))
                consumed[j] = True
                total_principal += pr; total_qty_bought += pr / float(row.open); fees += fe; buys += int(pr > 0)

        # Intraday limit buys accelerate one still-future DCA tranche. Multiple
        # levels may fill on a crash day; total capital is unchanged.
        for k, lev in enumerate(dip_levels):
            if dip_used[k] or cash <= 1e-8:
                continue
            target_px = entry_px * (1.0 + lev)
            if float(row.low) <= target_px:
                candidates = [j for j, used in enumerate(consumed) if not used and schedule[j] is not None and schedule[j] >= i]
                if not candidates:
                    continue
                j = candidates[-1]
                amt = min(tranche_total, cash)
                cash, qty, pr, fe = buy(cash, qty, amt, target_px)
                consumed[j] = True
                dip_used[k] = True
                total_principal += pr; total_qty_bought += pr / target_px; fees += fe; buys += int(pr > 0)

        if full_deploy_idx is None and cash <= 1e-8:
            full_deploy_idx = i
        equity.append(cash + qty * float(row.close))

    final_eq = equity[-1]
    avg_buy = total_principal / total_qty_bought if total_qty_bought > 0 else np.nan
    deploy_days = ((pd.Timestamp(d.iloc[full_deploy_idx].date) - start_date).days
                   if full_deploy_idx is not None else np.nan)
    return {
        "strategy": f"P{int(initial_pct*100)}_DCA{dca_months}_{dip_name}",
        "initial_pct": initial_pct,
        "dca_months": dca_months,
        "dip_set": dip_name,
        "start": start_date,
        "end": pd.Timestamp(d.iloc[end_idx].date),
        "entry_open": entry_px,
        "final_equity": final_eq,
        "return": final_eq / CAPITAL - 1.0,
        "mdd": max_drawdown(np.asarray(equity)),
        "avg_buy_price": avg_buy,
        "cash_end": cash,
        "buys": buys,
        "deploy_days": deploy_days,
        "dip_fills": int(sum(dip_used)),
    }


def simulate_pure_dca(d: pd.DataFrame, start_idx: int, months: int) -> dict | None:
    start_date = pd.Timestamp(d.iloc[start_idx].date)
    target_end = start_date + pd.Timedelta(days=HORIZON_DAYS)
    end_idx = int(np.searchsorted(d.date.values, np.datetime64(target_end), side="right") - 1)
    if end_idx <= start_idx or end_idx >= len(d): return None
    if pd.Timestamp(d.iloc[end_idx].date) < target_end - pd.Timedelta(days=2): return None
    cash, qty = CAPITAL, 0.0
    tranche = CAPITAL / months
    buy_idxs = []
    for k in range(months):
        idx = month_date_index(d.date, start_date + pd.DateOffset(months=k), end_idx)
        if idx is not None: buy_idxs.append(idx)
    equity=[]; principal=0.; qtot=0.; fees=0.; buys=0
    for i in range(start_idx, end_idx+1):
        if i in buy_idxs and cash > 1e-8:
            px=float(d.iloc[i].open); amt=min(tranche,cash)
            cash,qty,pr,fe=buy(cash,qty,amt,px)
            principal+=pr; qtot+=pr/px; fees+=fe; buys+=1
        equity.append(cash+qty*float(d.iloc[i].close))
    return {
        "strategy": f"BASE_DCA{months}", "initial_pct": 0.0, "dca_months": months, "dip_set":"baseline",
        "start":start_date,"end":pd.Timestamp(d.iloc[end_idx].date),"entry_open":float(d.iloc[start_idx].open),
        "final_equity":equity[-1],"return":equity[-1]/CAPITAL-1,"mdd":max_drawdown(np.asarray(equity)),
        "avg_buy_price":principal/qtot if qtot else np.nan,"cash_end":cash,"buys":buys,
        "deploy_days":(pd.Timestamp(d.iloc[buy_idxs[-1]].date)-start_date).days if buy_idxs else np.nan,"dip_fills":0,
    }


def simulate_lump(d: pd.DataFrame, start_idx: int) -> dict | None:
    return simulate_grid_one(d,start_idx,1.0,0,"lump",[])


def summarize(z: pd.DataFrame, label: str) -> pd.DataFrame:
    if z.empty: return pd.DataFrame()
    g = z.groupby("strategy", as_index=False).agg(
        cohorts=("return","size"), median_return=("return","median"), mean_return=("return","mean"),
        p25_return=("return",lambda x: x.quantile(.25)), p10_return=("return",lambda x:x.quantile(.10)),
        worst_return=("return","min"), median_mdd=("mdd","median"), worst_mdd=("mdd","min"),
        median_avg_buy=("avg_buy_price","median"), median_deploy_days=("deploy_days","median"),
        median_dip_fills=("dip_fills","median"),
    )
    # same-cohort benchmark comparisons
    bench = z[z.strategy=="BASE_LUMP"][["start","return"]].rename(columns={"return":"lump_ret"})
    zz=z.merge(bench,on="start",how="left")
    win=zz.groupby("strategy").apply(lambda q: float((q["return"]>q["lump_ret"]).mean()), include_groups=False).rename("win_vs_lump").reset_index()
    g=g.merge(win,on="strategy",how="left")
    g.insert(0,"segment",label)
    return g.sort_values(["median_return","p10_return"],ascending=False).reset_index(drop=True)


def main():
    d=fetch_binance_daily()
    info=validate(d)
    d=add_features(d)
    d.to_csv(OUT/"ethusdt_daily_used.csv",index=False)
    (OUT/"data_quality.json").write_text(json.dumps(info,indent=2),encoding="utf-8")

    # One cohort per month, decision at the first daily bar of each month. Regime
    # features use ONLY the previous day's close to avoid same-day look-ahead.
    month_first=d.groupby(d.date.dt.to_period("M"),as_index=False).head(1).index.tolist()
    starts=[i for i in month_first if i>=365 and (d.iloc[i].date + pd.Timedelta(days=HORIZON_DAYS) <= d.date.iloc[-1])]

    rows=[]
    for n,si in enumerate(starts,1):
        prev=d.iloc[si-1]
        meta={
            "signal_date":pd.Timestamp(prev.date),
            "dd365":float(prev.dd365),"mom30":float(prev.mom30),"mom7":float(prev.mom7),
            "price_vs_ma200":float(prev.close/prev.ma200-1.0) if pd.notna(prev.ma200) else np.nan,
        }
        b=simulate_lump(d,si)
        if b:
            b["strategy"]="BASE_LUMP"; b.update(meta); rows.append(b)
        for m in [3,6,12]:
            b=simulate_pure_dca(d,si,m)
            if b: b.update(meta); rows.append(b)
        for p in INITIAL_PCTS:
            for m in DCA_MONTHS:
                for dip_name,levels in DIP_SETS.items():
                    r=simulate_grid_one(d,si,p,m,dip_name,levels)
                    if r: r.update(meta); rows.append(r)
        if n%12==0: print(f"cohorts {n}/{len(starts)}",flush=True)

    cr=pd.DataFrame(rows)
    cr.to_csv(OUT/"cohort_results.csv",index=False)

    start_year=cr.start.dt.year
    segs={
        "ALL":cr,
        "TRAIN_2018_2021":cr[start_year<=2021],
        "VALID_2022_2023":cr[(start_year>=2022)&(start_year<=2023)],
        "OOS_2024_PLUS":cr[start_year>=2024],
        "DEEP_DD40":cr[cr.dd365<=-0.40],
        "CURRENTLIKE_LOOSE":cr[(cr.dd365<=-0.35)&(cr.mom30>=0.10)],
        "CURRENTLIKE_STRICT":cr[(cr.dd365<=-0.40)&(cr.mom30>=0.15)],
    }
    summaries=[]
    for name,z in segs.items():
        s=summarize(z,name)
        s.to_csv(OUT/f"summary_{name.lower()}.csv",index=False)
        summaries.append(s)
    allsum=pd.concat(summaries,ignore_index=True)

    # Robust shortlist selected without OOS: require at least 12 train and 12 validation cohorts,
    # rank by return and downside separately, then average rank. OOS is reported, not used to select.
    tr=allsum[allsum.segment=="TRAIN_2018_2021"].copy()
    va=allsum[allsum.segment=="VALID_2022_2023"].copy()
    cand=tr.merge(va,on="strategy",suffixes=("_tr","_va"))
    cand=cand[(cand.cohorts_tr>=12)&(cand.cohorts_va>=12)].copy()
    for col,asc in [("median_return_tr",False),("p10_return_tr",False),("median_return_va",False),("p10_return_va",False),("median_mdd_va",False)]:
        cand["rank_"+col]=cand[col].rank(ascending=asc,method="average",pct=True)
    rankcols=[c for c in cand if c.startswith("rank_")]
    cand["robust_rank_score"]=cand[rankcols].mean(axis=1)
    cand=cand.sort_values("robust_rank_score").reset_index(drop=True)
    oos=allsum[allsum.segment=="OOS_2024_PLUS"][["strategy","cohorts","median_return","p10_return","worst_return","median_mdd","worst_mdd","win_vs_lump"]]
    oos=oos.rename(columns={c:f"{c}_oos" for c in oos.columns if c!="strategy"})
    cand=cand.merge(oos,on="strategy",how="left")
    cand.head(40).to_csv(OUT/"robust_shortlist_top40.csv",index=False)

    latest=d.iloc[-1]
    current={
        "latest_date":str(pd.Timestamp(latest.date).date()),"latest_close":float(latest.close),
        "dd365":float(latest.dd365),"mom30":float(latest.mom30),"mom7":float(latest.mom7),
        "price_vs_ma200":float(latest.close/latest.ma200-1.0),
        "cohorts":len(starts),
        "strict_currentlike_cohorts":int(segs["CURRENTLIKE_STRICT"].start.nunique()),
        "loose_currentlike_cohorts":int(segs["CURRENTLIKE_LOOSE"].start.nunique()),
        "deep_dd40_cohorts":int(segs["DEEP_DD40"].start.nunique()),
    }
    (OUT/"latest_state.json").write_text(json.dumps(current,indent=2),encoding="utf-8")

    # Compact human-readable report.
    lines=[]
    lines.append("# ETH entry allocation daily backtest\n")
    lines.append(f"Data: Binance ETHUSDT daily {info['start']} to {info['end']} ({info['rows']} rows).")
    lines.append(f"Monthly rolling 24M cohorts: {len(starts)}; fee per buy: {FEE:.3%}.")
    lines.append("Grid: initial 20-80%; remaining capital DCA 1/3/6/9/12 months; dip triggers accelerate future DCA tranches without increasing total capital.\n")
    lines.append("## Latest state")
    lines.append(json.dumps(current,indent=2))
    lines.append("\n## Robust shortlist (selected on train+validation; OOS not used for selection)")
    cols=["strategy","robust_rank_score","median_return_tr","p10_return_tr","median_return_va","p10_return_va","median_return_oos","p10_return_oos","median_mdd_oos","win_vs_lump_oos"]
    lines.append(cand.head(15)[[c for c in cols if c in cand]].to_markdown(index=False))
    for seg in ["DEEP_DD40","CURRENTLIKE_LOOSE","CURRENTLIKE_STRICT","OOS_2024_PLUS"]:
        s=allsum[allsum.segment==seg]
        lines.append(f"\n## {seg} top 15 by median return")
        keep=["strategy","cohorts","median_return","p10_return","worst_return","median_mdd","worst_mdd","win_vs_lump"]
        lines.append(s.head(15)[keep].to_markdown(index=False) if len(s) else "No cohorts")
    (OUT/"README_results.md").write_text("\n".join(lines),encoding="utf-8")
    print("DONE",json.dumps(current),flush=True)

if __name__=="__main__":
    main()
