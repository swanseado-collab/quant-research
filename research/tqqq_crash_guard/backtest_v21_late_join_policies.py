from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from backtest_v1 import perf
from backtest_v6_bull_hold import Cfg
from backtest_v13_validation import RunSpec, simulate_spec, execute_target_var
from backtest_v15_cycle_and_synth_stress import build_live_panel

OUT = Path(__file__).resolve().parent / "results_v21_late_join_policies"
OUT.mkdir(parents=True, exist_ok=True)
BASE = Cfg(-0.25, 80, 110, 0.35, 110, 50, -0.18)
COST = 0.0005

POLICIES = [
    ("IMM25", "immediate", 0.25, None),
    ("IMM50", "immediate", 0.50, None),
    ("IMM100", "immediate", 1.00, None),
    ("PB5_50", "pullback", 0.50, -0.05),
    ("PB5_100", "pullback", 1.00, -0.05),
    ("PB10_50", "pullback", 0.50, -0.10),
    ("PB10_100", "pullback", 1.00, -0.10),
    ("PB15_50", "pullback", 0.50, -0.15),
    ("PB15_100", "pullback", 1.00, -0.15),
    ("WAIT_NEXT", "wait_next", 0.0, None),
]


def age_bucket(age):
    if age <= 504: return "253_504"
    if age <= 756: return "505_756"
    if age <= 1008: return "757_1008"
    return "1009_plus"


def run_policy(p, base_tr, start_idx, end_idx, mode, late_weight, pb_threshold):
    dates = pd.DatetimeIndex(p.date)
    tpx = p.tqqq.to_numpy(float)
    q = p.qqq.to_numpy(float)
    cy = p.cash_yield_pct.to_numpy(float)
    trade_map = {pd.Timestamp(r.date): r for _, r in base_tr.iterrows()}

    # Pullback is measured from the running QQQ high observed after the adoption date.
    running_high = q[start_idx]
    cash = 1.0; shares = 0.0; avg = np.nan
    joined_current = False
    following_future = False
    late_entry_date = pd.NaT
    late_entry_qqq_dd = np.nan
    eq = []

    for i in range(start_idx, end_idx + 1):
        if i > start_idx:
            cash *= 1.0 + max(0.0, cy[i-1] / 100.0) / 252.0
        d = dates[i]
        running_high = max(running_high, q[i])
        since_start_dd = q[i] / running_high - 1.0

        # Late-adoption action occurs after the adoption close; earliest fill is next trading-day close.
        if i == start_idx + 1 and mode == "immediate":
            cash, shares, avg, _, _ = execute_target_var(cash, shares, avg, tpx[i], late_weight, COST)
            joined_current = shares > 0
            if joined_current:
                late_entry_date = d; late_entry_qqq_dd = since_start_dd

        # For pullback policy, only join if current V7 bull has not already ended.
        if mode == "pullback" and (not joined_current) and (not following_future) and i > start_idx:
            if since_start_dd <= pb_threshold:
                # Confirm the V7 state on this date is still BULL; infer from whether an exit has happened below.
                cash, shares, avg, _, _ = execute_target_var(cash, shares, avg, tpx[i], late_weight, COST)
                joined_current = shares > 0
                if joined_current:
                    late_entry_date = d; late_entry_qqq_dd = since_start_dd

        if d in trade_map and i > start_idx:
            r = trade_map[d]; reason = str(r.reason); target = float(r.target_weight)

            # If current campaign ends before a pullback entry, cancel the stale pullback plan.
            if reason == "BULL_EXIT":
                if joined_current:
                    cash, shares, avg, _, _ = execute_target_var(cash, shares, avg, tpx[i], 0.0, COST)
                    joined_current = False
                # From now on all policies wait for and follow subsequent fresh V7 campaigns.
                following_future = True

            elif following_future and reason == "REVERSAL_ENTRY":
                cash, shares, avg, _, _ = execute_target_var(cash, shares, avg, tpx[i], target, COST)
            elif following_future and reason == "BULL_FULL" and shares > 0:
                cash, shares, avg, _, _ = execute_target_var(cash, shares, avg, tpx[i], target, COST)

            # A fresh campaign may itself later exit.
            if following_future and reason == "BULL_EXIT" and shares > 0:
                cash, shares, avg, _, _ = execute_target_var(cash, shares, avg, tpx[i], 0.0, COST)

        eq.append(cash + shares * tpx[i])

    s = pd.Series(eq, index=dates[start_idx:end_idx+1])
    m = perf(s)
    return m, late_entry_date, late_entry_qqq_dd


def main():
    p = build_live_panel(); dates = pd.DatetimeIndex(p.date)
    _, _, tr, path = simulate_spec(p, RunSpec(BASE), save_path=True)

    fulls = [pd.Timestamp(x) for x in tr.loc[tr.reason.eq("BULL_FULL"), "date"]]
    exits = [pd.Timestamp(x) for x in tr.loc[tr.reason.eq("BULL_EXIT"), "date"]]
    full_idx = {d: dates.get_loc(d) for d in fulls}

    # Build monthly adoption starts only after 504 trading days of an active BULL.
    cur_full = None; cur_full_i = None; rows0 = []
    for i, d in enumerate(dates):
        if d in full_idx:
            cur_full = d; cur_full_i = i
        if d in exits:
            cur_full = None; cur_full_i = None
        if cur_full is not None and str(path.iloc[i].state) == "BULL":
            age = i - cur_full_i
            if age >= 505:
                rows0.append((i, d, cur_full, age))
    starts = pd.DataFrame(rows0, columns=["idx","date","full_date","age_td"])
    starts["month"] = starts.date.dt.to_period("M")
    starts = starts.groupby("month", as_index=False).first()

    raw = []
    for _, r in starts.iterrows():
        si = int(r.idx); sd = pd.Timestamp(r.date); age = int(r.age_td)
        for yrs in (3,5):
            target_date = sd + pd.DateOffset(years=yrs)
            valid = np.flatnonzero(dates <= target_date)
            if len(valid) == 0: continue
            ei = int(valid[-1])
            if (dates[ei] - sd).days < int(365.2425 * yrs) - 10: continue
            for name, mode, weight, pb in POLICIES:
                m, ld, ldd = run_policy(p, tr, si, ei, mode, weight, pb)
                raw.append({
                    "start_date": sd, "full_date": pd.Timestamp(r.full_date), "age_td": age,
                    "age_bucket": age_bucket(age), "horizon_y": yrs,
                    "policy": name, "CAGR": m["CAGR"], "MDD": m["MDD"], "Final": m["Final"],
                    "late_entry_date": ld, "late_entry_qqq_drawdown": ldd,
                })
    rdf = pd.DataFrame(raw)
    rdf.to_csv(OUT / "v21_raw.csv", index=False)

    # Compare each policy against WAIT_NEXT on identical adoption dates.
    wait = rdf[rdf.policy.eq("WAIT_NEXT")][["start_date","horizon_y","Final","CAGR"]].rename(columns={"Final":"WaitFinal","CAGR":"WaitCAGR"})
    comp = rdf.merge(wait, on=["start_date","horizon_y"], how="left")
    comp["OutperformedWait"] = comp.Final > comp.WaitFinal
    comp["CAGR_Diff_vs_Wait"] = comp.CAGR - comp.WaitCAGR

    summaries = []
    for (h, b, pol), z in comp.groupby(["horizon_y","age_bucket","policy"]):
        summaries.append({
            "horizon_y": h, "age_bucket": b, "policy": pol,
            "n_month_starts": len(z), "n_distinct_campaigns": z.full_date.nunique(),
            "CAGR_Median": z.CAGR.median(), "CAGR_P10": z.CAGR.quantile(.10),
            "MDD_Median": z.MDD.median(), "MDD_Worst": z.MDD.min(),
            "Final_Median": z.Final.median(),
            "WinRate_vs_Wait": z.OutperformedWait.mean(),
            "CAGR_Diff_vs_Wait_Median": z.CAGR_Diff_vs_Wait.median(),
            "CAGR_Diff_vs_Wait_P10": z.CAGR_Diff_vs_Wait.quantile(.10),
            "EntryRate": z.late_entry_date.notna().mean(),
        })
    sdf = pd.DataFrame(summaries)
    sdf.to_csv(OUT / "v21_summary.csv", index=False)

    # Current-age matched completed cycles: test each policy starting at exactly today's age.
    latest_i = len(p)-1; latest_date = dates[-1]
    prior_full = [d for d in fulls if d <= latest_date]
    current_full = max(prior_full); current_age = latest_i - dates.get_loc(current_full)
    snaps = []
    for fd in fulls:
        fi = dates.get_loc(fd); si = fi + current_age
        later = [x for x in exits if x > fd]
        if not later: continue
        ex = later[0]; exi = dates.get_loc(ex)
        if si >= exi: continue
        for name, mode, weight, pb in POLICIES:
            m, ld, ldd = run_policy(p, tr, si, exi, mode, weight, pb)
            snaps.append({
                "campaign_full_date": fd, "matched_start_date": dates[si], "exit_date": ex,
                "matched_age_td": current_age, "policy": name,
                "remaining_days": (ex-dates[si]).days,
                "Return_to_exit": m["Final"]-1.0, "CAGR_to_exit": m["CAGR"], "MDD_to_exit": m["MDD"],
                "late_entry_date": ld, "late_entry_qqq_drawdown": ldd,
            })
    snap = pd.DataFrame(snaps)
    snap.to_csv(OUT / "v21_exact_current_age_completed_cycles.csv", index=False)
    pd.DataFrame([{"latest_date": latest_date, "current_full_date": current_full, "current_age_td": current_age}]).to_csv(OUT / "v21_current_age.csv", index=False)

    print("=== V21 CURRENT AGE ===")
    print(f"latest={latest_date.date()} full={current_full.date()} age_td={current_age}")
    print("\n=== V21 AGE 757-1008 SUMMARY ===")
    show = sdf[sdf.age_bucket.eq("757_1008")]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== V21 EXACT CURRENT-AGE COMPLETED CYCLES ===")
    print(snap.to_string(index=False, float_format=lambda x: f"{x:.4f}") if len(snap) else "none")

if __name__ == "__main__":
    main()
