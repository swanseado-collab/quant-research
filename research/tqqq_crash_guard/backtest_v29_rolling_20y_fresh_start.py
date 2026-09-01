from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from backtest_v1 import perf
from backtest_v6_bull_hold import Cfg
from backtest_v13_validation import RunSpec, simulate_spec
from backtest_v23_ndx_history_extension import build_extended_panel

OUT = Path(__file__).resolve().parent / "results_v29_rolling_20y_fresh_start"
OUT.mkdir(parents=True, exist_ok=True)
BASE = Cfg(-0.25, 80, 110, 0.35, 110, 50, -0.18)


def nearest_on_or_after(dates: pd.DatetimeIndex, target: pd.Timestamp) -> pd.Timestamp | None:
    x = dates[dates >= target]
    return x[0] if len(x) else None


def nearest_on_or_before(dates: pd.DatetimeIndex, target: pd.Timestamp) -> pd.Timestamp | None:
    x = dates[dates <= target]
    return x[-1] if len(x) else None


def bh_metrics(price: pd.Series) -> dict:
    x = price.dropna().astype(float)
    if len(x) < 2:
        return {"CAGR": np.nan, "MDD": np.nan, "Final": np.nan}
    eq = x / x.iloc[0]
    return perf(eq)


def main():
    p, _, _ = build_extended_panel()
    p = p.copy()
    p["date"] = pd.to_datetime(p.date)
    dates = pd.DatetimeIndex(p.date)
    last = dates[-1]

    rows = []
    event_rows = []

    # Annual fresh starts that can complete a full 20 calendar years in the panel.
    first_year = max(1986, int(dates[0].year))
    last_start_year = int(last.year) - 20

    for year in range(first_year, last_start_year + 1):
        requested_start = pd.Timestamp(f"{year}-01-01")
        requested_end = requested_start + pd.DateOffset(years=20)
        start = nearest_on_or_after(dates, requested_start)
        end = nearest_on_or_before(dates, requested_end)
        if start is None or end is None or end <= start:
            continue

        # IMPORTANT: reset to cash/unarmed at the start, but retain precomputed MA/DD features
        # because they were observable on that start date. Simulate beyond the scoring end so
        # a signal on the final scored close does not get incorrectly filled on that same close.
        sim_end_target = end + pd.Timedelta(days=10)
        sim_end = nearest_on_or_before(dates, min(sim_end_target, last))
        sub = p[(p.date >= start) & (p.date <= sim_end)].copy().reset_index(drop=True)
        eq, w, tr, path = simulate_spec(sub, RunSpec(BASE), save_path=True)
        score = eq.loc[:end]
        m = perf(score)

        z = sub[sub.date <= end].copy()
        q = bh_metrics(pd.Series(z.qqq.to_numpy(float), index=pd.DatetimeIndex(z.date)))
        t = bh_metrics(pd.Series(z.tqqq.to_numpy(float), index=pd.DatetimeIndex(z.date)))

        tr20 = tr[(pd.to_datetime(tr.date) >= start) & (pd.to_datetime(tr.date) <= end)].copy()
        starter = int((tr20.reason == "REVERSAL_ENTRY").sum()) if len(tr20) else 0
        full = int((tr20.reason == "BULL_FULL").sum()) if len(tr20) else 0
        exits = int((tr20.reason == "BULL_EXIT").sum()) if len(tr20) else 0
        invested_pct = float((w.loc[:end] > 0.01).mean())

        rows.append({
            "start_year": year,
            "start": start,
            "end": end,
            "calendar_years": (end-start).days / 365.2425,
            "V7_FinalMultiple": m["Final"],
            "V7_CAGR": m["CAGR"],
            "V7_MDD": m["MDD"],
            "V7_StarterEntries": starter,
            "V7_FullEntries": full,
            "V7_Exits": exits,
            "V7_TotalTrades": len(tr20),
            "V7_InvestedDayPct": invested_pct,
            "QQQ_FinalMultiple": q["Final"],
            "QQQ_CAGR": q["CAGR"],
            "QQQ_MDD": q["MDD"],
            "TQQQ3x_FinalMultiple": t["Final"],
            "TQQQ3x_CAGR": t["CAGR"],
            "TQQQ3x_MDD": t["MDD"],
        })

        if len(tr20):
            tt = tr20[["date","reason","target_weight","price"]].copy()
            tt.insert(0, "start_year", year)
            event_rows.extend(tt.to_dict("records"))

    df = pd.DataFrame(rows)
    ev = pd.DataFrame(event_rows)
    df.to_csv(OUT / "v29_20y_fresh_start_windows.csv", index=False)
    ev.to_csv(OUT / "v29_20y_trade_events.csv", index=False)

    # Selected representative start years for easy reading.
    selected_years = [1986, 1990, 1995, 2000, 2006]
    selected = df[df.start_year.isin(selected_years)].copy()
    selected.to_csv(OUT / "v29_selected_windows.csv", index=False)

    summary = pd.DataFrame([{
        "n_windows": len(df),
        "start_year_min": int(df.start_year.min()),
        "start_year_max": int(df.start_year.max()),
        "V7_CAGR_min": df.V7_CAGR.min(),
        "V7_CAGR_median": df.V7_CAGR.median(),
        "V7_CAGR_max": df.V7_CAGR.max(),
        "V7_MDD_worst": df.V7_MDD.min(),
        "V7_MDD_median": df.V7_MDD.median(),
        "V7_FinalMultiple_min": df.V7_FinalMultiple.min(),
        "V7_FinalMultiple_median": df.V7_FinalMultiple.median(),
        "V7_FinalMultiple_max": df.V7_FinalMultiple.max(),
        "Starter_min": int(df.V7_StarterEntries.min()),
        "Starter_median": float(df.V7_StarterEntries.median()),
        "Starter_max": int(df.V7_StarterEntries.max()),
        "Full_min": int(df.V7_FullEntries.min()),
        "Full_median": float(df.V7_FullEntries.median()),
        "Full_max": int(df.V7_FullEntries.max()),
        "TradeCount_min": int(df.V7_TotalTrades.min()),
        "TradeCount_median": float(df.V7_TotalTrades.median()),
        "TradeCount_max": int(df.V7_TotalTrades.max()),
        "V7_beats_QQQ_CAGR_fraction": float((df.V7_CAGR > df.QQQ_CAGR).mean()),
        "V7_beats_QQQ_final_fraction": float((df.V7_FinalMultiple > df.QQQ_FinalMultiple).mean()),
    }])
    summary.to_csv(OUT / "v29_summary.csv", index=False)

    worst = df.sort_values("V7_CAGR").head(5)
    best = df.sort_values("V7_CAGR", ascending=False).head(5)
    worst.to_csv(OUT / "v29_worst5_by_cagr.csv", index=False)
    best.to_csv(OUT / "v29_best5_by_cagr.csv", index=False)

    print("=== V29 SUMMARY ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== SELECTED 20Y FRESH-START WINDOWS ===")
    print(selected.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== WORST 5 BY V7 CAGR ===")
    print(worst.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== BEST 5 BY V7 CAGR ===")
    print(best.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
