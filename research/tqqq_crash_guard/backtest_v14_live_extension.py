from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

from backtest_v1 import load_panel, perf
from backtest_v2 import add_features
from backtest_v6_bull_hold import Cfg
from backtest_v13_validation import RunSpec, ensure_features, simulate_spec

OUT = Path(__file__).resolve().parent / "results_v14_live_extension"
OUT.mkdir(parents=True, exist_ok=True)
BASE = Cfg(-0.25, 80, 110, 0.35, 110, 50, -0.18)
END = "2026-09-02"  # yfinance end is exclusive; captures completed Aug-31 session


def yf_series(ticker: str, start="2025-01-01", end=END, adjusted=True):
    d = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False, threads=False)
    if d is None or len(d) == 0:
        raise RuntimeError(f"No Yahoo data for {ticker}")
    if isinstance(d.columns, pd.MultiIndex):
        # Recent yfinance may return (field, ticker) MultiIndex even for one symbol.
        field = "Adj Close" if adjusted and ("Adj Close", ticker) in d.columns else "Close"
        if (field, ticker) in d.columns:
            s = d[(field, ticker)]
        else:
            # fallback: first matching field at level 0
            cols = [c for c in d.columns if c[0] == field]
            if not cols:
                field = "Close"
                cols = [c for c in d.columns if c[0] == field]
            s = d[cols[0]]
    else:
        field = "Adj Close" if adjusted and "Adj Close" in d.columns else "Close"
        s = d[field]
    s = pd.to_numeric(s, errors="coerce").dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def normalized_overlap_stats(frozen: pd.Series, fresh: pd.Series, anchor: pd.Timestamp, lookback=60):
    z = pd.concat([frozen.rename("frozen"), fresh.rename("fresh")], axis=1).dropna()
    z = z[z.index <= anchor].tail(lookback)
    ratio = z.frozen / z.fresh
    anchor_ratio = float(ratio.iloc[-1])
    normalized = ratio / anchor_ratio - 1.0
    return {
        "n_overlap": len(z),
        "anchor_date": z.index[-1],
        "scale_factor": anchor_ratio,
        "normalized_ratio_mean_abs": float(normalized.abs().mean()),
        "normalized_ratio_max_abs": float(normalized.abs().max()),
    }


def slice_stats(eq: pd.Series, start, end):
    x = eq.loc[pd.Timestamp(start):pd.Timestamp(end)]
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    ret = float(x.iloc[-1] / x.iloc[0] - 1)
    yrs = (x.index[-1] - x.index[0]).days / 365.2425
    cagr = float((x.iloc[-1] / x.iloc[0]) ** (1 / yrs) - 1) if yrs > 0 else np.nan
    mdd = float((x / x.cummax() - 1).min())
    return ret, cagr, mdd


def main():
    frozen = load_panel().copy()
    frozen["date"] = pd.to_datetime(frozen.date)
    frozen_last = pd.Timestamp(frozen.date.max())

    yq = yf_series("QQQ")
    yt = yf_series("TQQQ")
    try:
        yi = yf_series("^IRX", adjusted=False)
    except Exception:
        yi = pd.Series(dtype=float)

    fq = frozen.set_index("date").qqq
    ft = frozen.set_index("date").tqqq
    qdiag = normalized_overlap_stats(fq, yq, frozen_last)
    tdiag = normalized_overlap_stats(ft, yt, frozen_last)
    qscale = qdiag["scale_factor"]
    tscale = tdiag["scale_factor"]

    yq = yq * qscale
    yt = yt * tscale
    fresh = pd.concat([yq.rename("qqq"), yt.rename("tqqq")], axis=1).dropna()
    fresh = fresh[fresh.index > frozen_last].copy()
    if fresh.empty:
        raise RuntimeError(f"Yahoo extension has no dates after frozen last date {frozen_last.date()}")

    if len(yi):
        fresh["cash_yield_pct"] = yi.reindex(fresh.index).ffill()
        fresh["cash_yield_pct"] = fresh["cash_yield_pct"].fillna(float(frozen.cash_yield_pct.iloc[-1]))
    else:
        fresh["cash_yield_pct"] = float(frozen.cash_yield_pct.iloc[-1])
    fresh["price_source"] = "actual_live"
    extra = fresh.reset_index().rename(columns={"index": "date"})

    cols = ["date", "qqq", "tqqq", "price_source", "cash_yield_pct"]
    p = pd.concat([frozen[cols], extra[cols]], ignore_index=True).sort_values("date").drop_duplicates("date", keep="last")
    p = ensure_features(add_features(p.reset_index(drop=True)))

    eq, w, tr, path = simulate_spec(p, RunSpec(BASE), save_path=True)
    latest = p.iloc[-1]
    i = len(p) - 1
    q = float(latest.qqq)
    ma80 = float(latest.ma80)
    ma110 = float(latest.ma110)
    dd252 = float(latest.dd252)
    ma80_slope20 = float(latest.ma80 / p.ma80.iloc[i - 20] - 1) if i >= 20 else np.nan
    ma110_slope20 = float(latest.ma110 / p.ma110.iloc[i - 20] - 1) if i >= 20 else np.nan
    ma110_slope50 = float(latest.ma110 / p.ma110.iloc[i - 50] - 1) if i >= 50 else np.nan
    exit_signal = bool(q < ma110 and ma110 < p.ma110.iloc[i - 50] and dd252 <= -0.18)
    entry_fast_ok = bool(latest.above_ma80_3d and ma80 > p.ma80.iloc[i - 20])
    full_ok = bool(q > ma110 and ma110 > p.ma110.iloc[i - 20])

    oos_start = frozen_last + pd.Timedelta(days=1)
    latest_date = pd.Timestamp(latest.date)
    strat_ret, strat_cagr, strat_mdd = slice_stats(eq, oos_start, latest_date)
    qret = float(p.loc[p.date >= oos_start, "qqq"].iloc[-1] / p.loc[p.date >= oos_start, "qqq"].iloc[0] - 1)
    tret = float(p.loc[p.date >= oos_start, "tqqq"].iloc[-1] / p.loc[p.date >= oos_start, "tqqq"].iloc[0] - 1)

    live_tr = tr[tr.date > frozen_last].copy() if len(tr) else tr.copy()
    state = str(path.iloc[-1].state)
    weight = float(path.iloc[-1].weight)
    last_trade_date = pd.Timestamp(tr.iloc[-1].date) if len(tr) else pd.NaT
    last_trade_reason = str(tr.iloc[-1].reason) if len(tr) else ""

    diag = pd.DataFrame([
        {"asset": "QQQ", **qdiag},
        {"asset": "TQQQ", **tdiag},
    ])
    diag.to_csv(OUT / "v14_overlap_diagnostics.csv", index=False)

    summary = pd.DataFrame([{
        "frozen_last_date": frozen_last,
        "latest_date": latest_date,
        "fresh_rows": len(extra),
        "latest_qqq_scaled_adjclose": q,
        "latest_tqqq_scaled_adjclose": float(latest.tqqq),
        "ma80": ma80,
        "ma110": ma110,
        "dd252": dd252,
        "ma80_slope20": ma80_slope20,
        "ma110_slope20": ma110_slope20,
        "ma110_slope50": ma110_slope50,
        "above_ma80_3d": bool(latest.above_ma80_3d),
        "entry_fast_ok": entry_fast_ok,
        "full_ok": full_ok,
        "exit_signal": exit_signal,
        "strategy_state": state,
        "strategy_weight": weight,
        "last_trade_date": last_trade_date,
        "last_trade_reason": last_trade_reason,
        "fresh_period_strategy_return": strat_ret,
        "fresh_period_strategy_cagr_annualized": strat_cagr,
        "fresh_period_strategy_mdd": strat_mdd,
        "fresh_period_qqq_return": qret,
        "fresh_period_tqqq_return": tret,
        "fresh_period_trade_count": len(live_tr),
    }])
    summary.to_csv(OUT / "v14_live_summary.csv", index=False)
    live_tr.to_csv(OUT / "v14_fresh_trades.csv", index=False)
    path[path.date >= frozen_last - pd.Timedelta(days=180)].to_csv(OUT / "v14_recent_path.csv", index=False)

    print("=== OVERLAP DIAGNOSTICS ===")
    print(diag.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\n=== LIVE / FRESH OOS SUMMARY ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\n=== FRESH TRADES ===")
    print(live_tr.to_string(index=False) if len(live_tr) else "none")


if __name__ == "__main__":
    main()
