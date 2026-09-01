from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from backtest_v6_bull_hold import Cfg
from backtest_v13_validation import RunSpec, execute_target_var, simulate_spec
from backtest_v23_ndx_history_extension import build_extended_panel

OUT = Path(__file__).resolve().parent / "results_v30_rolling_20y_monthly_accumulation"
OUT.mkdir(parents=True, exist_ok=True)
BASE = Cfg(-0.25, 80, 110, 0.35, 110, 50, -0.18)
MONTHLY_CONTRIBUTION = 1.0
COST = 0.0005


def nearest_on_or_after(dates: pd.DatetimeIndex, target: pd.Timestamp):
    x = dates[dates >= target]
    return x[0] if len(x) else None


def nearest_on_or_before(dates: pd.DatetimeIndex, target: pd.Timestamp):
    x = dates[dates <= target]
    return x[-1] if len(x) else None


def contribution_mask(dates: pd.DatetimeIndex) -> np.ndarray:
    periods = dates.to_period("M")
    return np.r_[True, periods[1:] != periods[:-1]]


def xirr(cashflows: list[tuple[pd.Timestamp, float]]) -> float:
    if not cashflows:
        return np.nan
    cfs = [(pd.Timestamp(d), float(v)) for d, v in cashflows]
    d0 = cfs[0][0]
    years = np.array([(d - d0).days / 365.2425 for d, _ in cfs], float)
    vals = np.array([v for _, v in cfs], float)

    def npv(r):
        return float(np.sum(vals / np.power(1.0 + r, years)))

    lo, hi = -0.9999, 1.0
    flo, fhi = npv(lo), npv(hi)
    while flo * fhi > 0 and hi < 1e6:
        hi = hi * 2.0 + 1.0
        fhi = npv(hi)
    if flo * fhi > 0:
        return np.nan
    for _ in range(200):
        mid = (lo + hi) / 2.0
        fm = npv(mid)
        if abs(fm) < 1e-12:
            return mid
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return (lo + hi) / 2.0


def direct_buy(cash, shares, avg_cost, px, budget, cost=COST):
    budget = min(float(budget), float(cash))
    if budget <= 1e-12:
        return cash, shares, avg_cost, 0.0, 0.0
    notional = budget / (1.0 + cost)
    fee = notional * cost
    qty = notional / px
    old_basis = shares * avg_cost if (shares > 0 and np.isfinite(avg_cost)) else 0.0
    shares += qty
    cash -= notional + fee
    avg_cost = (old_basis + notional + fee) / shares
    return cash, shares, avg_cost, notional, fee


def adjusted_mdd(equity: list[float], flows: list[float]) -> float:
    if len(equity) < 2:
        return np.nan
    nav = [1.0]
    for i in range(1, len(equity)):
        prev = equity[i - 1]
        if prev <= 0:
            nav.append(nav[-1])
            continue
        r = (equity[i] - flows[i]) / prev - 1.0
        nav.append(nav[-1] * (1.0 + r))
    s = pd.Series(nav, dtype=float)
    return float((s / s.cummax() - 1.0).min())


def finish_metrics(dates, equity, flows, contrib_dates):
    total_contrib = float(np.sum(flows))
    final = float(equity[-1])
    cfs = [(d, -MONTHLY_CONTRIBUTION) for d in contrib_dates]
    cfs.append((pd.Timestamp(dates[-1]), final))
    return {
        "TotalContributed": total_contrib,
        "FinalWealth": final,
        "FinalToContributed": final / total_contrib if total_contrib > 0 else np.nan,
        "XIRR": xirr(cfs),
        "FlowAdjusted_MDD": adjusted_mdd(equity, flows),
    }


def simulate_v7_accum(p: pd.DataFrame, mode: str):
    # mode STRICT: all new monthly contributions stay in cash during an existing campaign.
    # mode STATE_DCA: new contributions follow the current V7 stage (35% starter / 100% bull).
    cfg = BASE
    q = p.qqq.to_numpy(float)
    t = p.tqqq.to_numpy(float)
    dates = pd.DatetimeIndex(p.date)
    cy = p.cash_yield_pct.to_numpy(float)
    dd = p.dd252.to_numpy(float)
    fma = p[f"ma{cfg.fast_ma}"].to_numpy(float)
    fullma = p[f"ma{cfg.full_ma}"].to_numpy(float)
    xma = p[f"ma{cfg.exit_ma}"].to_numpy(float)
    fast3 = p[f"above_ma{cfg.fast_ma}_3d"].to_numpy(bool)
    is_contrib = contribution_mask(dates)

    cash = 0.0
    shares = 0.0
    avg = np.nan
    armed = False
    stage = 0
    scheduled = None  # (execution_index, target, reason)
    eqs, flows, trades = [], [], []
    contrib_dates = []

    for i in range(len(p)):
        if i > 0:
            cash *= 1.0 + max(0.0, cy[i - 1] / 100.0) / 252.0

        flow = MONTHLY_CONTRIBUTION if is_contrib[i] else 0.0
        if flow > 0:
            cash += flow
            contrib_dates.append(dates[i])

        executed_today = False
        if scheduled is not None and i >= scheduled[0]:
            _, target, reason = scheduled
            before = shares
            cash, shares, avg, notional, fee = execute_target_var(cash, shares, avg, t[i], target, COST)
            if abs(notional) > 1e-12:
                trades.append((dates[i], reason, target, t[i], notional, fee))
            if before > 0 and shares == 0:
                stage = 0
                armed = bool(np.isfinite(dd[i]) and dd[i] <= cfg.arm_dd)
            elif shares > 0:
                stage = 2 if target >= 0.999 else 1
            scheduled = None
            executed_today = True

        # Optional ongoing DCA only uses already-known state, never a same-close signal.
        if mode == "STATE_DCA" and flow > 0 and not executed_today and shares > 0:
            frac = cfg.starter_w if stage == 1 else (1.0 if stage == 2 else 0.0)
            if frac > 0:
                cash, shares, avg, notional, fee = direct_buy(cash, shares, avg, t[i], flow * frac, COST)
                if notional > 1e-12:
                    trades.append((dates[i], "MONTHLY_ADD", frac, t[i], notional, fee))

        eq = cash + shares * t[i]
        eqs.append(eq)
        flows.append(flow)

        if np.isfinite(dd[i]) and dd[i] <= cfg.arm_dd:
            armed = True

        if scheduled is not None:
            continue

        exit_signal = False
        if shares > 0 and i >= cfg.exit_slope_lb and np.isfinite(xma[i]) and np.isfinite(xma[i - cfg.exit_slope_lb]):
            exit_signal = (
                q[i] < xma[i]
                and xma[i] < xma[i - cfg.exit_slope_lb]
                and np.isfinite(dd[i])
                and dd[i] <= cfg.exit_dd
            )
        if exit_signal:
            if i + 1 < len(p):
                scheduled = (i + 1, 0.0, "BULL_EXIT")
            continue

        if shares <= 0:
            if armed and i >= cfg.entry_slope_lb and fast3[i] and np.isfinite(fma[i - cfg.entry_slope_lb]):
                if fma[i] > fma[i - cfg.entry_slope_lb] and i + 1 < len(p):
                    scheduled = (i + 1, cfg.starter_w, "REVERSAL_ENTRY")
            continue

        if stage == 1 and i >= cfg.entry_slope_lb and np.isfinite(fullma[i]) and np.isfinite(fullma[i - cfg.entry_slope_lb]):
            full_ok = q[i] > fullma[i] and fullma[i] > fullma[i - cfg.entry_slope_lb]
            if full_ok and i + 1 < len(p):
                scheduled = (i + 1, 1.0, "BULL_FULL")

    met = finish_metrics(dates, eqs, flows, contrib_dates)
    tr = pd.DataFrame(trades, columns=["date", "reason", "target_weight", "price", "notional", "fee"])
    met.update({
        "StarterEntries": int((tr.reason == "REVERSAL_ENTRY").sum()) if len(tr) else 0,
        "FullEntries": int((tr.reason == "BULL_FULL").sum()) if len(tr) else 0,
        "Exits": int((tr.reason == "BULL_EXIT").sum()) if len(tr) else 0,
        "SignalTrades": int(tr.reason.isin(["REVERSAL_ENTRY", "BULL_FULL", "BULL_EXIT"]).sum()) if len(tr) else 0,
        "MonthlyAdds": int((tr.reason == "MONTHLY_ADD").sum()) if len(tr) else 0,
    })
    return met, tr


def simulate_dca_asset(p: pd.DataFrame, asset: str):
    dates = pd.DatetimeIndex(p.date)
    px = p[asset].to_numpy(float)
    cy = p.cash_yield_pct.to_numpy(float)
    is_contrib = contribution_mask(dates)
    cash, shares = 0.0, 0.0
    eqs, flows, contrib_dates = [], [], []
    for i in range(len(p)):
        if i > 0:
            cash *= 1.0 + max(0.0, cy[i - 1] / 100.0) / 252.0
        flow = MONTHLY_CONTRIBUTION if is_contrib[i] else 0.0
        if flow > 0:
            cash += flow
            contrib_dates.append(dates[i])
            budget = cash
            notional = budget / (1.0 + COST)
            fee = notional * COST
            shares += notional / px[i]
            cash -= notional + fee
        eqs.append(cash + shares * px[i])
        flows.append(flow)
    return finish_metrics(dates, eqs, flows, contrib_dates)


def simulate_cash_accum(p: pd.DataFrame):
    dates = pd.DatetimeIndex(p.date)
    cy = p.cash_yield_pct.to_numpy(float)
    is_contrib = contribution_mask(dates)
    cash = 0.0
    eqs, flows, contrib_dates = [], [], []
    for i in range(len(p)):
        if i > 0:
            cash *= 1.0 + max(0.0, cy[i - 1] / 100.0) / 252.0
        flow = MONTHLY_CONTRIBUTION if is_contrib[i] else 0.0
        if flow > 0:
            cash += flow
            contrib_dates.append(dates[i])
        eqs.append(cash)
        flows.append(flow)
    return finish_metrics(dates, eqs, flows, contrib_dates)


def main():
    p, _, _ = build_extended_panel()
    p = p.copy()
    p["date"] = pd.to_datetime(p.date)
    dates = pd.DatetimeIndex(p.date)
    last = dates[-1]

    rows = []
    event_rows = []
    first_year = max(1986, int(dates[0].year))
    last_start_year = int(last.year) - 20

    for year in range(first_year, last_start_year + 1):
        requested_start = pd.Timestamp(f"{year}-01-01")
        requested_end = requested_start + pd.DateOffset(years=20)
        start = nearest_on_or_after(dates, requested_start)
        end = nearest_on_or_before(dates, requested_end)
        if start is None or end is None or end <= start:
            continue
        sub = p[(p.date >= start) & (p.date <= end)].copy().reset_index(drop=True)

        strict, tr_s = simulate_v7_accum(sub, "STRICT")
        state, tr_d = simulate_v7_accum(sub, "STATE_DCA")
        cash = simulate_cash_accum(sub)
        qqq = simulate_dca_asset(sub, "qqq")
        tqqq = simulate_dca_asset(sub, "tqqq")

        row = {"start_year": year, "start": start, "end": end, "months_contributed": int(strict["TotalContributed"])}
        for prefix, m in [("V7_STRICT", strict), ("V7_STATE_DCA", state), ("CASH", cash), ("QQQ_DCA", qqq), ("TQQQ_DCA", tqqq)]:
            for k, v in m.items():
                row[f"{prefix}_{k}"] = v
        rows.append(row)

        for label, tr in [("V7_STRICT", tr_s), ("V7_STATE_DCA", tr_d)]:
            if len(tr):
                z = tr.copy()
                z.insert(0, "strategy", label)
                z.insert(0, "start_year", year)
                event_rows.extend(z.to_dict("records"))

    df = pd.DataFrame(rows)
    ev = pd.DataFrame(event_rows)
    df.to_csv(OUT / "v30_20y_monthly_accumulation_windows.csv", index=False)
    ev.to_csv(OUT / "v30_trade_events.csv", index=False)

    selected_years = [1986, 1990, 1995, 2000, 2006]
    selected = df[df.start_year.isin(selected_years)].copy()
    selected.to_csv(OUT / "v30_selected_windows.csv", index=False)

    def summary_for(prefix):
        return {
            f"{prefix}_XIRR_min": df[f"{prefix}_XIRR"].min(),
            f"{prefix}_XIRR_median": df[f"{prefix}_XIRR"].median(),
            f"{prefix}_XIRR_max": df[f"{prefix}_XIRR"].max(),
            f"{prefix}_FinalToContributed_min": df[f"{prefix}_FinalToContributed"].min(),
            f"{prefix}_FinalToContributed_median": df[f"{prefix}_FinalToContributed"].median(),
            f"{prefix}_FinalToContributed_max": df[f"{prefix}_FinalToContributed"].max(),
            f"{prefix}_MDD_worst": df[f"{prefix}_FlowAdjusted_MDD"].min(),
            f"{prefix}_MDD_median": df[f"{prefix}_FlowAdjusted_MDD"].median(),
        }

    srow = {"n_windows": len(df), "start_year_min": int(df.start_year.min()), "start_year_max": int(df.start_year.max())}
    for prefix in ["V7_STRICT", "V7_STATE_DCA", "CASH", "QQQ_DCA", "TQQQ_DCA"]:
        srow.update(summary_for(prefix))
    srow.update({
        "STRICT_beats_CASH_XIRR_fraction": float((df.V7_STRICT_XIRR > df.CASH_XIRR).mean()),
        "STRICT_beats_QQQ_XIRR_fraction": float((df.V7_STRICT_XIRR > df.QQQ_DCA_XIRR).mean()),
        "STATE_DCA_beats_CASH_XIRR_fraction": float((df.V7_STATE_DCA_XIRR > df.CASH_XIRR).mean()),
        "STATE_DCA_beats_QQQ_XIRR_fraction": float((df.V7_STATE_DCA_XIRR > df.QQQ_DCA_XIRR).mean()),
        "STRICT_vs_STATE_XIRR_win_fraction": float((df.V7_STRICT_XIRR > df.V7_STATE_DCA_XIRR).mean()),
        "STRICT_Starter_median": float(df.V7_STRICT_StarterEntries.median()),
        "STRICT_Full_median": float(df.V7_STRICT_FullEntries.median()),
    })
    summary = pd.DataFrame([srow])
    summary.to_csv(OUT / "v30_summary.csv", index=False)

    cols = [
        "start_year","start","end","months_contributed",
        "V7_STRICT_FinalWealth","V7_STRICT_FinalToContributed","V7_STRICT_XIRR","V7_STRICT_FlowAdjusted_MDD","V7_STRICT_StarterEntries","V7_STRICT_FullEntries",
        "V7_STATE_DCA_FinalWealth","V7_STATE_DCA_FinalToContributed","V7_STATE_DCA_XIRR","V7_STATE_DCA_FlowAdjusted_MDD",
        "CASH_FinalWealth","CASH_XIRR",
        "QQQ_DCA_FinalWealth","QQQ_DCA_XIRR","QQQ_DCA_FlowAdjusted_MDD",
        "TQQQ_DCA_FinalWealth","TQQQ_DCA_XIRR","TQQQ_DCA_FlowAdjusted_MDD",
    ]
    selected[cols].to_csv(OUT / "v30_selected_compact.csv", index=False)
    df.sort_values("V7_STRICT_XIRR").head(5)[cols].to_csv(OUT / "v30_strict_worst5.csv", index=False)
    df.sort_values("V7_STRICT_XIRR", ascending=False).head(5)[cols].to_csv(OUT / "v30_strict_best5.csv", index=False)

    print("=== V30 SUMMARY ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== SELECTED WINDOWS ===")
    print(selected[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== STRICT WORST 5 ===")
    print(df.sort_values("V7_STRICT_XIRR").head(5)[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== STRICT BEST 5 ===")
    print(df.sort_values("V7_STRICT_XIRR", ascending=False).head(5)[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
