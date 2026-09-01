from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_v1 import load_panel, perf
from backtest_v2 import add_features
from backtest_v6_bull_hold import Cfg

OUT = Path(__file__).resolve().parent / "results_v13_validation"
OUT.mkdir(parents=True, exist_ok=True)

BASE = Cfg(-0.25, 80, 110, 0.35, 110, 50, -0.18)


@dataclass(frozen=True)
class RunSpec:
    cfg: Cfg
    delay: int = 0          # extra trading days beyond the baseline t+1-close execution
    cost: float = 0.0005    # one-way proportional transaction cost
    cash_y_mult: float = 1.0


def ensure_features(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    mas = (60, 70, 80, 90, 100, 110, 120)
    for ma in mas:
        p[f"ma{ma}"] = p.qqq.rolling(ma, min_periods=ma).mean()
        above = p.qqq > p[f"ma{ma}"]
        p[f"above_ma{ma}_3d"] = above.rolling(3, min_periods=3).sum().eq(3)
    p["hi252"] = p.qqq.rolling(252, min_periods=252).max()
    p["dd252"] = p.qqq / p.hi252 - 1.0
    return p


def execute_target_var(cash, shares, avg_cost, px, target, cost):
    equity = cash + shares * px
    target = float(np.clip(target, 0.0, 1.0))
    desired = target * equity
    current = shares * px
    delta = desired - current
    notional = fee = 0.0
    if delta > 1e-12:
        buy = min(delta, cash / (1.0 + cost))
        if buy > 0:
            qty = buy / px
            fee = buy * cost
            old_basis = shares * avg_cost if (shares > 0 and np.isfinite(avg_cost)) else 0.0
            shares += qty
            cash -= buy + fee
            avg_cost = (old_basis + buy + fee) / shares
            notional = buy
    elif delta < -1e-12:
        sell = min(-delta, current)
        if sell > 0:
            qty = sell / px
            fee = sell * cost
            shares -= qty
            cash += sell - fee
            notional = -sell
            if shares <= 1e-12:
                shares = 0.0
                avg_cost = np.nan
    return cash, shares, avg_cost, notional, fee


def simulate_spec(p: pd.DataFrame, spec: RunSpec, save_path=False):
    cfg = spec.cfg
    q = p.qqq.to_numpy(float)
    t = p.tqqq.to_numpy(float)
    dates = pd.DatetimeIndex(p.date)
    cy = p.cash_yield_pct.to_numpy(float)
    dd = p.dd252.to_numpy(float)
    fma = p[f"ma{cfg.fast_ma}"].to_numpy(float)
    fullma = p[f"ma{cfg.full_ma}"].to_numpy(float)
    xma = p[f"ma{cfg.exit_ma}"].to_numpy(float)
    fast3 = p[f"above_ma{cfg.fast_ma}_3d"].to_numpy(bool)

    cash = 1.0
    shares = 0.0
    avg = np.nan
    armed = False
    stage = 0
    scheduled = None  # (execution_index, target, reason)
    eqs, ws, states, trades = [], [], [], []

    for i in range(len(p)):
        if i > 0:
            cash *= 1.0 + max(0.0, cy[i - 1] / 100.0) * spec.cash_y_mult / 252.0

        if scheduled is not None and i >= scheduled[0]:
            _, target, reason = scheduled
            before = shares
            cash, shares, avg, notional, fee = execute_target_var(cash, shares, avg, t[i], target, spec.cost)
            if abs(notional) > 1e-12:
                trades.append((dates[i], reason, target, t[i], notional, fee, cash, shares, avg))
            if before > 0 and shares == 0:
                stage = 0
                armed = bool(np.isfinite(dd[i]) and dd[i] <= cfg.arm_dd)
            elif shares > 0:
                stage = 2 if target >= .999 else 1
            scheduled = None

        eq = cash + shares * t[i]
        w = shares * t[i] / eq if eq > 0 else 0.0
        eqs.append(eq)
        ws.append(w)

        if np.isfinite(dd[i]) and dd[i] <= cfg.arm_dd:
            armed = True

        # Do not stack additional orders while a delayed order is waiting to execute.
        if scheduled is not None:
            states.append("ORDER_WAIT")
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
            j = min(len(p) - 1, i + 1 + spec.delay)
            scheduled = (j, 0.0, "BULL_EXIT")
            states.append("EXIT_SIGNAL")
            continue

        if shares <= 0:
            states.append("ARMED" if armed else "CASH")
            if armed and i >= cfg.entry_slope_lb and fast3[i] and np.isfinite(fma[i - cfg.entry_slope_lb]):
                if fma[i] > fma[i - cfg.entry_slope_lb]:
                    j = min(len(p) - 1, i + 1 + spec.delay)
                    scheduled = (j, cfg.starter_w, "REVERSAL_ENTRY")
            continue

        states.append("STARTER" if stage == 1 else "BULL")
        if stage == 1 and i >= cfg.entry_slope_lb and np.isfinite(fullma[i]) and np.isfinite(fullma[i - cfg.entry_slope_lb]):
            full_ok = q[i] > fullma[i] and fullma[i] > fullma[i - cfg.entry_slope_lb]
            if full_ok:
                j = min(len(p) - 1, i + 1 + spec.delay)
                scheduled = (j, 1.0, "BULL_FULL")

    eq = pd.Series(eqs, index=dates, name="equity")
    w = pd.Series(ws, index=dates, name="weight")
    tr = pd.DataFrame(trades, columns=[
        "date", "reason", "target_weight", "price", "notional", "fee",
        "cash_after", "shares_after", "avg_cost_after"
    ])
    path = None
    if save_path:
        path = pd.DataFrame({
            "date": dates, "qqq": q, "tqqq": t, "equity": eqs,
            "weight": ws, "state": states, "dd252": dd,
        })
    return eq, w, tr, path


def metrics_slice(eq: pd.Series, start=None, end=None):
    x = eq
    if start is not None:
        x = x.loc[pd.Timestamp(start):]
    if end is not None:
        x = x.loc[:pd.Timestamp(end)]
    if len(x) < 2:
        return {"CAGR": np.nan, "MDD": np.nan, "Final": np.nan}
    return perf(x)


def test_return_mdd(eq: pd.Series, start, end):
    x = eq.loc[pd.Timestamp(start):pd.Timestamp(end)]
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    ret = float(x.iloc[-1] / x.iloc[0] - 1.0)
    yrs = (x.index[-1] - x.index[0]).days / 365.2425
    cagr = float((x.iloc[-1] / x.iloc[0]) ** (1 / yrs) - 1) if yrs > 0 else np.nan
    mdd = float((x / x.cummax() - 1.0).min())
    return ret, cagr, mdd


def neighborhood_cfgs():
    vals = itertools.product(
        (-0.225, -0.25, -0.275),
        (70, 80, 90),
        (100, 110, 120),
        (0.25, 0.35, 0.50),
        (100, 110, 120),
        (40, 50, 60),
        (-0.16, -0.18, -0.20),
    )
    return [Cfg(*v) for v in vals]


def cfg_key(c: Cfg):
    return (c.arm_dd, c.fast_ma, c.full_ma, c.starter_w, c.exit_ma, c.exit_slope_lb, c.exit_dd)


def cfg_row(c: Cfg):
    return {
        "arm_dd": c.arm_dd, "fast_ma": c.fast_ma, "full_ma": c.full_ma,
        "starter_w": c.starter_w, "exit_ma": c.exit_ma,
        "exit_slope_lb": c.exit_slope_lb, "exit_dd": c.exit_dd,
    }


def main():
    p = ensure_features(add_features(load_panel()))
    last_date = p.date.max()
    actual_start = p.loc[p.price_source.eq("actual"), "date"].min()
    cfgs = neighborhood_cfgs()
    print(f"panel {p.date.min().date()} -> {last_date.date()} rows={len(p)} actual_start={actual_start.date()}")
    print(f"neighborhood configs={len(cfgs)}")

    # 1) Parameter-neighborhood robustness + data needed for true expanding walk-forward selection.
    splits = [
        ("WF1", "2009-12-31", "2010-01-01", "2015-12-31"),
        ("WF2", "2015-12-31", "2016-01-01", "2020-12-31"),
        ("WF3", "2020-12-31", "2021-01-01", str(last_date.date())),
    ]
    rows = []
    train_rows = []
    for k, cfg in enumerate(cfgs, 1):
        eq, w, tr, _ = simulate_spec(p, RunSpec(cfg))
        r = cfg_row(cfg)
        r.update(metrics_slice(eq))
        am = metrics_slice(eq, actual_start, last_date)
        r["ActualEra_CAGR"] = am["CAGR"]
        r["ActualEra_MDD"] = am["MDD"]
        r["TradeCount"] = len(tr)
        r["InvestedDayPct"] = float((w > 0.01).mean())
        rows.append(r)
        for label, train_end, test_start, test_end in splits:
            tm = metrics_slice(eq, None, train_end)
            train_rows.append({"split": label, **cfg_row(cfg), "Train_CAGR": tm["CAGR"], "Train_MDD": tm["MDD"]})
        if k % 500 == 0:
            print(f"neighborhood {k}/{len(cfgs)}")
    nd = pd.DataFrame(rows)
    nd.to_csv(OUT / "v13_parameter_neighborhood.csv", index=False)

    # Robustness distribution rather than only the optimum.
    summary = []
    for risk_floor in (-0.70, -0.65, -0.60):
        z = nd[nd.MDD >= risk_floor]
        summary.append({
            "risk_floor": risk_floor, "n": len(z),
            "CAGR_P10": z.CAGR.quantile(.10) if len(z) else np.nan,
            "CAGR_Median": z.CAGR.median() if len(z) else np.nan,
            "CAGR_P90": z.CAGR.quantile(.90) if len(z) else np.nan,
            "ActualEra_CAGR_P10": z.ActualEra_CAGR.quantile(.10) if len(z) else np.nan,
            "ActualEra_CAGR_Median": z.ActualEra_CAGR.median() if len(z) else np.nan,
            "MDD_Median": z.MDD.median() if len(z) else np.nan,
        })
    pd.DataFrame(summary).to_csv(OUT / "v13_neighborhood_summary.csv", index=False)

    # BASE percentile within its local neighborhood.
    base_key = cfg_key(BASE)
    base_row = nd[
        (nd.arm_dd == BASE.arm_dd) & (nd.fast_ma == BASE.fast_ma) &
        (nd.full_ma == BASE.full_ma) & (nd.starter_w == BASE.starter_w) &
        (nd.exit_ma == BASE.exit_ma) & (nd.exit_slope_lb == BASE.exit_slope_lb) &
        (nd.exit_dd == BASE.exit_dd)
    ].iloc[0].to_dict()
    base_row["CAGR_percentile"] = float((nd.CAGR <= base_row["CAGR"]).mean())
    base_row["ActualEra_CAGR_percentile"] = float((nd.ActualEra_CAGR <= base_row["ActualEra_CAGR"]).mean())
    base_row["MDD_percentile_less_negative"] = float((nd.MDD <= base_row["MDD"]).mean())
    pd.DataFrame([base_row]).to_csv(OUT / "v13_base_local_percentile.csv", index=False)

    # 2) Genuine expanding walk-forward: choose cfg using TRAIN ONLY, then score untouched next block.
    train_df = pd.DataFrame(train_rows)
    wf = []
    for label, train_end, test_start, test_end in splits:
        z = train_df[train_df.split == label].copy()
        # Catastrophe-first objective: among configurations whose historical MDD did not exceed 65%, maximize CAGR.
        feasible = z[z.Train_MDD >= -0.65].copy()
        if feasible.empty:
            feasible = z.copy()
        chosen = feasible.sort_values(["Train_CAGR", "Train_MDD"], ascending=[False, False]).iloc[0]
        c = Cfg(chosen.arm_dd, chosen.fast_ma, chosen.full_ma, chosen.starter_w,
                chosen.exit_ma, chosen.exit_slope_lb, chosen.exit_dd)
        eq, w, tr, _ = simulate_spec(p, RunSpec(c))
        tret, tcagr, tmdd = test_return_mdd(eq, test_start, test_end)
        wf.append({
            "split": label, "train_end": train_end, "test_start": test_start, "test_end": test_end,
            **cfg_row(c), "Train_CAGR": chosen.Train_CAGR, "Train_MDD": chosen.Train_MDD,
            "Test_Return": tret, "Test_CAGR": tcagr, "Test_MDD": tmdd,
            "Test_Trades": int(((tr.date >= pd.Timestamp(test_start)) & (tr.date <= pd.Timestamp(test_end))).sum()) if len(tr) else 0,
        })
    wfdf = pd.DataFrame(wf)
    wfdf.to_csv(OUT / "v13_walk_forward.csv", index=False)

    # Fixed V7 subperiod table is robustness only (NOT true OOS because V7 was selected on the full sample).
    beq, bw, btr, bpath = simulate_spec(p, RunSpec(BASE), save_path=True)
    fixed = []
    fixed_windows = [
        ("2010_2015", "2010-02-11", "2015-12-31"),
        ("2016_2020", "2016-01-01", "2020-12-31"),
        ("2021_latest", "2021-01-01", str(last_date.date())),
        ("2018_latest", "2018-01-01", str(last_date.date())),
        ("2020_latest", "2020-01-01", str(last_date.date())),
    ]
    for name, a, b in fixed_windows:
        ret, cagr, mdd = test_return_mdd(beq, a, b)
        fixed.append({"window": name, "start": a, "end": b, "Return": ret, "CAGR": cagr, "MDD": mdd})
    pd.DataFrame(fixed).to_csv(OUT / "v13_fixed_v7_subperiods.csv", index=False)

    # 3) Execution-delay stress. delay=0 means the existing baseline t+1 close.
    delay_rows = []
    for d in (0, 1, 2, 3, 5, 10):
        eq, w, tr, _ = simulate_spec(p, RunSpec(BASE, delay=d))
        r = {"extra_delay_days": d, "effective_execution": f"t+{1+d}_close", **perf(eq)}
        am = perf(eq.loc[actual_start:])
        r["ActualEra_CAGR"] = am["CAGR"]; r["ActualEra_MDD"] = am["MDD"]
        r["TradeCount"] = len(tr); r["InvestedDayPct"] = float((w > .01).mean())
        delay_rows.append(r)
    pd.DataFrame(delay_rows).to_csv(OUT / "v13_delay_stress.csv", index=False)

    # 4) Cost stress. Cost is one-way and applies to every rebalance transaction.
    cost_rows = []
    for bps in (0, 5, 10, 25, 50, 100):
        eq, w, tr, _ = simulate_spec(p, RunSpec(BASE, cost=bps / 10000.0))
        r = {"one_way_cost_bps": bps, **perf(eq)}
        am = perf(eq.loc[actual_start:])
        r["ActualEra_CAGR"] = am["CAGR"]; r["ActualEra_MDD"] = am["MDD"]
        r["TradeCount"] = len(tr)
        cost_rows.append(r)
    pd.DataFrame(cost_rows).to_csv(OUT / "v13_cost_stress.csv", index=False)

    # 5) Waiting-cash-yield sensitivity.
    cash_rows = []
    for mult in (0.0, 0.5, 1.0):
        eq, w, tr, _ = simulate_spec(p, RunSpec(BASE, cash_y_mult=mult))
        r = {"cash_yield_multiplier": mult, **perf(eq)}
        am = perf(eq.loc[actual_start:])
        r["ActualEra_CAGR"] = am["CAGR"]; r["ActualEra_MDD"] = am["MDD"]
        cash_rows.append(r)
    pd.DataFrame(cash_rows).to_csv(OUT / "v13_cash_yield_stress.csv", index=False)

    bpath.to_csv(OUT / "v13_base_path.csv", index=False)
    btr.to_csv(OUT / "v13_base_trades.csv", index=False)

    print("\n=== BASE LOCAL PERCENTILE ===")
    print(pd.DataFrame([base_row]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== NEIGHBORHOOD SUMMARY ===")
    print(pd.DataFrame(summary).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== WALK FORWARD (TRAIN-ONLY SELECTION) ===")
    print(wfdf.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== FIXED V7 SUBPERIODS (ROBUSTNESS, NOT OOS) ===")
    print(pd.DataFrame(fixed).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== DELAY STRESS ===")
    print(pd.DataFrame(delay_rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== COST STRESS ===")
    print(pd.DataFrame(cost_rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== CASH YIELD STRESS ===")
    print(pd.DataFrame(cash_rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
