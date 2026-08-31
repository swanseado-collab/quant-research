from __future__ import annotations

import io
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://raw.githubusercontent.com/marcusdrewry/tqqq-qqq-tfsa-allocation/main/data"
OUT = Path(__file__).resolve().parent / "results_v1"
OUT.mkdir(parents=True, exist_ok=True)

START = "1999-03-10"
ONE_WAY_COST = 0.0005  # 5 bp
ENTRY_DD = (-0.06, -0.10, -0.15, -0.22)
TARGET_W = (0.15, 0.35, 0.60, 1.00)
TP = 0.50

PERIODS = {
    "DotCom": ("2000-03-10", "2003-03-31"),
    "GFC": ("2007-10-01", "2009-06-30"),
    "COVID": ("2020-02-01", "2020-08-31"),
    "2022_Bear": ("2021-11-01", "2023-03-31"),
    "Actual_TQQQ_Era": ("2010-02-11", "2099-12-31"),
}


def download_csv(name: str) -> pd.DataFrame:
    url = f"{BASE}/{name}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
    return pd.read_csv(io.BytesIO(raw))


def load_panel() -> pd.DataFrame:
    q = download_csv("QQQ.csv")
    s = download_csv("synthetic_TQQQ.csv")
    irx = download_csv("IRX.csv")

    q["date"] = pd.to_datetime(q["date"])
    s["date"] = pd.to_datetime(s["date"])
    irx["date"] = pd.to_datetime(irx["date"])

    q = q[["date", "adjclose"]].rename(columns={"adjclose": "qqq"})
    s["real_adjclose"] = pd.to_numeric(s["real_adjclose"], errors="coerce")
    s["synth_adjclose"] = pd.to_numeric(s["synth_adjclose"], errors="coerce")
    s["tqqq"] = s["real_adjclose"].fillna(s["synth_adjclose"])
    s["price_source"] = np.where(s["real_adjclose"].notna(), "actual", "synthetic")

    # IRX.csv in this repository may have yield_pct or close; support both.
    ycol = "yield_pct" if "yield_pct" in irx.columns else ("close" if "close" in irx.columns else irx.columns[-1])
    irx[ycol] = pd.to_numeric(irx[ycol], errors="coerce")
    irx = irx[["date", ycol]].rename(columns={ycol: "cash_yield_pct"})

    p = q.merge(s[["date", "tqqq", "price_source"]], on="date", how="inner")
    p = p.merge(irx, on="date", how="left").sort_values("date").reset_index(drop=True)
    p["cash_yield_pct"] = p["cash_yield_pct"].ffill().fillna(0.0).clip(lower=0.0, upper=25.0)
    p = p[p["date"] >= pd.Timestamp(START)].copy().reset_index(drop=True)

    # Features reused across the parameter grid.
    for ma in sorted({40, 60, 80, 100, 120, 150, 200, 250, 300}):
        p[f"ma{ma}"] = p["qqq"].rolling(ma, min_periods=ma).mean()
    p["hi252"] = p["qqq"].rolling(252, min_periods=252).max()
    p["dd252"] = p["qqq"] / p["hi252"] - 1.0
    for ma in (40, 60, 80):
        above = p["qqq"] > p[f"ma{ma}"]
        for days in (3, 5, 10):
            p[f"above_ma{ma}_{days}d"] = above.rolling(days, min_periods=days).sum().eq(days)
    return p


@dataclass(frozen=True)
class Guard:
    long_ma: int = 250
    slope_lb: int = 20
    guard_dd: float = -0.25
    rec1_ma: int = 60
    rec1_days: int = 5
    rec2_ma: int = 120

    @property
    def name(self) -> str:
        return (
            f"C_L{self.long_ma}_S{self.slope_lb}_DD{abs(int(round(self.guard_dd*100)))}_"
            f"R{self.rec1_ma}x{self.rec1_days}_{self.rec2_ma}"
        )


def calc_red(p: pd.DataFrame, g: Guard) -> np.ndarray:
    ma = p[f"ma{g.long_ma}"]
    return (
        (p["qqq"] < ma)
        & (ma < ma.shift(g.slope_lb))
        & (p["dd252"] <= g.guard_dd)
    ).fillna(False).to_numpy(dtype=bool)


def execute_target(cash: float, shares: float, avg_cost: float, px: float, target: float):
    equity = cash + shares * px
    target = float(np.clip(target, 0.0, 1.0))
    desired = target * equity
    current = shares * px
    delta = desired - current
    notional = 0.0
    fee = 0.0

    if delta > 1e-12:
        buy = min(delta, cash / (1.0 + ONE_WAY_COST))
        if buy > 0:
            qty = buy / px
            fee = buy * ONE_WAY_COST
            old_basis = shares * avg_cost if (shares > 0 and np.isfinite(avg_cost)) else 0.0
            shares += qty
            cash -= buy + fee
            avg_cost = (old_basis + buy + fee) / shares
            notional = buy
    elif delta < -1e-12:
        sell = min(-delta, current)
        if sell > 0:
            qty = sell / px
            fee = sell * ONE_WAY_COST
            shares -= qty
            cash += sell - fee
            notional = -sell
            if shares <= 1e-12:
                shares = 0.0
                avg_cost = np.nan

    return cash, shares, avg_cost, notional, fee


def simulate(p: pd.DataFrame, mode: str, guard: Guard | None = None, save_path: bool = False):
    q = p["qqq"].to_numpy(float)
    lev = p["tqqq"].to_numpy(float)
    dates = pd.DatetimeIndex(p["date"])
    cash_y = p["cash_yield_pct"].to_numpy(float)

    if guard is None:
        guard = Guard()
    red = calc_red(p, guard) if mode == "C" else np.zeros(len(p), dtype=bool)
    long_ma = p[f"ma{guard.long_ma}"].to_numpy(float)
    rec2 = p[f"ma{guard.rec2_ma}"].to_numpy(float)
    rec1ok = p[f"above_ma{guard.rec1_ma}_{guard.rec1_days}d"].to_numpy(bool)

    cash = 1.0
    shares = 0.0
    avg_cost = np.nan
    pending_target = None
    pending_reason = None

    cycle_peak = np.nan
    entry_started = False
    tier_used = [False] * len(ENTRY_DD)
    crash_mode = False
    b_below = False

    equity_arr = np.empty(len(p), dtype=float)
    weight_arr = np.empty(len(p), dtype=float)
    crash_arr = np.zeros(len(p), dtype=bool)
    trade_rows = []

    for i in range(len(p)):
        # Accrue historical T-bill-like return on cash between closes.
        if i > 0:
            annual = max(0.0, cash_y[i - 1] / 100.0)
            cash *= 1.0 + annual / 252.0

        # Execute prior close's signal at today's close. This deliberately uses t+1
        # close because the pre-2010 synthetic series is close-only.
        if pending_target is not None:
            before_shares = shares
            cash, shares, avg_cost, notional, fee = execute_target(
                cash, shares, avg_cost, lev[i], pending_target
            )
            if abs(notional) > 1e-12:
                trade_rows.append(
                    (dates[i], mode, pending_reason, pending_target, lev[i], notional, fee, cash, shares, avg_cost)
                )
            if before_shares > 0 and shares == 0 and not crash_mode:
                cycle_peak = q[i]
                entry_started = False
                tier_used = [False] * len(ENTRY_DD)

        pending_target = None
        pending_reason = None

        equity = cash + shares * lev[i]
        weight = shares * lev[i] / equity if equity > 0 else 0.0
        equity_arr[i] = equity
        weight_arr[i] = weight
        crash_arr[i] = crash_mode

        # Common take-profit: isolate the value of the regime guard in V1.
        if shares > 0 and np.isfinite(avg_cost) and lev[i] >= avg_cost * (1.0 + TP):
            pending_target = 0.0
            pending_reason = "TP_50"
            continue

        if mode == "B":
            # Hard long-MA safety filter. Use the V1 250-day MA regardless of C-grid.
            ma250 = p["ma250"].iat[i]
            if np.isfinite(ma250) and q[i] < ma250:
                b_below = True
                if shares > 0:
                    pending_target = 0.0
                    pending_reason = "below_MA250"
                continue
            if b_below and np.isfinite(ma250) and q[i] >= ma250:
                b_below = False
                cycle_peak = q[i]
                entry_started = False
                tier_used = [False] * len(ENTRY_DD)

        elif mode == "C":
            if (not crash_mode) and red[i]:
                crash_mode = True
                pending_target = 0.0
                pending_reason = "RED_enter"
                continue

            if crash_mode:
                if red[i]:
                    if shares > 0:
                        pending_target = 0.0
                        pending_reason = "RED_reassert"
                    continue

                if np.isfinite(long_ma[i]) and q[i] > long_ma[i]:
                    crash_mode = False
                    cycle_peak = q[i]
                    entry_started = False
                    tier_used = [False] * len(ENTRY_DD)
                    continue

                if np.isfinite(rec2[i]) and q[i] > rec2[i]:
                    if weight < 0.50 - 1e-6:
                        pending_target = 0.50
                        pending_reason = f"recovery_MA{guard.rec2_ma}"
                    continue

                if rec1ok[i]:
                    if weight < 0.25 - 1e-6:
                        pending_target = 0.25
                        pending_reason = f"recovery_MA{guard.rec1_ma}_{guard.rec1_days}d"
                    continue
                continue

        # Normal dip-buy logic. Peak is frozen after the first entry.
        if not entry_started:
            cycle_peak = q[i] if not np.isfinite(cycle_peak) else max(cycle_peak, q[i])

        dd = q[i] / cycle_peak - 1.0 if (np.isfinite(cycle_peak) and cycle_peak > 0) else 0.0
        fired = None
        for j, threshold in enumerate(ENTRY_DD):
            if dd <= threshold and not tier_used[j]:
                fired = j
        if fired is not None:
            for j in range(fired + 1):
                tier_used[j] = True
            entry_started = True
            target = TARGET_W[fired]
            if target > weight + 1e-6:
                pending_target = target
                pending_reason = f"dip_{ENTRY_DD[fired]:.0%}_to_{target:.0%}"

    eq = pd.Series(equity_arr, index=dates, name="equity")
    w = pd.Series(weight_arr, index=dates, name="weight")
    tr = pd.DataFrame(
        trade_rows,
        columns=["date", "mode", "reason", "target_weight", "price", "notional", "fee", "cash_after", "shares_after", "avg_cost_after"],
    )
    path = None
    if save_path:
        path = pd.DataFrame(
            {
                "date": dates,
                "qqq": q,
                "tqqq": lev,
                "equity": equity_arr,
                "weight": weight_arr,
                "red": red,
                "crash_mode": crash_arr,
            }
        )
    return eq, w, tr, path


def perf(eq: pd.Series) -> dict:
    eq = eq.dropna()
    if len(eq) < 2:
        return {"CAGR": np.nan, "MDD": np.nan, "Calmar": np.nan, "Final": np.nan, "MaxUnderwaterDays": np.nan}
    years = (eq.index[-1] - eq.index[0]).days / 365.2425
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0
    dd = eq / eq.cummax() - 1.0
    mdd = float(dd.min())

    peak = eq.cummax()
    uw = eq < peak * (1 - 1e-12)
    max_uw = 0
    start = None
    for d, flag in uw.items():
        if flag and start is None:
            start = d
        elif (not flag) and start is not None:
            max_uw = max(max_uw, (d - start).days)
            start = None
    if start is not None:
        max_uw = max(max_uw, (eq.index[-1] - start).days)

    return {
        "CAGR": cagr,
        "MDD": mdd,
        "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "Final": float(eq.iloc[-1] / eq.iloc[0]),
        "MaxUnderwaterDays": max_uw,
    }


def period_metrics(eq: pd.Series) -> dict:
    out = {}
    for name, (a, b) in PERIODS.items():
        x = eq.loc[a:b]
        if len(x) < 2:
            out[f"{name}_Return"] = np.nan
            out[f"{name}_MDD"] = np.nan
            continue
        out[f"{name}_Return"] = x.iloc[-1] / x.iloc[0] - 1.0
        out[f"{name}_MDD"] = float((x / x.cummax() - 1.0).min())
    return out


def summarize(name: str, eq: pd.Series, w: pd.Series, tr: pd.DataFrame, guard: Guard | None = None) -> dict:
    row = {"strategy": name}
    if guard is not None:
        row.update(
            {
                "long_ma": guard.long_ma,
                "slope_lb": guard.slope_lb,
                "guard_dd": guard.guard_dd,
                "rec1_ma": guard.rec1_ma,
                "rec1_days": guard.rec1_days,
                "rec2_ma": guard.rec2_ma,
            }
        )
    row.update(perf(eq))
    row.update(period_metrics(eq))
    row["InvestedDayPct"] = float((w > 0.01).mean())
    row["AvgWeight"] = float(w.mean())
    row["TradeCount"] = int(len(tr))
    return row


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    # Pareto on: maximize CAGR, maximize DotCom_MDD (less negative), maximize Actual-era CAGR proxy.
    # Actual_TQQQ_Era_Return is total return, so compare via full actual-era MDD plus CAGR is handled separately below.
    x = df.copy().reset_index(drop=True)
    vals = x[["CAGR", "DotCom_MDD", "MDD"]].to_numpy(float)
    keep = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        if not keep[i]:
            continue
        dominates_i = (
            (vals[:, 0] >= vals[i, 0])
            & (vals[:, 1] >= vals[i, 1])
            & (vals[:, 2] >= vals[i, 2])
            & (
                (vals[:, 0] > vals[i, 0])
                | (vals[:, 1] > vals[i, 1])
                | (vals[:, 2] > vals[i, 2])
            )
        )
        if dominates_i.any():
            keep[i] = False
    return x.loc[keep].sort_values(["CAGR", "DotCom_MDD"], ascending=[False, False])


def main():
    p = load_panel()
    print(f"DATA rows={len(p)} start={p.date.iloc[0].date()} end={p.date.iloc[-1].date()}")
    actual_start = p.loc[p["price_source"].eq("actual"), "date"].min()
    print(f"Actual TQQQ begins in hybrid series: {actual_start.date() if pd.notna(actual_start) else 'NA'}")

    # Baselines.
    rows = []
    base_paths = {}
    base_trades = []
    for mode, name, g in [
        ("A", "A_aggressive", None),
        ("B", "B_hard_MA250", None),
        ("C", "C_V1", Guard()),
    ]:
        eq, w, tr, path = simulate(p, mode, g, save_path=True)
        rows.append(summarize(name, eq, w, tr, g if mode == "C" else None))
        path = path.rename(columns={"equity": name + "_equity", "weight": name + "_weight"})
        base_paths[name] = path[["date", name + "_equity", name + "_weight"]]
        if not tr.empty:
            tr = tr.copy(); tr["strategy"] = name; base_trades.append(tr)

    baseline = pd.DataFrame(rows)
    baseline.to_csv(OUT / "baselines.csv", index=False)
    merged = p[["date", "qqq", "tqqq", "price_source"]].copy()
    for z in base_paths.values():
        merged = merged.merge(z, on="date", how="left")
    merged.to_csv(OUT / "baseline_paths.csv", index=False)
    pd.concat(base_trades, ignore_index=True).to_csv(OUT / "baseline_trades.csv", index=False)

    print("\n=== BASELINES ===")
    show = ["strategy", "CAGR", "MDD", "DotCom_MDD", "GFC_MDD", "COVID_MDD", "2022_Bear_MDD", "InvestedDayPct", "AvgWeight", "TradeCount"]
    print(baseline[show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Selective crash-guard robustness grid. Entries and +50% TP stay fixed.
    sweep_rows = []
    total = 0
    for long_ma in (200, 250, 300):
        for slope_lb in (20, 40, 60):
            for guard_dd in (-0.20, -0.25, -0.30, -0.35):
                for rec1_ma in (40, 60, 80):
                    for rec1_days in (3, 5, 10):
                        for rec2_ma in (100, 120, 150):
                            if rec2_ma <= rec1_ma:
                                continue
                            g = Guard(long_ma, slope_lb, guard_dd, rec1_ma, rec1_days, rec2_ma)
                            eq, w, tr, _ = simulate(p, "C", g, save_path=False)
                            sweep_rows.append(summarize(g.name, eq, w, tr, g))
                            total += 1
    sweep = pd.DataFrame(sweep_rows)

    # Add actual-TQQQ-era CAGR/MDD calculated from each strategy equity path only for top-level validation.
    # Re-simulation is acceptable and keeps the row schema simple.
    actual_metrics = []
    actual_date = pd.Timestamp(actual_start)
    for r in sweep.itertuples(index=False):
        g = Guard(int(r.long_ma), int(r.slope_lb), float(r.guard_dd), int(r.rec1_ma), int(r.rec1_days), int(r.rec2_ma))
        eq, _, _, _ = simulate(p, "C", g, save_path=False)
        z = eq.loc[actual_date:]
        m = perf(z)
        actual_metrics.append((m["CAGR"], m["MDD"]))
    sweep["ActualEra_CAGR"] = [x[0] for x in actual_metrics]
    sweep["ActualEra_MDD"] = [x[1] for x in actual_metrics]

    sweep = sweep.sort_values("CAGR", ascending=False).reset_index(drop=True)
    sweep.to_csv(OUT / "guard_sweep.csv", index=False)
    pf = pareto_front(sweep)
    pf.to_csv(OUT / "guard_pareto.csv", index=False)

    # Risk-tier winners: highest CAGR satisfying progressively tighter dot-com damage caps.
    tier_rows = []
    for cap in (-0.70, -0.65, -0.60, -0.55, -0.50):
        elig = sweep[(sweep["DotCom_MDD"] >= cap) & (sweep["MDD"] >= min(cap - 0.05, -0.55))]
        if not elig.empty:
            z = elig.sort_values(["CAGR", "ActualEra_CAGR"], ascending=False).iloc[0].copy()
            z["risk_tier_dotcom_cap"] = cap
            tier_rows.append(z)
    tiers = pd.DataFrame(tier_rows)
    tiers.to_csv(OUT / "risk_tier_winners.csv", index=False)

    print(f"\nGrid tested: {total} selective-guard configurations")
    print("\n=== TOP 15 BY CAGR ===")
    cols = ["strategy", "CAGR", "MDD", "DotCom_MDD", "GFC_MDD", "ActualEra_CAGR", "ActualEra_MDD", "InvestedDayPct", "TradeCount"]
    print(sweep[cols].head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== RISK-TIER WINNERS ===")
    if not tiers.empty:
        print(tiers[["risk_tier_dotcom_cap"] + cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        print("No configurations met tier constraints.")
    print("\n=== PARETO FRONT (first 20) ===")
    print(pf[cols].head(20).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Save a small machine-readable recommendation candidate set for V2.
    cand = pd.concat([sweep.head(10), tiers.drop(columns=["risk_tier_dotcom_cap"], errors="ignore"), pf.head(10)], ignore_index=True)
    cand = cand.drop_duplicates("strategy").reset_index(drop=True)
    cand.to_csv(OUT / "v2_candidates.csv", index=False)


if __name__ == "__main__":
    main()
