from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from backtest_v1 import (
    load_panel, execute_target, perf, period_metrics,
    ENTRY_DD, TARGET_W, TP
)

OUT = Path(__file__).resolve().parent / "results_v2"
OUT.mkdir(parents=True, exist_ok=True)


class V2:
    def __init__(self, trend_ma, slope_lb, yellow_cap, red_dd, red_target,
                 rec_fast_ma, rec_fast_days, rec_mid_ma, release_ma):
        self.trend_ma = int(trend_ma)
        self.slope_lb = int(slope_lb)
        self.yellow_cap = float(yellow_cap)
        self.red_dd = float(red_dd)
        self.red_target = float(red_target)
        self.rec_fast_ma = int(rec_fast_ma)
        self.rec_fast_days = int(rec_fast_days)
        self.rec_mid_ma = int(rec_mid_ma)
        self.release_ma = int(release_ma)

    @property
    def name(self):
        return (
            f"V2_M{self.trend_ma}_S{self.slope_lb}_YC{int(self.yellow_cap*100)}_"
            f"RD{abs(int(self.red_dd*100))}_RT{int(self.red_target*100)}_"
            f"RF{self.rec_fast_ma}x{self.rec_fast_days}_RM{self.rec_mid_ma}_RL{self.release_ma}"
        )


def add_features(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    for ma in (120, 150, 180, 200, 220, 250):
        if f"ma{ma}" not in p:
            p[f"ma{ma}"] = p["qqq"].rolling(ma, min_periods=ma).mean()
    for ma in (40, 60, 80, 100):
        if f"ma{ma}" not in p:
            p[f"ma{ma}"] = p["qqq"].rolling(ma, min_periods=ma).mean()
        above = p["qqq"] > p[f"ma{ma}"]
        for d in (3, 5, 10):
            col = f"above_ma{ma}_{d}d"
            if col not in p:
                p[col] = above.rolling(d, min_periods=d).sum().eq(d)
    return p


def simulate_v2(p: pd.DataFrame, cfg: V2, save_path=False):
    q = p["qqq"].to_numpy(float)
    lev = p["tqqq"].to_numpy(float)
    dates = pd.DatetimeIndex(p["date"])
    cash_y = p["cash_yield_pct"].to_numpy(float)

    trend = p[f"ma{cfg.trend_ma}"].to_numpy(float)
    release = p[f"ma{cfg.release_ma}"].to_numpy(float)
    rec_mid = p[f"ma{cfg.rec_mid_ma}"].to_numpy(float)
    rec_fast_ok = p[f"above_ma{cfg.rec_fast_ma}_{cfg.rec_fast_days}d"].to_numpy(bool)
    dd252 = p["dd252"].to_numpy(float)

    cash = 1.0
    shares = 0.0
    avg_cost = np.nan
    pending_target = None
    pending_reason = None

    cycle_peak = np.nan
    entry_started = False
    tier_used = [False] * len(ENTRY_DD)
    red_mode = False

    equity_arr = np.empty(len(p), float)
    weight_arr = np.empty(len(p), float)
    state_arr = np.empty(len(p), object)
    trades = []

    for i in range(len(p)):
        if i > 0:
            annual = max(0.0, cash_y[i - 1] / 100.0)
            cash *= 1.0 + annual / 252.0

        if pending_target is not None:
            cash, shares, avg_cost, notional, fee = execute_target(
                cash, shares, avg_cost, lev[i], pending_target
            )
            if abs(notional) > 1e-12:
                trades.append((dates[i], pending_reason, pending_target, lev[i], notional, fee, cash, shares, avg_cost))
        pending_target = None
        pending_reason = None

        equity = cash + shares * lev[i]
        weight = shares * lev[i] / equity if equity > 0 else 0.0
        equity_arr[i] = equity
        weight_arr[i] = weight

        trend_down = False
        if i >= cfg.slope_lb and np.isfinite(trend[i]) and np.isfinite(trend[i - cfg.slope_lb]):
            trend_down = (q[i] < trend[i]) and (trend[i] < trend[i - cfg.slope_lb])
        yellow = trend_down

        # RED: deep correction inside a confirmed downtrend.
        red_signal = yellow and np.isfinite(dd252[i]) and dd252[i] <= cfg.red_dd
        if red_signal:
            red_mode = True

        if red_mode:
            state_arr[i] = "RED"
            # On the first/renewed red signal, cut exposure promptly.
            if red_signal and weight > cfg.red_target + 1e-6:
                pending_target = cfg.red_target
                pending_reason = "RED_cut"
                continue

            # Recovery ladder while still below long release MA.
            if np.isfinite(release[i]) and q[i] > release[i]:
                red_mode = False
                cycle_peak = q[i]
                entry_started = False
                tier_used = [False] * len(ENTRY_DD)
                continue

            if np.isfinite(rec_mid[i]) and q[i] > rec_mid[i]:
                target = min(0.60, cfg.yellow_cap if yellow else 0.60)
                if weight < target - 1e-6:
                    pending_target = target
                    pending_reason = "RED_recovery_mid"
                continue

            if rec_fast_ok[i]:
                target = min(0.25, cfg.yellow_cap if yellow else 0.25)
                if weight < target - 1e-6:
                    pending_target = target
                    pending_reason = "RED_recovery_fast"
                continue
            continue

        state_arr[i] = "YELLOW" if yellow else "GREEN"

        # Common take profit.
        if shares > 0 and np.isfinite(avg_cost) and lev[i] >= avg_cost * (1.0 + TP):
            pending_target = 0.0
            pending_reason = "TP_50"
            continue

        # The key V2 change: if trend is deteriorating, cap exposure BEFORE deep dip buys.
        if yellow and weight > cfg.yellow_cap + 1e-6:
            pending_target = cfg.yellow_cap
            pending_reason = "YELLOW_cap"
            continue

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
            if yellow:
                target = min(target, cfg.yellow_cap)
            if target > weight + 1e-6:
                pending_target = target
                pending_reason = f"dip_{ENTRY_DD[fired]:.0%}_to_{target:.0%}"

    eq = pd.Series(equity_arr, index=dates)
    w = pd.Series(weight_arr, index=dates)
    tr = pd.DataFrame(trades, columns=["date","reason","target_weight","price","notional","fee","cash_after","shares_after","avg_cost_after"])
    path = None
    if save_path:
        path = pd.DataFrame({"date": dates, "qqq": q, "tqqq": lev, "equity": equity_arr, "weight": weight_arr, "state": state_arr})
    return eq, w, tr, path


def summarize(cfg: V2, eq, w, tr):
    row = {
        "strategy": cfg.name,
        "trend_ma": cfg.trend_ma,
        "slope_lb": cfg.slope_lb,
        "yellow_cap": cfg.yellow_cap,
        "red_dd": cfg.red_dd,
        "red_target": cfg.red_target,
        "rec_fast_ma": cfg.rec_fast_ma,
        "rec_fast_days": cfg.rec_fast_days,
        "rec_mid_ma": cfg.rec_mid_ma,
        "release_ma": cfg.release_ma,
    }
    row.update(perf(eq))
    row.update(period_metrics(eq))
    row["InvestedDayPct"] = float((w > 0.01).mean())
    row["AvgWeight"] = float(w.mean())
    row["TradeCount"] = int(len(tr))
    return row


def main():
    p = add_features(load_panel())
    actual_start = p.loc[p["price_source"].eq("actual"), "date"].min()

    rows = []
    configs = []
    # 3*2*4*3*2*2*2*2*2 = 2304 before validity filters.
    # Keep recovery/release logical and focused around robust neighborhoods.
    for trend_ma in (150, 180, 200):
        for slope_lb in (20, 40):
            for yellow_cap in (0.15, 0.25, 0.35, 0.50):
                for red_dd in (-0.15, -0.20, -0.25):
                    for red_target in (0.0, 0.15):
                        for rec_fast_ma in (40, 60):
                            for rec_fast_days in (3, 5):
                                for rec_mid_ma in (100, 120):
                                    for release_ma in (180, 200):
                                        if release_ma < rec_mid_ma or release_ma < trend_ma - 20:
                                            continue
                                        configs.append(V2(trend_ma, slope_lb, yellow_cap, red_dd, red_target,
                                                          rec_fast_ma, rec_fast_days, rec_mid_ma, release_ma))

    for n, cfg in enumerate(configs, 1):
        eq, w, tr, _ = simulate_v2(p, cfg)
        row = summarize(cfg, eq, w, tr)
        z = eq.loc[actual_start:]
        am = perf(z)
        row["ActualEra_CAGR"] = am["CAGR"]
        row["ActualEra_MDD"] = am["MDD"]
        rows.append(row)
        if n % 500 == 0:
            print("tested", n, "/", len(configs))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "v2_sweep.csv", index=False)

    # Robust objective: favor CAGR, but impose crash and actual-era floors.
    # Produce frontiers rather than one opaque score.
    tiers = []
    for cap in (-0.65, -0.60, -0.55, -0.50, -0.45, -0.40):
        elig = df[(df["DotCom_MDD"] >= cap) & (df["MDD"] >= cap - 0.03)]
        if len(elig):
            # Require the strategy not to collapse in the actual TQQQ era.
            elig2 = elig[elig["ActualEra_CAGR"] >= 0.18]
            if len(elig2):
                elig = elig2
            best = elig.sort_values(["CAGR","ActualEra_CAGR","GFC_MDD"], ascending=[False,False,False]).iloc[0].copy()
            best["dotcom_cap"] = cap
            tiers.append(best)
    tiers = pd.DataFrame(tiers)
    tiers.to_csv(OUT / "v2_risk_tiers.csv", index=False)

    # Stability: parameter neighborhoods around top candidates.
    # Score uses broad objectives, not just full-period CAGR.
    z = df.copy()
    z["score"] = (
        4.0*z["CAGR"] + 2.0*z["ActualEra_CAGR"]
        + 1.2*z["DotCom_MDD"] + 0.6*z["GFC_MDD"] + 0.3*z["2022_Bear_MDD"]
    )
    z = z.sort_values("score", ascending=False)
    z.head(100).to_csv(OUT / "v2_top100_score.csv", index=False)

    # Save detailed paths/trades for unique tier winners plus best score/CAGR.
    selected = []
    if len(tiers): selected += list(tiers["strategy"])
    selected += list(df.sort_values("CAGR", ascending=False).head(3)["strategy"])
    selected += list(z.head(3)["strategy"])
    selected = list(dict.fromkeys(selected))

    for name in selected:
        r = df.loc[df.strategy.eq(name)].iloc[0]
        cfg = V2(r.trend_ma, r.slope_lb, r.yellow_cap, r.red_dd, r.red_target,
                 r.rec_fast_ma, r.rec_fast_days, r.rec_mid_ma, r.release_ma)
        eq, w, tr, path = simulate_v2(p, cfg, save_path=True)
        safe = name.replace("%", "pct")
        path.to_csv(OUT / f"path_{safe}.csv", index=False)
        tr.to_csv(OUT / f"trades_{safe}.csv", index=False)

    cols = ["strategy","CAGR","MDD","DotCom_MDD","GFC_MDD","COVID_MDD","2022_Bear_MDD","ActualEra_CAGR","ActualEra_MDD","InvestedDayPct","AvgWeight","TradeCount"]
    print("\n=== V2 TOP CAGR ===")
    print(df.sort_values("CAGR", ascending=False)[cols].head(15).to_string(index=False, float_format=lambda x:f"{x:.4f}"))
    print("\n=== V2 RISK TIERS ===")
    if len(tiers):
        print(tiers[["dotcom_cap"]+cols].to_string(index=False, float_format=lambda x:f"{x:.4f}"))
    print("\n=== V2 TOP ROBUST SCORE ===")
    print(z[cols+['score']].head(15).to_string(index=False, float_format=lambda x:f"{x:.4f}"))


if __name__ == "__main__":
    main()
