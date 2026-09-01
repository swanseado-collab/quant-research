from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from backtest_v1 import load_panel, perf, period_metrics
from backtest_v2 import add_features
from backtest_v6_bull_hold import Cfg
from backtest_v13_validation import RunSpec, ensure_features, simulate_spec
from backtest_v14_live_extension import yf_series, normalized_overlap_stats, END

OUT = Path(__file__).resolve().parent / "results_v15_cycle_synth_stress"
OUT.mkdir(parents=True, exist_ok=True)
BASE = Cfg(-0.25, 80, 110, 0.35, 110, 50, -0.18)


def build_live_panel():
    frozen = load_panel().copy()
    frozen["date"] = pd.to_datetime(frozen.date)
    last = pd.Timestamp(frozen.date.max())
    yq = yf_series("QQQ", end=END)
    yt = yf_series("TQQQ", end=END)
    fq = frozen.set_index("date").qqq
    ft = frozen.set_index("date").tqqq
    qscale = normalized_overlap_stats(fq, yq, last)["scale_factor"]
    tscale = normalized_overlap_stats(ft, yt, last)["scale_factor"]
    fresh = pd.concat([(yq*qscale).rename("qqq"), (yt*tscale).rename("tqqq")], axis=1).dropna()
    fresh = fresh[fresh.index > last].copy()
    fresh["cash_yield_pct"] = float(frozen.cash_yield_pct.iloc[-1])
    fresh["price_source"] = "actual_live"
    extra = fresh.reset_index(); extra = extra.rename(columns={extra.columns[0]: "date"})
    cols = ["date", "qqq", "tqqq", "price_source", "cash_yield_pct"]
    p = pd.concat([frozen[cols], extra[cols]], ignore_index=True).sort_values("date").drop_duplicates("date", keep="last")
    return ensure_features(add_features(p.reset_index(drop=True)))


def cycle_attribution(p, eq, w, tr):
    path_dates = eq.index
    rows = []
    entries = tr[tr.reason == "REVERSAL_ENTRY"].reset_index(drop=True)
    exits = tr[tr.reason == "BULL_EXIT"].reset_index(drop=True)
    total_log = np.log(eq.iloc[-1] / eq.iloc[0])
    for ci, er in entries.iterrows():
        start = pd.Timestamp(er.date)
        later_exits = exits[exits.date > start]
        end = pd.Timestamp(later_exits.iloc[0].date) if len(later_exits) else pd.Timestamp(eq.index[-1])
        ongoing = len(later_exits) == 0
        loc = path_dates.get_loc(start)
        base_date = path_dates[max(0, loc - 1)]
        start_eq = float(eq.loc[base_date])
        end_eq = float(eq.loc[end])
        z = eq.loc[base_date:end]
        wz = w.loc[base_date:end]
        peak_eq = float(z.max())
        peak_date = pd.Timestamp(z.idxmax())
        dd = z / z.cummax() - 1
        cycle_mdd = float(dd.min())
        final_mult = end_eq / start_eq
        peak_mult = peak_eq / start_eq
        giveback = 1 - final_mult / peak_mult if peak_mult > 0 else np.nan
        fulls = tr[(tr.reason == "BULL_FULL") & (tr.date >= start) & (tr.date <= end)]
        full_date = pd.Timestamp(fulls.iloc[0].date) if len(fulls) else pd.NaT
        log_growth = np.log(final_mult)
        rows.append({
            "cycle": ci + 1,
            "entry_date": start,
            "full_date": full_date,
            "exit_or_latest": end,
            "ongoing": ongoing,
            "calendar_days": (end - start).days,
            "entry_tqqq": float(er.price),
            "start_equity": start_eq,
            "end_equity": end_eq,
            "portfolio_multiple": final_mult,
            "peak_portfolio_multiple": peak_mult,
            "peak_equity_date": peak_date,
            "cycle_mdd": cycle_mdd,
            "giveback_from_cycle_peak": giveback,
            "max_weight": float(wz.max()),
            "log_growth": log_growth,
            "share_total_log_growth": log_growth / total_log if total_log != 0 else np.nan,
        })
    df = pd.DataFrame(rows)
    cash_log_residual = total_log - df.log_growth.sum() if len(df) else total_log
    return df, total_log, cash_log_residual


def backcast_synthetic(base: pd.DataFrame, leverage: float, annual_drag: float):
    p = base.copy()
    q = p.qqq.to_numpy(float)
    y = p.cash_yield_pct.to_numpy(float)
    actual_mask = p.price_source.eq("actual").to_numpy(bool)
    first_actual = int(np.flatnonzero(actual_mask)[0])
    t = p.tqqq.to_numpy(float).copy()
    qr = np.empty(len(p)); qr[:] = np.nan
    qr[1:] = q[1:] / q[:-1] - 1.0
    # Modeled daily return uses daily target leverage, financing on borrowed (L-1)x capital,
    # plus an explicit annual all-in drag for expenses/spread/tracking shortfall.
    mr = np.zeros(len(p))
    for i in range(1, first_actual + 1):
        financing = max(0.0, y[i-1] / 100.0) * max(0.0, leverage - 1.0) / 252.0
        mr[i] = leverage * qr[i] - financing - annual_drag / 252.0
        mr[i] = max(mr[i], -0.99)
    # Anchor exactly at the first actual TQQQ close and backcast, avoiding arbitrary level differences.
    for i in range(first_actual, 0, -1):
        t[i-1] = t[i] / (1.0 + mr[i])
    p["tqqq"] = t
    p.loc[p.index < first_actual, "price_source"] = f"stress_L{leverage:.1f}_D{annual_drag:.3f}"
    return p


def main():
    # A) Cycle attribution on the latest panel through the most recent completed session.
    live = build_live_panel()
    eq, w, tr, path = simulate_spec(live, RunSpec(BASE), save_path=True)
    cycles, total_log, cash_log = cycle_attribution(live, eq, w, tr)
    cycles.to_csv(OUT / "v15_cycle_attribution.csv", index=False)
    pd.DataFrame([{
        "start": eq.index[0], "end": eq.index[-1], "final_multiple": float(eq.iloc[-1]/eq.iloc[0]),
        "total_log_growth": total_log, "sum_cycle_log_growth": float(cycles.log_growth.sum()),
        "cash_wait_log_growth_residual": cash_log,
        "top2_cycle_share_of_total_log_growth": float(cycles.sort_values("log_growth", ascending=False).head(2).log_growth.sum()/total_log),
        "top3_cycle_share_of_total_log_growth": float(cycles.sort_values("log_growth", ascending=False).head(3).log_growth.sum()/total_log),
    }]).to_csv(OUT / "v15_cycle_concentration.csv", index=False)

    # B) Pre-2010 synthetic stress. Signals stay QQQ-based and unchanged; only hypothetical 3x path assumptions vary.
    frozen = ensure_features(add_features(load_panel()))
    stress_rows = []
    for lev in (2.8, 3.0, 3.2):
        for drag in (0.008, 0.015, 0.025, 0.040):
            ps = backcast_synthetic(frozen, lev, drag)
            e, ww, tt, _ = simulate_spec(ps, RunSpec(BASE))
            r = {"leverage": lev, "annual_drag": drag, **perf(e), **period_metrics(e)}
            actual_start = ps.loc[ps.price_source.eq("actual"), "date"].min()
            am = perf(e.loc[actual_start:])
            r["ActualEra_CAGR"] = am["CAGR"]; r["ActualEra_MDD"] = am["MDD"]
            r["TradeCount"] = len(tt)
            stress_rows.append(r)
    stress = pd.DataFrame(stress_rows)
    stress.to_csv(OUT / "v15_synthetic_stress.csv", index=False)

    print("=== CYCLE ATTRIBUTION ===")
    cols = ["cycle","entry_date","full_date","exit_or_latest","ongoing","portfolio_multiple","peak_portfolio_multiple","cycle_mdd","giveback_from_cycle_peak","share_total_log_growth"]
    print(cycles[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== SYNTHETIC STRESS ===")
    scols = ["leverage","annual_drag","CAGR","MDD","DotCom_MDD","GFC_MDD","ActualEra_CAGR","ActualEra_MDD"]
    print(stress[scols].sort_values(["leverage","annual_drag"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
