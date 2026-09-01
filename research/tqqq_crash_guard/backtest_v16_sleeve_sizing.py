from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from backtest_v1 import perf, period_metrics
from backtest_v6_bull_hold import Cfg
from backtest_v13_validation import RunSpec, simulate_spec
from backtest_v15_cycle_and_synth_stress import build_live_panel

OUT = Path(__file__).resolve().parent / "results_v16_sleeve_sizing"
OUT.mkdir(parents=True, exist_ok=True)
BASE = Cfg(-0.25, 80, 110, 0.35, 110, 50, -0.18)
COST = 0.0005


def buy_with_cost(cash, shares, px, buy_notional):
    buy = min(max(0.0, buy_notional), cash / (1 + COST))
    if buy <= 0:
        return cash, shares, 0.0
    fee = buy * COST
    shares += buy / px
    cash -= buy + fee
    return cash, shares, fee


def sell_with_cost(shares, px, sell_notional):
    current = shares * px
    sell = min(max(0.0, sell_notional), current)
    if sell <= 0:
        return shares, 0.0, 0.0
    fee = sell * COST
    shares -= sell / px
    if shares < 1e-12:
        shares = 0.0
    return shares, sell - fee, fee


def simulate_portfolio(p, base_tr, budget_pct: float, harvest_cap: float | None):
    dates = pd.DatetimeIndex(p.date)
    px = p.tqqq.to_numpy(float)
    cy = p.cash_yield_pct.to_numpy(float)
    trade_map = {pd.Timestamp(r.date): r for _, r in base_tr.iterrows()}

    core_cash = 1.0
    tact_cash = 0.0
    shares = 0.0
    in_campaign = False
    pending_cap_trim = False
    rows = []
    trades = []

    for i, d in enumerate(dates):
        if i > 0:
            rate = max(0.0, cy[i-1] / 100.0) / 252.0
            core_cash *= 1.0 + rate
            tact_cash *= 1.0 + rate

        # Base V7 trades are already t+1-close executions. Apply them first.
        if d in trade_map:
            r = trade_map[d]
            reason = str(r.reason)
            target = float(r.target_weight)

            if reason == "REVERSAL_ENTRY":
                total_before = core_cash + tact_cash + shares * px[i]
                if not in_campaign:
                    budget = min(core_cash, budget_pct * total_before)
                    core_cash -= budget
                    tact_cash += budget
                    in_campaign = True
                camp_eq = tact_cash + shares * px[i]
                desired = target * camp_eq
                current = shares * px[i]
                if desired > current:
                    tact_cash, shares, fee = buy_with_cost(tact_cash, shares, px[i], desired-current)
                    trades.append((d, reason, total_before, target, fee))

            elif reason == "BULL_FULL" and in_campaign:
                total_before = core_cash + tact_cash + shares * px[i]
                camp_eq = tact_cash + shares * px[i]
                desired = target * camp_eq
                current = shares * px[i]
                if desired > current:
                    tact_cash, shares, fee = buy_with_cost(tact_cash, shares, px[i], desired-current)
                    trades.append((d, reason, total_before, target, fee))

            elif reason == "BULL_EXIT" and in_campaign:
                total_before = core_cash + tact_cash + shares * px[i]
                shares, proceeds, fee = sell_with_cost(shares, px[i], shares*px[i])
                tact_cash += proceeds
                core_cash += tact_cash
                tact_cash = 0.0
                in_campaign = False
                pending_cap_trim = False
                trades.append((d, reason, total_before, 0.0, fee))

        # Risk-overlay harvest, scheduled from prior close. It never causes re-entry.
        if pending_cap_trim and in_campaign and shares > 0 and harvest_cap is not None:
            total_now = core_cash + tact_cash + shares * px[i]
            desired_tqqq = harvest_cap * total_now
            current_tqqq = shares * px[i]
            if current_tqqq > desired_tqqq:
                shares, proceeds, fee = sell_with_cost(shares, px[i], current_tqqq - desired_tqqq)
                core_cash += proceeds
                trades.append((d, "HARVEST_CAP", total_now, harvest_cap, fee))
            pending_cap_trim = False

        total = core_cash + tact_cash + shares * px[i]
        tqqq_value = shares * px[i]
        tqqq_w = tqqq_value / total if total > 0 else 0.0
        rows.append((d, total, tqqq_w, core_cash, tact_cash, tqqq_value, in_campaign))

        if harvest_cap is not None and in_campaign and tqqq_w > harvest_cap + 1e-9:
            pending_cap_trim = True

    path = pd.DataFrame(rows, columns=["date","equity","tqqq_weight","core_cash","tactical_cash","tqqq_value","in_campaign"])
    eq = pd.Series(path.equity.to_numpy(), index=pd.DatetimeIndex(path.date))
    tr = pd.DataFrame(trades, columns=["date","reason","total_before","target_or_cap","fee"])
    return eq, path, tr


def window_metrics(eq, start):
    x = eq.loc[pd.Timestamp(start):]
    return perf(x) if len(x) >= 2 else {"CAGR":np.nan,"MDD":np.nan,"Final":np.nan,"Calmar":np.nan,"MaxUnderwaterDays":np.nan}


def main():
    p = build_live_panel()
    # Get V7's fixed trade dates from the same extended panel.
    _, _, base_tr, _ = simulate_spec(p, RunSpec(BASE))
    actual_start = p.loc[p.price_source.eq("actual"), "date"].min()

    rows = []
    paths_to_save = []
    for budget in (0.05, 0.10, 0.15, 0.20, 0.25):
        for cap in (None, 0.25, 0.40, 0.60):
            # Absolute harvest cap cannot reasonably be below a campaign's intended budget only after appreciation;
            # still allow budget=25%, cap=25% as a strict total-portfolio ceiling.
            eq, path, tr = simulate_portfolio(p, base_tr, budget, cap)
            r = {
                "budget_pct": budget,
                "harvest_cap": "none" if cap is None else cap,
                **perf(eq), **period_metrics(eq),
                "Max_TQQQ_Weight": float(path.tqqq_weight.max()),
                "Avg_TQQQ_Weight": float(path.tqqq_weight.mean()),
                "Days_TQQQ_Over25": int((path.tqqq_weight > .25).sum()),
                "Days_TQQQ_Over50": int((path.tqqq_weight > .50).sum()),
                "HarvestCount": int((tr.reason == "HARVEST_CAP").sum()) if len(tr) else 0,
                "TotalTradeEvents": len(tr),
            }
            am = window_metrics(eq, actual_start)
            r["ActualEra_CAGR"] = am["CAGR"]; r["ActualEra_MDD"] = am["MDD"]
            r["FinalMultiple"] = float(eq.iloc[-1]/eq.iloc[0])
            rows.append(r)
            if (budget in (0.10,0.15,0.20)) and (cap in (None,0.40)):
                paths_to_save.append((budget, cap, path, tr))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "v16_sleeve_grid.csv", index=False)

    # Pareto-like display: maximize CAGR while preferring smaller MDD and lower concentration.
    z = df.copy()
    z["score"] = 2.0*z.CAGR + 0.8*z.ActualEra_CAGR + 0.9*z.MDD - 0.25*z.Max_TQQQ_Weight
    z = z.sort_values("score", ascending=False)
    z.to_csv(OUT / "v16_ranked.csv", index=False)

    for budget, cap, path, tr in paths_to_save:
        tag = f"B{int(budget*100)}_C{'NONE' if cap is None else int(cap*100)}"
        path.to_csv(OUT / f"path_{tag}.csv", index=False)
        tr.to_csv(OUT / f"trades_{tag}.csv", index=False)

    print("=== V16 SLEEVE GRID ===")
    cols = ["budget_pct","harvest_cap","CAGR","MDD","ActualEra_CAGR","ActualEra_MDD","Max_TQQQ_Weight","Avg_TQQQ_Weight","HarvestCount","FinalMultiple"]
    print(df[cols].sort_values(["budget_pct","harvest_cap"], key=lambda s: s.astype(str)).to_string(index=False, float_format=lambda x:f"{x:.4f}"))
    print("\n=== V16 RANKED ===")
    print(z[cols+["score"]].head(20).to_string(index=False, float_format=lambda x:f"{x:.4f}"))


if __name__ == "__main__":
    main()
