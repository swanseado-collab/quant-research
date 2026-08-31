# TQQQ Panic-Buy Tactical Sleeve — 2026-09-01

## Research question
Can a KRW investor hold tactical capital in KOFR, buy TQQQ only after deep drawdowns, and sell the rebound, instead of chasing an old QQQ trend?

## Method integrity
- Actual TQQQ used from 2010-02-11 onward.
- Synthetic 3x QQQ used only to stress pre-launch dot-com and GFC periods.
- Grid: 1,200 rules. Drawdown reference 60/120/252-day high; first trigger 10/15/20/25/30%; add spacing 5/10%; 1-4 tranches; take-profit 10/15/20/30/40%; QQQ MA250_C3 regime filter on/off.
- Signals use completed closes and transact at the next open; take-profit uses an intraday standing limit.
- Economic screen assumed 10 bp each side. Accelerated and exact event engines matched on selected rules.
- Final actual-data rule selection used 2010-2015 Train + 2016-2020 Validation, with synthetic 2000-2009 as a stress filter. Actual 2021-2026 was untouched OOS.

## What failed
The high-return unrestricted variants were not robust.
- Actual-selected unrestricted: 120-day high, buys -10/-20/-30/-40%, four tranches, +40% TP.
- Actual Train CAGR 37.53%, Validation 40.81%, OOS 28.33%.
- But OOS MDD was -75.20%, max OOS hold 1,098 days.
- Synthetic pre-2010 CAGR -43.33%, MDD -99.94%; dot-com MDD about -99.92%.
Therefore this is essentially leveraged falling-knife averaging and is rejected.

A seemingly safer no-regime rule also failed in actual OOS:
- 120-day high; buy -30% and -35%; two equal tranches; +15% TP.
- Actual Train CAGR 10.27%, MDD -12.52%.
- Validation CAGR 13.54%, MDD -11.48%.
- OOS CAGR 12.83%, but MDD -74.82%, maximum hold 846 days.
Despite every OOS trade eventually finishing positive, the path risk is unacceptable as a standalone low-risk tactical strategy.

## Low-risk survivor
The only stable low-drawdown family required a QQQ regime filter:
- QQQ MA250_C3 must be ON.
- TQQQ drawdown measured from its prior 60-day high.
- Buy at -25%, -30%, -35%, three equal tranches.
- Exit all at average cost +10% or exit at next open if QQQ regime turns OFF.

Actual TQQQ results with cash yield set to zero:
- Train 2010-2015: CAGR 1.41%, MDD -7.97%, 4 trades, 75% win rate, worst trade -13.86%.
- Validation 2016-2020: CAGR 4.40%, MDD -10.42%, 5 trades, 80% win rate, worst trade -11.76%.
- OOS 2021-2026: CAGR 1.70%, MDD -10.85%, 7 trades, 57.1% win rate, worst trade -14.97%, median hold 6 days, max hold 18 days, about 2.8% time in market.
- Synthetic pre-2010 stress: CAGR 4.56%, MDD -9.08%; dot-com CAGR 9.97%, MDD -9.08%; GFC CAGR 3.77%, MDD -2.42%.

## KRW execution test versus KOFR
The low-risk survivor was then fixed; no further parameter selection used OOS.
Execution assumptions:
- Idle KRW earns a Korean 3-month rate proxy (FRED IR3TIB01KRM156N), lagged one month, with 15.4% interest tax.
- TQQQ trades cost 5 bp each side.
- FX sensitivity: 5/10/25 bp per KRW-USD conversion.
- Conservative default tax: 22% marginal tax on positive annual TQQQ realized gains, with no dedicated KRW 2.5m exemption reserved for this sleeve.
- Execution/valuation uses the previous completed USD/KRW close to avoid same-day FX look-ahead.

For a KRW 10m tactical sleeve, 10 bp FX cost per conversion, 22% marginal tax:
- Train: tactical+KOFR CAGR 3.143%, KOFR-only 2.320%, edge +0.823 pp; MDD -7.58%.
- Validation: tactical+KOFR CAGR 4.391%, KOFR-only 1.238%, edge +3.154 pp; MDD -11.52%.
- OOS 2021-2026: tactical+KOFR CAGR 2.704%, KOFR-only 2.225%, edge +0.479 pp; MDD -11.16%; terminal wealth advantage +2.68% over KOFR.

OOS FX sensitivity under 22% marginal tax:
- 5 bp per conversion: CAGR 2.765%, edge over KOFR +0.540 pp, terminal wealth +3.02%.
- 10 bp: CAGR 2.704%, edge +0.479 pp, terminal wealth +2.68%.
- 25 bp: CAGR 2.522%, edge +0.297 pp, terminal wealth +1.65%.
Capital size does not materially change these percentages under the marginal-tax assumption.

## Current state, 2026-08-31 completed data
- QQQ MA250_C3 regime: ON.
- Actual TQQQ drawdown from its 60-day high: -14.78%.
- Low-risk entry levels: -25%, -30%, -35%.
- Therefore: NO TQQQ entry now under the low-risk rule.

## Interpretation
The user's intuition has a real but limited edge: deep-pullback TQQQ rebound trading can add value, but unrestricted averaging hides catastrophic path risk. The low-risk regime-filtered version survived actual OOS and pre-launch crisis stress, but after KOFR yield, tax, FX, and fees its OOS advantage over simply holding KOFR is only about 0.3-0.5 percentage points per year on the tactical sleeve. If this sleeve is only 10% of the total portfolio, that corresponds to only about 0.03-0.05 percentage points of total-portfolio CAGR before interaction effects.

The next decision should therefore be portfolio-level risk budgeting: test more aggressive variants only at 5% and 10% total-portfolio caps. A standalone -75% tactical-sleeve drawdown may be unacceptable, but at a 5% cap its direct contribution is roughly 3.75 percentage points before correlation/rebalancing interaction. Whole-portfolio CAGR/MDD, not standalone sleeve MDD, should determine whether a more aggressive TQQQ sleeve is worthwhile.
