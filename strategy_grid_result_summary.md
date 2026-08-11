# SK hynix synthetic 2x accumulation strategy grid — result summary

Run date: 2026-08-11

## Design

- Underlying: SK hynix 000660 daily OHLCV, synthetic daily-reset 2x series.
- Evaluation windows: monthly starts, 164 independent 3-year windows and 140 independent 5-year windows.
- Strategies: 864 combinations.
- Base trading friction: 5 bps per side.
- Time split counts: 5 / 10 / 20 / 40 trading days.
- Drawdown split: first 1/N tranche at start; remaining N-1 tranches spread from 252-day-high drawdown -30% to -60%, forcing -40% and -50% anchors. A trigger observed at close fills the next open; start-day already-known drawdown can activate due rungs at the start close.
- Permanent cash reserve: 0 / 10 / 20 / 30%.
- Exposure sleeves: 100% core hold; 80% core + 20% trading; 70% core + 30% trading.
- Trading-sleeve take profit: +20 / +30 / +40 / +50% from sleeve average; entire trading sleeve is realized at the chosen target.
- Re-entry: realized trading cash rebuys after synthetic 2x price falls 20% from the last take-profit price; same-day sell/rebuy is disallowed.
- Trend overlay: none / cut 20% / cut 30% after five consecutive closes below underlying SMA200; signal executes next open and restores after five consecutive closes back above SMA200.

## 1. Time split vs drawdown split

Across 432 exactly matched parameter pairs (same split count, reserve, core/trading mix, TP, and trend overlay):

- Time split had the higher median CAGR in 93.5% of 3-year comparisons and 92.6% of 5-year comparisons.
- Drawdown split had the better (less negative) median MDD in 100% of 3-year and 100% of 5-year comparisons.
- Drawdown split also had a lower 5-year loss rate in 100% of matched comparisons.

Interpretation: time split is the return-seeking choice; drawdown split is the capital-preservation / path-risk choice.

## 2. Split count

Within matched parameter sets, the split count producing the highest 5-year median CAGR was:

- Drawdown entry: 40 splits in 77.8% of cases, 20 splits in 22.2%.
- Time entry: 20 splits in 40.7%, 5 splits in 29.6%, 10 splits in 25.9%, 40 splits in only 3.7%.

For risk-adjusted performance across all matched groups, 40 splits was best most often (63.4%), driven mainly by the drawdown-entry family.

## 3. Cash reserve trade-off

Matched against the same strategy with no permanent reserve, the median 5-year effect was:

- 10% reserve: median CAGR -1.35 percentage points; median MDD improved by 2.68 points.
- 20% reserve: median CAGR -2.77 points; median MDD improved by 5.66 points.
- 30% reserve: median CAGR -4.26 points; median MDD improved by 9.16 points.

The 10% reserve was the most efficient compromise in the top-return strategies; 20–30% reserves increasingly traded return for drawdown control.

## 4. 100% hold vs core + trading sleeve

Across matched configurations, fixed mixed-sleeve overlays behaved as follows relative to 100% core hold:

- 70% core + 30% trading at +50% TP improved 3-year median CAGR in 100% of matched configurations and 5-year median CAGR in 54.2%.
- The same 70/30 +50% overlay improved MDD in 100% of matched 3-year and 5-year configurations.
- Median improvement versus pure hold: +1.38 percentage points to 3-year CAGR, +0.72 points to 5-year CAGR, while median MDD improved by 4.24 points (3y) and 6.93 points (5y).
- 70/30 with +40% TP was similarly strong and slightly more robust on the 3-year/5-year combined return ranking.

Factor-level best 5-year median CAGR by exit family:

- 70/30 +20% TP: 18.49%
- 70/30 +30% TP: 19.51%
- 70/30 +40% TP: 22.02%
- 70/30 +50% TP: 22.20%
- 80/20 +40% TP: 21.39%
- 80/20 +50% TP: 21.19%
- 100% hold: 20.26%

The best results clustered around a 30% trading sleeve and relatively wide +40 to +50% profit targets, not frequent +10 to +20% clipping.

## 5. Highest-return strategy with a mandatory reserve

Best combined 3y/5y median-CAGR strategy among reserve > 0 configurations:

**TIME20_R10_CORE70_TP40_RB20**

- 20-trading-day time split
- 10% permanent cash reserve
- 70% core hold / 30% trading sleeve
- trade sleeve TP +40%
- re-entry after -20% pullback from sale price
- no MA200 cut

Results:

- 3y median CAGR: 18.37%
- 5y median CAGR: 20.21%
- 3y median MDD: -60.60%
- 5y median MDD: -70.53%
- 3y 25th-percentile total return: +19.28%
- 5y 25th-percentile total return: +84.61%
- 3y loss rate: 16.46%
- 5y loss rate: 12.14%
- Beat fully-invested B&H: 51.2% of 3y windows and 60.7% of 5y windows.

A close alternative, **TIME20_R10_CORE70_TP50_RB20**, had lower 3y median CAGR (17.81%) but slightly higher 5y median CAGR (20.61%).

## 6. Best risk-adjusted strategy with a mandatory reserve

**DD40_R10_CORE70_TP40_RB20**

- 40-step drawdown ladder
- 10% permanent cash reserve
- 70% core / 30% trading
- trade sleeve TP +40%
- re-entry after -20%
- no MA200 cut

Results:

- 3y median CAGR: 17.58%
- 5y median CAGR: 16.87%
- 3y median MDD: -26.59%
- 5y median MDD: -43.08%
- 3y 25th-percentile total return: +26.69%
- 5y 25th-percentile total return: +82.88%
- 3y loss rate: 5.49%
- 5y loss rate: 0.00%
- Calmar-like score: 0.4945, the best among reserve-required configurations.

The +50% TP version was almost identical on risk-adjusted performance.

## 7. Pure hold entry comparison

With a required 10% reserve:

**DD40_R10_CORE100**
- 3y median CAGR 16.35%
- 5y median CAGR 18.77%
- median MDD -31.09% / -51.33%
- loss rate 8.54% / 0.71%

**TIME20_R10_CORE100**
- 3y median CAGR 15.25%
- 5y median CAGR 16.55%
- median MDD -65.20% / -75.93%
- loss rate 20.73% / 15.71%

For a pure hold mandate, waiting to deploy additional tranches at deep drawdowns was substantially more efficient than blindly finishing a short time split, because unused cash remained available when no large drawdown arrived.

## 8. SMA200 partial reduction

Matched against the same strategy with no trend overlay:

- 20% cut: median 5y CAGR decreased by 0.41 percentage points; median 5y MDD improved by 3.88 points.
- 30% cut: median 5y CAGR decreased by 0.68 points; median 5y MDD improved by 4.96 points.

A particularly attractive high-return compromise was:

**TIME20_R10_CORE70_TP50_RB20_MA200CUT30**

- 3y median CAGR: 17.04%
- 5y median CAGR: 20.23%
- median MDD: -56.85% / -59.75%
- 5y loss rate: 11.43%

Compared with the same strategy without the MA200 cut (17.81% / 20.61% CAGR and -60.85% / -71.30% MDD), the 30% trend cut surrendered only about 0.38 percentage points of 5y median CAGR while improving 5y median MDD by about 11.54 points in this particular configuration.

## 9. Re-entry sensitivity

For top time-split finalists, -15% to -25% re-entry thresholds all remained viable; no knife-edge optimum appeared.

Example TIME20_R10_CORE70_TP40:
- RB10: 3y/5y CAGR 17.37% / 18.58%
- RB15: 18.07% / 20.43%
- RB20: 18.37% / 20.21%
- RB25: 17.58% / 18.51%

Example TIME20_R10_CORE70_TP50:
- RB15: 17.65% / 19.57%
- RB20: 17.81% / 20.61%
- RB25: 17.77% / 20.32%

A -20% re-entry is therefore a reasonable central rule, not a uniquely optimized point.

## 10. Trading-cost sensitivity

For TIME20_R10_CORE70_TP40:

- 0 bps: 3y/5y median CAGR 18.42% / 20.25%
- 5 bps: 18.37% / 20.21%
- 10 bps: 18.31% / 20.17%

The result is not explained by ignoring modest transaction friction.

## 11. Deep-drawdown / high-volatility historical analogs

The historical filter was underlying drawdown <= -35% from the rolling 252-day high and 20-day annualized volatility in the top quartile. Seven historical starts had a full 3-year forward window; the current 2026-08 episode itself was detected but cannot have a forward result yet.

Historical 3-year medians:

- Fully invested 2x B&H: +258.1%, median MDD -59.0%.
- 40-day time DCA then hold: +210.3%.
- TIME20_R10_CORE80_TP50_RB20: +211.2%, median MDD -52.36%.
- TIME20_R10_CORE70_TP40_RB20: +187.7%, median MDD -49.96%.
- TIME20_R10_CORE70_TP50_RB20_MA200CUT30: +185.5%, median MDD -47.17%.

All listed mixed strategies had a 0% loss rate across the seven 3-year analog windows, but fully invested B&H still had the highest median return. This is the key caution for using profit-taking after an already-large drawdown: it can reduce path risk, but it can also clip the strongest multi-year recovery trend.

## Bottom line

The grid does not support either extreme as a universal rule.

1. For maximum expected return across arbitrary start dates, a short time split plus a wide-target 30% trading sleeve was strongest, with 10% permanent cash reserve a reasonable compromise.
2. For materially lower drawdown and much lower loss probability, a 40-step drawdown ladder was superior, especially with 10–20% reserve.
3. Small, frequent profit-taking is not supported. The useful trading overlay used only 20–30% of the position and wide +40/+50% targets.
4. SMA200-based partial reduction acts mainly as insurance: it usually lowers return slightly and improves drawdown.
5. After an already-severe drawdown, historical analogs still favored keeping a large core exposed to the full recovery rather than repeatedly harvesting it.

This is a synthetic historical study. The real single-stock leveraged ETF has fees, tracking error, financing/futures effects, NAV deviations, liquidity/spread effects, and a short live history that are not fully represented here.
