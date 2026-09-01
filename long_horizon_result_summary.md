# SK hynix synthetic 2x — long horizon audit

Run date: 2026-09-01

## Setup
- Data: synthetic daily-reset 2x series based on SK hynix OHLCV, 2010-01-04 through 2026-08-07.
- Friction: 5 bps per side.
- Strategies compared:
  - BH: fully invested synthetic 2x buy-and-hold.
  - HOLD90: 10% initial permanent reserve, remaining 90% invested over 20 trading days and held.
  - CORE80_TP50: 10% reserve, 20-day entry, 80% core + 20% trading sleeve, trading sleeve exits at +50% and re-enters after -20% from sale price.
  - CORE80_SPLIT40_50: 10% reserve, 20-day entry, 80% core + two 10% trading sleeves, exits at +40% and +50%, each re-enters after -20%.
- Long horizons: 7, 10, 12, 15 years. Monthly starts and annual starts were both tested.
- Intraday events are processed before same-day close entries, and same-day sell/rebuy is disallowed.

## Monthly-start medians
| Horizon | Windows | BH CAGR | HOLD90 CAGR | CORE80 TP50 CAGR | CORE80 split 40/50 CAGR | BH MDD | TP50 MDD | Split MDD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 7y | 116 | 19.51% | 18.60% | 20.08% | 19.99% | -79.11% | -73.86% | -73.32% |
| 10y | 80 | 16.87% | 16.04% | 19.24% | 18.30% | -80.43% | -74.11% | -73.97% |
| 12y | 56 | 19.66% | 18.84% | 18.43% | 18.46% | -80.43% | -74.11% | -74.00% |
| 15y | 20 | 27.73% | 26.49% | 29.42% | 27.74% | -84.45% | -78.14% | -78.02% |

All strategies had a 0% terminal loss rate in every 7y, 10y, 12y and 15y monthly-start window. This does not mean path risk was low: median MDDs remained roughly -73% to -84% depending on strategy and horizon.

Worst terminal total returns in monthly windows were still positive at 7y: BH +13.1%, HOLD90 +15.6%, CORE80_TP50 +20.9%, CORE80_SPLIT40_50 +24.1%. At 10y the worst total returns were BH +81.3%, HOLD90 +79.3%, TP50 +127.8%, split +140.6%.

## Annual-start medians
| Horizon | Windows | BH CAGR | HOLD90 CAGR | CORE80 TP50 CAGR | CORE80 split 40/50 CAGR |
|---|---:|---:|---:|---:|---:|
| 7y | 10 | 18.54% | 17.19% | 19.15% | 18.47% |
| 10y | 7 | 16.91% | 15.13% | 18.76% | 19.90% |
| 12y | 5 | 19.75% | 17.53% | 17.29% | 18.28% |
| 15y | 2 | 23.22% | 21.77% | 25.18% | 24.70% |

The 12-year windows are an important counterexample: buy-and-hold beat both trading overlays on median CAGR. Therefore the profit-taking overlay is not uniformly superior as horizon increases.

## Full available period: 2010-01-04 to 2026-08-07 (16.59 years)
| Strategy | Final from KRW 10m | CAGR | MDD |
|---|---:|---:|---:|
| BH | KRW 1.591bn | 35.74% | -84.45% |
| HOLD90 | KRW 1.390bn | 34.64% | -83.37% |
| CORE80_TP50 | KRW 1.266bn | 33.89% | -81.37% |
| CORE80_SPLIT40_50 | KRW 1.225bn | 33.62% | -81.90% |

Over the single full-history path, fully invested buy-and-hold had the highest terminal wealth and CAGR. Relative to HOLD90, the 20% trading overlay reduced full-period CAGR by about 0.76 percentage points while improving MDD by about 2.0 points.

## Interpretation
1. The prior 3y/5y finding survives into 7y and 10y: a modest trading sleeve can improve median CAGR and reduce MDD in many starting windows.
2. It is not monotonic. At 12y, pure buy-and-hold won on median CAGR; over the full 16.59y path it also won decisively in terminal wealth.
3. The single +50% trading sleeve was at least as good as, and usually better than, splitting the 20% trading sleeve into +40%/+50% targets. The split rule therefore does not have enough evidence to justify its extra complexity.
4. A trading sleeve consistently reduced drawdown relative to a same-reserve HOLD90 strategy, but the improvement was only a few percentage points while absolute MDD remained severe.
5. Long holding horizons eliminated terminal losses in this historical sample, but did not eliminate catastrophic interim drawdowns.
6. Monthly windows overlap heavily, and 12y/15y annual samples are very small. These results are descriptive historical evidence, not independent statistical trials.
7. The next decisive validation should be out-of-sample / walk-forward: choose rules using an earlier training period and evaluate later years without retuning.
