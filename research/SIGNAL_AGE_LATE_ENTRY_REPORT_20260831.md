# Signal-age / late-entry study — 2026-08-31

## Method
- Existing rules only: SPY MA250_C5, QQQ MA250_C3, BTC MA150_C3, KOSPI200 MA100_C3.
- Signal is formed on completed close and executable at the next native-market open.
- Signal-age buckets: 0–20, 21–60, 61–120, 121–250, 251+ native bars.
- Calendar-entry analysis evaluates every possible fresh start while ON.
- Episode-checkpoint analysis uses at most one observation per ON episode at ages 0, 20, 60, 120, 250 to reduce overweighting of long trends.
- Return-to-OFF includes approximate entry and exit trading fees. 63/126/252-bar returns continue following future ON/OFF signals; cash yield is zero to isolate the signal-age effect.

## Current signals / ages
| Asset | State | Signal start close | Age (native bars) | Bucket | Close vs MA |
|---|---:|---|---:|---|---:|
| SPY | ON | 2025-05-14 | 324 | 251+ | +10.23% |
| QQQ | ON | 2025-05-12 | 326 | 251+ | +11.36% |
| BTC | ON | 2026-08-21 | 9 | 0–20 | +11.55% |
| KOSPI200 | OFF | — | — | — | -5.36% |

## SPY
Full-history calendar-entry by age:
- 0–20: median return-to-OFF +0.44%, p10 -5.87%, win 53.1%; 63/126/252-bar strategy returns +4.55% / +6.49% / +16.22%.
- 21–60: +11.21%, p10 -5.76%, win 65.2%; +5.19% / +9.22% / +15.96%.
- 61–120: +20.81%, p10 -9.83%, win 71.2%; +3.98% / +8.34% / +14.19%.
- 121–250: +9.48%, p10 -13.59%, win 64.7%; +4.50% / +7.90% / +11.33%.
- 251+: +2.56%, p10 -8.20%, win 57.4%; +3.30% / +5.77% / +7.71%.

2018+ calendar-entry, 251+ only: median return-to-OFF -3.16%, p10 -10.45%, win 25.3%, median remaining life 108 bars; 63/126/252-bar strategy returns +1.58% / -1.82% / ~0%.

Episode checkpoints (full history): age 0 / 20 / 60 / 120 / 250 median return-to-OFF = -0.25% / +8.22% / +17.35% / +20.55% / +4.94%. At age 250, n=10 episodes, p10 -7.39%, win 77.8%.

Interpretation: SPY continuation weakens after very old signal ages, especially in the recent sample, but the episode-level evidence is not monotonic and does not support a hard wait-for-next-ON rule by itself.

## QQQ
Full-history calendar-entry by age:
- 0–20: median return-to-OFF +0.10%, p10 -6.00%, win 50.6%; 63/126/252 +5.34% / +9.45% / +16.84%.
- 21–60: +3.08%, p10 -7.50%, win 55.7%; +4.53% / +9.52% / +11.21%.
- 61–120: +1.97%, p10 -9.16%, win 60.0%; +4.56% / +9.55% / +10.53%.
- 121–250: -3.01%, p10 -11.26%, win 40.4%; +4.13% / +4.74% / +13.26%.
- 251+: -1.33%, p10 -9.64%, win 43.9%; +2.64% / +5.28% / +2.96%.

2018+ calendar-entry, 251+ only: median return-to-OFF -2.54%, p10 -9.90%, win 36.1%; 63/126/252 +2.65% / -0.05% / +2.51%.

Episode checkpoints: age 0 / 20 / 60 / 120 / 250 median return-to-OFF = -0.95% / +1.48% / +2.11% / -0.30% / -1.35%. Age250 has n=10 episodes, p10 -10.52%, win 44.4%.

Interpretation: QQQ shows the clearest late-entry deterioration after roughly 120+ bars. However, forward strategy returns that continue following later ON/OFF signals remain generally positive; this study alone does not prove that waiting until the next fresh ON improves portfolio-level after-tax results.

## BTC
Current BTC age is only 9 daily bars, so it is a fresh 0–20 signal, not a late entry.

Calendar-entry:
- 0–20: median return-to-OFF +0.83%, p10 -15.63%, worst -39.47%, win 52.1%, median remaining life 105.5 days, median MAE -6.38%, p10 MAE -16.76%; 63/126/252 strategy returns +5.52% / +13.56% / +11.90%.
- 21–60: +7.62%, p10 -13.27%, win 71.5%; +15.41% / +12.32% / +24.05%.
- 61–120: -7.08%, p10 -20.90%, win 34.5%; -2.60% / -6.75% / +5.08%.
- 121–250: -4.69%, p10 -19.80%, win 34.8%; -3.77% / -2.79% / +28.07%.
- 251+: -17.08%, p10 -27.27%, win 24.2%; -11.75% / -15.28% / -0.51%. Only two long episodes reach this bucket, so this last row is weak evidence.

Episode checkpoints: age 0 / 20 / 60 / 120 / 250 median return-to-OFF = -3.42% / +11.28% / +3.98% / -3.13% / +8.63%. Age250 has only n=2 and should not be used for inference.

Interpretation: BTC freshness is nonlinear. Fresh 0–20 signals have false-breakout risk, but current age 9 is clearly not a stale-signal problem. Historically 21–60 was the strongest continuation zone; deterioration becomes visible around 60–120+ bars.

## KOSPI200
Current signal is OFF. There is no fresh-entry decision today. Very old 251+ signals were poor, but only two episodes reached that age.

## Execution implication
- BTC: no freshness-based reason to defer the current ON allocation; current signal age = 9 bars. The relevant risk is whipsaw, not late chasing.
- SPY/VOO: current signal is old, but evidence is mixed. Do not add a new late-entry gate without a portfolio-level launch-policy backtest.
- QQQ/QQQM: current signal is old and late-entry deterioration is real enough to justify a dedicated launch-gating test.
- RISE200: remain OFF; its sleeve stays in KOFR.

## Required next test before changing current VOO/QQQM execution
Portfolio-level initial-launch gate, keeping all later rules unchanged:
1. Baseline: if ON at portfolio launch, enter immediately.
2. Gate120: if initial signal age >120, park that sleeve in KOFR until the next OFF→ON.
3. Gate250: same with >250.
4. Asset-specific candidate: QQQ gate >120, SPY no gate; BTC gate >60 or >120 only as sensitivity.
Use rolling monthly launch cohorts and the actual execution route (VOO / QQQM / KRW BTC / RISE200 / KOFR), including tax, trading cost, and FX. Select the launch policy without using final OOS.
