import math
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

CAP = 10_000_000.0
BASE_BPS = 5.0
REENTRY_DROP = 0.20
SPLITS = [5, 10, 20, 40]
RESERVES = [0.0, 0.10, 0.20, 0.30]
CORE_RATIOS = [1.0, 0.80, 0.70]
TP_TARGETS = [0.20, 0.30, 0.40, 0.50]
TREND_CUTS = [0.0, 0.20, 0.30]


def dd_thresholds(n):
    """N-1 drawdown thresholds from -30% to -60%, forcing -40/-50 anchors."""
    m = n - 1
    x = np.linspace(-0.30, -0.60, m)
    for anchor in (-0.40, -0.50):
        j = int(np.argmin(np.abs(x - anchor)))
        x[j] = anchor
    x = np.sort(x)[::-1]  # -0.30, ... , -0.60
    return x.astype(np.float64)


def update_avg(q, avg, add_q, price):
    if add_q <= 0:
        return avg
    if q <= 0:
        return price
    return (q * avg + add_q * price) / (q + add_q)


@njit(cache=True)
def simulate(o, h, l, c, dd, below5, above5,
             entry_kind, splits, reserve_frac, core_ratio, tp_target, trend_cut,
             dd_thr, fee_bps, reentry_drop):
    fee = fee_bps / 10000.0
    protected = CAP * reserve_frac
    entry_cash = CAP - protected
    tranche = entry_cash / splits
    core_q = 0.0
    trade_q = 0.0
    core_avg = 0.0
    trade_avg = 0.0
    trade_cash = 0.0
    trend_core_cash = 0.0
    trend_trade_cash = 0.0
    last_sale = 0.0
    tranches_bought = 0
    trend_reduced = False
    pending = 0  # -1 cut, +1 restore next open
    trades = 0

    peak_eq = CAP
    min_dd = 0.0

    n = len(c)
    for i in range(n):
        op = o[i]
        hi = h[i]
        lo = l[i]
        cl = c[i]

        # Execute previous-close MA200 action at today's open.
        if pending == -1 and trend_cut > 0.0:
            if core_q + trade_q > 0.0:
                sell_c = core_q * trend_cut
                sell_t = trade_q * trend_cut
                if sell_c > 0:
                    trend_core_cash += sell_c * op * (1.0 - fee)
                    core_q -= sell_c
                    trades += 1
                if sell_t > 0:
                    trend_trade_cash += sell_t * op * (1.0 - fee)
                    trade_q -= sell_t
                    trades += 1
                trend_reduced = True
        elif pending == 1 and trend_cut > 0.0:
            if trend_core_cash > 1e-12:
                inv = trend_core_cash / (1.0 + fee)
                addq = inv / op
                core_avg = update_avg(core_q, core_avg, addq, op)
                core_q += addq
                trend_core_cash -= inv * (1.0 + fee)
                if trend_core_cash < 1e-8: trend_core_cash = 0.0
                trades += 1
            if trend_trade_cash > 1e-12:
                inv = trend_trade_cash / (1.0 + fee)
                addq = inv / op
                trade_avg = update_avg(trade_q, trade_avg, addq, op)
                trade_q += addq
                trend_trade_cash -= inv * (1.0 + fee)
                if trend_trade_cash < 1e-8: trend_trade_cash = 0.0
                trades += 1
            trend_reduced = False
        pending = 0

        # Entry plan. Time split = one tranche per first N closes.
        # Drawdown split = one starter at day 0 + remaining tranches as DD levels are reached.
        due = 0
        entry_price = cl
        if entry_kind == 0:
            if tranches_bought < splits and i < splits:
                due = 1
                entry_price = cl
        else:
            if i == 0 and tranches_bought == 0:
                due = 1
                entry_price = cl
                # At the start close, current drawdown is already observable; deploy all already-triggered rungs.
                k = 0
                while k < len(dd_thr) and dd[i] <= dd_thr[k]:
                    due += 1
                    k += 1
            elif tranches_bought < splits and i > 0:
                # Use yesterday's close-based drawdown, fill today at open: no look-ahead.
                k = tranches_bought - 1
                while k < len(dd_thr) and dd[i-1] <= dd_thr[k]:
                    due += 1
                    k += 1
                entry_price = op

        if due > 0:
            max_due = splits - tranches_bought
            if due > max_due: due = max_due
            gross_budget = tranche * due
            if gross_budget > entry_cash: gross_budget = entry_cash
            if gross_budget > 1e-12:
                core_amt = gross_budget * core_ratio
                trade_amt = gross_budget - core_amt
                if core_amt > 0:
                    inv = core_amt / (1.0 + fee)
                    addq = inv / entry_price
                    core_avg = update_avg(core_q, core_avg, addq, entry_price)
                    core_q += addq
                    trades += 1
                if trade_amt > 0:
                    inv = trade_amt / (1.0 + fee)
                    addq = inv / entry_price
                    trade_avg = update_avg(trade_q, trade_avg, addq, entry_price)
                    trade_q += addq
                    trades += 1
                entry_cash -= gross_budget
                tranches_bought += due

        sold_today = False
        # Profit-taking only on trading sleeve; fill at limit target if day's high touches it.
        if tp_target > 0.0 and trade_q > 1e-12 and trade_avg > 0.0:
            target = trade_avg * (1.0 + tp_target)
            if hi >= target:
                trade_cash += trade_q * target * (1.0 - fee)
                trade_q = 0.0
                trade_avg = 0.0
                last_sale = target
                sold_today = True
                trades += 1

        # Rebuy realized trading sleeve after a 2x-price pullback from the last sale.
        # No same-day sell/rebuy to avoid optimistic intraday path assumptions.
        if (not sold_today) and trade_cash > 1e-12 and last_sale > 0.0:
            rebuy = last_sale * (1.0 - reentry_drop)
            if lo <= rebuy:
                inv = trade_cash / (1.0 + fee)
                addq = inv / rebuy
                trade_avg = update_avg(trade_q, trade_avg, addq, rebuy)
                trade_q += addq
                trade_cash -= inv * (1.0 + fee)
                if trade_cash < 1e-8: trade_cash = 0.0
                last_sale = 0.0
                trades += 1

        # Generate trend signal at close, execute next open.
        if trend_cut > 0.0:
            if (not trend_reduced) and below5[i] == 1 and (core_q + trade_q) > 1e-12:
                pending = -1
            elif trend_reduced and above5[i] == 1 and (trend_core_cash + trend_trade_cash) > 1e-12:
                pending = 1

        eq = protected + entry_cash + trade_cash + trend_core_cash + trend_trade_cash + (core_q + trade_q) * cl
        if eq > peak_eq: peak_eq = eq
        curdd = eq / peak_eq - 1.0
        if curdd < min_dd: min_dd = curdd

    final = protected + entry_cash + trade_cash + trend_core_cash + trend_trade_cash + (core_q + trade_q) * c[-1]
    return final / CAP - 1.0, min_dd, trades


def make_windows(dates, years):
    starts = pd.date_range(dates.min().to_period('M').start_time, dates.max(), freq='MS')
    out = []
    seen = set()
    arr = dates.to_numpy()
    for cand in starts:
        si = int(np.searchsorted(arr, np.datetime64(cand), side='left'))
        if si >= len(dates):
            continue
        actual = dates.iloc[si]
        if actual in seen:
            continue
        seen.add(actual)
        target = actual + pd.DateOffset(years=years)
        if dates.iloc[-1] < target:
            continue
        ei = int(np.searchsorted(arr, np.datetime64(target), side='right')) - 1
        if ei - si + 1 < years * 200:
            continue
        out.append((si, ei, actual, dates.iloc[ei]))
    return out


def cagr_from_return(r, years):
    if r <= -1:
        return -1.0
    return (1.0 + r) ** (1.0 / years) - 1.0


def bh_return(close, fee_bps):
    f = fee_bps / 10000.0
    return (close[-1] / close[0]) / (1.0 + f) - 1.0


def strategy_name(entry_kind, splits, reserve, core, tp, trend):
    e = 'TIME' if entry_kind == 0 else 'DD'
    x = f'{e}{splits}_R{int(reserve*100):02d}_CORE{int(core*100):02d}'
    if core < 0.999:
        x += f'_TP{int(tp*100):02d}_RB20'
    if trend > 0:
        x += f'_MA200CUT{int(trend*100):02d}'
    return x


def build_strategies():
    rows=[]
    for entry_kind in [0,1]:
        for splits in SPLITS:
            thr = dd_thresholds(splits)
            for reserve in RESERVES:
                for core in CORE_RATIOS:
                    tps = [0.0] if core == 1.0 else TP_TARGETS
                    for tp in tps:
                        for trend in TREND_CUTS:
                            rows.append(dict(
                                entry_kind=entry_kind, splits=splits, reserve=reserve,
                                core=core, tp=tp, trend=trend, dd_thr=thr,
                                strategy=strategy_name(entry_kind,splits,reserve,core,tp,trend)
                            ))
    return rows


def summarize_results(results, years, bh_by_window):
    a = np.array([x[0] for x in results], dtype=float)
    m = np.array([x[1] for x in results], dtype=float)
    t = np.array([x[2] for x in results], dtype=float)
    cg = np.array([cagr_from_return(x, years) for x in a])
    bh = np.array(bh_by_window, dtype=float)
    return {
        f'n{years}': len(a),
        f'med_ret{years}': float(np.median(a)),
        f'med_cagr{years}': float(np.median(cg)),
        f'p25_ret{years}': float(np.quantile(a, .25)),
        f'p10_ret{years}': float(np.quantile(a, .10)),
        f'med_mdd{years}': float(np.median(m)),
        f'p10_mdd{years}': float(np.quantile(m, .10)),
        f'loss_rate{years}': float(np.mean(a < 0)),
        f'bh_win_rate{years}': float(np.mean(a > bh)),
        f'med_excess_bh{years}': float(np.median(a - bh)),
        f'med_trades{years}': float(np.median(t)),
    }


def pareto(df):
    # Maximize 5y median CAGR, minimize absolute 5y median MDD.
    x = df.copy().sort_values('med_cagr5', ascending=False)
    best_risk = np.inf
    keep=[]
    for idx,r in x.iterrows():
        risk=abs(r.med_mdd5)
        if risk < best_risk - 1e-12:
            keep.append(idx); best_risk=risk
    return df.loc[keep].sort_values(['med_cagr5','med_cagr3'],ascending=False)


def current_like_starts(u):
    f=u[['Date','Close']].copy()
    f['ret']=f.Close.pct_change()
    f['peak']=f.Close.rolling(252,min_periods=60).max()
    f['dd']=f.Close/f.peak-1
    f['vol']=f.ret.rolling(20).std()*np.sqrt(252)
    f['vp']=f.vol.rank(pct=True)
    f['m']=f.Date.dt.to_period('M')
    return f.groupby('m').first().query('dd <= -0.35 and vp >= 0.75').reset_index(drop=True)


def evaluate_param(u, L, features, windows, p, years, fee_bps=BASE_BPS, reentry_drop=REENTRY_DROP):
    out=[]
    bh=[]
    for si,ei,_,_ in windows:
        sl=slice(si,ei+1)
        r=simulate(
            L.Open.values[sl],L.High.values[sl],L.Low.values[sl],L.Close.values[sl],
            features.dd.values[sl],features.below5.values[sl],features.above5.values[sl],
            p['entry_kind'],p['splits'],p['reserve'],p['core'],p['tp'],p['trend'],
            p['dd_thr'],fee_bps,reentry_drop)
        out.append(r)
        bh.append(bh_return(L.Close.values[sl],fee_bps))
    return summarize_results(out,years,bh)


def main():
    outdir=Path('results_grid'); outdir.mkdir(exist_ok=True)
    u=pd.read_csv('results/underlying.csv',parse_dates=['Date'])
    L=pd.read_csv('results/synthetic2x.csv',parse_dates=['Date'])
    z=u[['Date','Close']].copy()
    z['peak252']=z.Close.rolling(252,min_periods=1).max()
    z['dd']=z.Close/z.peak252-1
    z['ma200']=z.Close.rolling(200,min_periods=200).mean()
    below=(z.Close<z.ma200).fillna(False).to_numpy(); above=(z.Close>z.ma200).fillna(False).to_numpy()
    bs=np.zeros(len(z),dtype=np.int8); ab=np.zeros(len(z),dtype=np.int8)
    run=0
    for i,v in enumerate(below):
        run=run+1 if v else 0
        if run>=5: bs[i]=1
    run=0
    for i,v in enumerate(above):
        run=run+1 if v else 0
        if run>=5: ab[i]=1
    z['below5']=bs; z['above5']=ab

    w3=make_windows(u.Date,3); w5=make_windows(u.Date,5)
    print('WINDOWS',len(w3),len(w5))
    assert len(w3)==164 and len(w5)==140, (len(w3),len(w5))

    strategies=build_strategies()
    print('STRATEGIES',len(strategies))
    # JIT warm-up
    p0=strategies[0]; si,ei,_,_=w3[0]; sl=slice(si,ei+1)
    simulate(L.Open.values[sl],L.High.values[sl],L.Low.values[sl],L.Close.values[sl],z.dd.values[sl],z.below5.values[sl],z.above5.values[sl],p0['entry_kind'],p0['splits'],p0['reserve'],p0['core'],p0['tp'],p0['trend'],p0['dd_thr'],BASE_BPS,REENTRY_DROP)

    rows=[]
    for j,p in enumerate(strategies):
        r={'strategy':p['strategy'],'entry':'TIME' if p['entry_kind']==0 else 'DRAWDOWN','splits':p['splits'],'reserve':p['reserve'],'core':p['core'],'tp':p['tp'],'trend_cut':p['trend']}
        r.update(evaluate_param(u,L,z,w3,p,3))
        r.update(evaluate_param(u,L,z,w5,p,5))
        r['avg_med_cagr']=(r['med_cagr3']+r['med_cagr5'])/2
        r['avg_abs_mdd']=(abs(r['med_mdd3'])+abs(r['med_mdd5']))/2
        r['calmar_like']=r['avg_med_cagr']/max(r['avg_abs_mdd'],1e-9)
        r['avg_loss_rate']=(r['loss_rate3']+r['loss_rate5'])/2
        rows.append(r)
        if (j+1)%100==0: print('DONE',j+1,'/',len(strategies))
    df=pd.DataFrame(rows)
    df.to_csv(outdir/'all_864_strategies.csv',index=False)

    # Reference summaries and factor slices.
    hold=df[(df.core==1.0)&(df.tp==0)&(df.trend_cut==0)].copy()
    hold.sort_values(['med_cagr5','med_cagr3'],ascending=False).to_csv(outdir/'entry_reserve_hold_comparison.csv',index=False)
    df[df.reserve>0].sort_values(['med_cagr5','med_cagr3'],ascending=False).head(50).to_csv(outdir/'top50_return_reserve_required.csv',index=False)
    df[df.reserve>0].sort_values(['calmar_like','avg_med_cagr'],ascending=False).head(50).to_csv(outdir/'top50_risk_adjusted_reserve_required.csv',index=False)
    par=pareto(df[df.reserve>0]); par.to_csv(outdir/'pareto_reserve_required.csv',index=False)

    fac_entry=df.groupby(['entry','splits','reserve'],as_index=False).agg(
        best_med_cagr5=('med_cagr5','max'),median_med_cagr5=('med_cagr5','median'),
        best_calmar=('calmar_like','max'),median_mdd5=('med_mdd5','median'))
    fac_entry.to_csv(outdir/'factor_entry.csv',index=False)
    fac_exit=df.groupby(['core','tp'],as_index=False).agg(
        best_med_cagr5=('med_cagr5','max'),median_med_cagr5=('med_cagr5','median'),
        best_calmar=('calmar_like','max'),median_mdd5=('med_mdd5','median'))
    fac_exit.to_csv(outdir/'factor_exit.csv',index=False)
    fac_trend=df.groupby(['trend_cut'],as_index=False).agg(
        best_med_cagr5=('med_cagr5','max'),median_med_cagr5=('med_cagr5','median'),
        median_mdd5=('med_mdd5','median'),median_loss5=('loss_rate5','median'))
    fac_trend.to_csv(outdir/'factor_trend.csv',index=False)

    # Finalist union: return leaders + risk-adjusted leaders, reserve required.
    elig=df[df.reserve>0].copy()
    finalists=pd.concat([
        elig.nlargest(12,'avg_med_cagr'),
        elig.nlargest(12,'calmar_like')
    ]).drop_duplicates('strategy').copy()
    finalists.to_csv(outdir/'finalists.csv',index=False)

    # Current-like historical analogs: same deep-DD/high-vol definition as prior work.
    analog=current_like_starts(u)
    print('CURRENT_LIKE_STARTS',len(analog),analog[['Date','dd','vol','vp']].to_string(index=False))
    currows=[]
    date_arr=u.Date.to_numpy()
    for _,ar in analog.iterrows():
        si=int(np.searchsorted(date_arr,np.datetime64(ar.Date),side='left'))
        for years in [1,2,3]:
            target=u.Date.iloc[si]+pd.DateOffset(years=years)
            if u.Date.iloc[-1]<target: continue
            ei=int(np.searchsorted(date_arr,np.datetime64(target),side='right'))-1
            if ei-si+1<years*200: continue
            sl=slice(si,ei+1)
            for _,fr in finalists.iterrows():
                p=dict(entry_kind=0 if fr.entry=='TIME' else 1,splits=int(fr.splits),reserve=float(fr.reserve),core=float(fr.core),tp=float(fr.tp),trend=float(fr.trend_cut),dd_thr=dd_thresholds(int(fr.splits)))
                rr=simulate(L.Open.values[sl],L.High.values[sl],L.Low.values[sl],L.Close.values[sl],z.dd.values[sl],z.below5.values[sl],z.above5.values[sl],p['entry_kind'],p['splits'],p['reserve'],p['core'],p['tp'],p['trend'],p['dd_thr'],BASE_BPS,REENTRY_DROP)
                currows.append(dict(strategy=fr.strategy,start=u.Date.iloc[si],years=years,ret=rr[0],mdd=rr[1],trades=rr[2],start_dd=ar.dd,start_vol=ar.vol))
    cur=pd.DataFrame(currows)
    cur.to_csv(outdir/'current_like_finalist_windows.csv',index=False)
    if len(cur):
        cs=cur.groupby(['strategy','years'],as_index=False).agg(n=('ret','size'),med_ret=('ret','median'),p25_ret=('ret',lambda x: x.quantile(.25)),med_mdd=('mdd','median'),loss_rate=('ret',lambda x:(x<0).mean()))
        cs.to_csv(outdir/'current_like_finalist_summary.csv',index=False)

    # Re-entry sensitivity on mixed-sleeve finalists.
    sensrows=[]
    mixed=finalists[finalists.core<1.0].nlargest(10,'avg_med_cagr')
    for _,fr in mixed.iterrows():
        for rb in [0.10,0.15,0.20,0.25]:
            p=dict(entry_kind=0 if fr.entry=='TIME' else 1,splits=int(fr.splits),reserve=float(fr.reserve),core=float(fr.core),tp=float(fr.tp),trend=float(fr.trend_cut),dd_thr=dd_thresholds(int(fr.splits)))
            s={'strategy':fr.strategy,'reentry_drop':rb}
            s.update(evaluate_param(u,L,z,w3,p,3,BASE_BPS,rb)); s.update(evaluate_param(u,L,z,w5,p,5,BASE_BPS,rb))
            sensrows.append(s)
    pd.DataFrame(sensrows).to_csv(outdir/'reentry_sensitivity.csv',index=False)

    # Cost sensitivity on top union.
    costrows=[]
    topcost=finalists.nlargest(10,'avg_med_cagr')
    for _,fr in topcost.iterrows():
        p=dict(entry_kind=0 if fr.entry=='TIME' else 1,splits=int(fr.splits),reserve=float(fr.reserve),core=float(fr.core),tp=float(fr.tp),trend=float(fr.trend_cut),dd_thr=dd_thresholds(int(fr.splits)))
        for bps in [0.0,5.0,10.0]:
            s={'strategy':fr.strategy,'bps':bps}
            s.update(evaluate_param(u,L,z,w3,p,3,bps,REENTRY_DROP)); s.update(evaluate_param(u,L,z,w5,p,5,bps,REENTRY_DROP))
            costrows.append(s)
    pd.DataFrame(costrows).to_csv(outdir/'cost_sensitivity_finalists.csv',index=False)

    # Compact console report.
    cols=['strategy','med_cagr3','med_cagr5','med_mdd3','med_mdd5','p25_ret3','p25_ret5','loss_rate3','loss_rate5','bh_win_rate3','bh_win_rate5','calmar_like']
    print('\n=== TOP RETURN, RESERVE REQUIRED ===')
    print(elig.nlargest(15,'avg_med_cagr')[cols].to_string(index=False))
    print('\n=== TOP RISK-ADJUSTED, RESERVE REQUIRED ===')
    print(elig.nlargest(15,'calmar_like')[cols].to_string(index=False))
    print('\n=== PARETO, RESERVE REQUIRED ===')
    print(par[cols].head(30).to_string(index=False))
    print('\n=== HOLD ENTRY/RESERVE ===')
    print(hold[['strategy','med_cagr3','med_cagr5','med_mdd3','med_mdd5','loss_rate3','loss_rate5']].sort_values('med_cagr5',ascending=False).to_string(index=False))
    print('\n=== FACTOR EXIT ===')
    print(fac_exit.to_string(index=False))
    print('\n=== FACTOR TREND ===')
    print(fac_trend.to_string(index=False))
    if len(cur):
        print('\n=== CURRENT-LIKE FINALISTS 3Y ===')
        q=cs[cs.years==3].sort_values(['med_ret','med_mdd'],ascending=[False,False]).head(20)
        print(q.to_string(index=False))
    print('\nSAVED',outdir.resolve())

if __name__=='__main__':
    main()
