#!/usr/bin/env python3
from __future__ import annotations
import io, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import yfinance as yf

OUT = Path("results/kospi200_long_trend")
OUT.mkdir(parents=True, exist_ok=True)
FEE = 0.0005
MA_WINDOWS = [100,150,200,250]
CONFIRMS = [1,3,5]
MONTH_RULES = [10,12]

def yf_ohlc(ticker, start="1990-01-01", auto_adjust=False):
    d = yf.download(ticker, start=start, auto_adjust=auto_adjust, progress=False, threads=False)
    if d is None or len(d)==0:
        raise RuntimeError(f"no data {ticker}")
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.reset_index()
    datecol = "Date" if "Date" in d.columns else d.columns[0]
    d = d.rename(columns={datecol:"date","Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close"})
    d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None)
    for c in ["open","high","low","close"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d[["date","open","high","low","close"]].dropna().sort_values("date").drop_duplicates("date").reset_index(drop=True)

def fred_korea_3m():
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=IR3TIB01KRM156N"
    r = requests.get(url, timeout=30); r.raise_for_status()
    x = pd.read_csv(io.StringIO(r.text))
    x.columns = ["obs_date","yield_pct"]
    x["obs_date"] = pd.to_datetime(x["obs_date"])
    x["yield_pct"] = pd.to_numeric(x["yield_pct"], errors="coerce")
    x = x.dropna().sort_values("obs_date").reset_index(drop=True)
    x["available_date"] = x["obs_date"] + pd.Timedelta(days=45)
    return x[["available_date","yield_pct"]]

def attach_rate(d):
    r = fred_korea_3m()
    z = pd.merge_asof(d.sort_values("date"), r.sort_values("available_date"), left_on="date", right_on="available_date", direction="backward")
    z["yield_pct"] = z["yield_pct"].ffill().fillna(0.0)
    return z.drop(columns=["available_date"])

def state_daily(close, ma, confirm):
    above = close > ma
    below = close < ma
    out = np.zeros(len(close), dtype=np.int8)
    cur = 0
    for i in range(len(close)):
        if i >= confirm-1 and bool(above.iloc[i-confirm+1:i+1].all()): cur = 1
        elif i >= confirm-1 and bool(below.iloc[i-confirm+1:i+1].all()): cur = 0
        out[i] = cur
    return out

def state_monthly(d, months):
    x = d[["date","close"]].copy(); x["period"] = x["date"].dt.to_period("M")
    last = x.groupby("period", sort=True).tail(1).copy()
    last["sma"] = last["close"].rolling(months, min_periods=months).mean()
    last["sig"] = (last["close"] > last["sma"]).astype(float)
    signal_map = dict(zip(last["date"], last["sig"]))
    out = np.zeros(len(d), dtype=np.int8); cur = 0
    for i,dt in enumerate(d["date"]):
        if dt in signal_map and not pd.isna(signal_map[dt]): cur = int(signal_map[dt])
        out[i] = cur
    return out

def rule_states(d):
    states = {"BH": np.ones(len(d), dtype=np.int8)}
    for w in MA_WINDOWS:
        ma = d["close"].rolling(w, min_periods=w).mean()
        for c in CONFIRMS: states[f"MA{w}_C{c}"] = state_daily(d["close"], ma, c)
    for mm in MONTH_RULES: states[f"M{mm}"] = state_monthly(d, mm)
    return states

def cash_factor(rate_pct, days):
    if days <= 0: return 1.0
    return float((1.0 + max(float(rate_pct),0.0)/100.0)**(days/365.2425))

def simulate(d, state, start_idx, end_idx, fee=FEE, use_rate=True):
    cash=1.0; qty=0.0; held=0; vals=[]; trades=0; prevdate=None
    for i in range(start_idx, end_idx+1):
        dt=d.loc[i,"date"]
        if prevdate is not None and cash>0:
            days=(dt-prevdate).days; rate=float(d.loc[i,"yield_pct"]) if use_rate else 0.0
            cash *= cash_factor(rate, days)
        desired = 1 if (i==start_idx and state[i]==1) else int(state[i-1] if i>0 else state[i])
        op=float(d.loc[i,"open"])
        if desired!=held:
            if desired and cash>0:
                qty=(cash/(1.0+fee))/op; cash=0.0; held=1; trades+=1
            elif (not desired) and qty>0:
                cash=qty*op*(1.0-fee); qty=0.0; held=0; trades+=1
        vals.append(cash + qty*float(d.loc[i,"close"])); prevdate=dt
    return np.asarray(vals,float), trades

def mdd(vals):
    v=np.r_[1.0,np.asarray(vals,float)]; p=np.maximum.accumulate(v)
    return float(np.min(v/p-1.0))

def cagr(final, start_date, end_date):
    years=(end_date-start_date).days/365.2425
    return float(final**(1.0/years)-1.0) if years>0 else np.nan

def idx_on_or_after(d, date):
    i=int(np.searchsorted(d["date"].values, pd.Timestamp(date).to_datetime64(), side="left"))
    return i if i<len(d) else None

def idx_on_or_before(d, date):
    i=int(np.searchsorted(d["date"].values, pd.Timestamp(date).to_datetime64(), side="right")-1)
    return i if i>=0 else None

def eval_path(d, state, start_date, end_date, use_rate=True, fee=FEE):
    s=idx_on_or_after(d,start_date); e=idx_on_or_before(d,end_date)
    if s is None or e is None or e<=s: return None
    eq,tr=simulate(d,state,s,e,fee=fee,use_rate=use_rate)
    return {"start":d.loc[s,"date"],"end":d.loc[e,"date"],"cagr":cagr(eq[-1],d.loc[s,"date"],d.loc[e,"date"]),"mdd":mdd(eq),"final":float(eq[-1]),"trades":tr}

def rolling_cohorts(d, states):
    rows=[]; first_year=max(1997, int(d["date"].dt.year.min())+1); last_year=int(d["date"].dt.year.max())
    for sy in range(first_year,last_year+1):
        s=idx_on_or_after(d,f"{sy}-01-01")
        if s is None or s<260: continue
        for h in [3,5]:
            target=d.loc[s,"date"]+pd.DateOffset(years=h); e=idx_on_or_before(d,target)
            if e is None or e<=s or abs((target-d.loc[e,"date"]).days)>7: continue
            end=d.loc[e,"date"]
            if end <= pd.Timestamp("2009-12-31"): seg="TRAIN"
            elif d.loc[s,"date"] >= pd.Timestamp("2010-01-01") and end <= pd.Timestamp("2016-12-31"): seg="VALID"
            elif d.loc[s,"date"] >= pd.Timestamp("2017-01-01"): seg="OOS"
            else: continue
            for rule,st in states.items():
                for rf in [0,1]:
                    eq,tr=simulate(d,st,s,e,fee=FEE,use_rate=bool(rf))
                    rows.append({"rule":rule,"cash_rate":rf,"h":h,"start":d.loc[s,"date"],"end":end,"segment":seg,"cagr":cagr(eq[-1],d.loc[s,"date"],end),"mdd":mdd(eq),"trades":tr})
    return pd.DataFrame(rows)

def summarize_cohorts(R):
    S=R.groupby(["rule","cash_rate","segment"]).agg(cohorts=("cagr","size"),median_cagr=("cagr","median"),p10_cagr=("cagr",lambda x:x.quantile(.1)),worst_cagr=("cagr","min"),median_mdd=("mdd","median"),worst_mdd=("mdd","min"),median_trades=("trades","median")).reset_index()
    S["median_calmar"]=S["median_cagr"]/S["median_mdd"].abs().replace(0,np.nan)
    return S

def rank_trainvalid(S, cash_rate=1):
    q=S[(S.cash_rate==cash_rate)&(S.segment.isin(["TRAIN","VALID"]))].copy(); wide=[]
    for rule,g in q.groupby("rule"):
        if set(g.segment)!={"TRAIN","VALID"}: continue
        row={"rule":rule}
        for seg in ["TRAIN","VALID"]:
            r=g[g.segment==seg].iloc[0]
            for c in ["median_cagr","p10_cagr","worst_cagr","median_mdd","worst_mdd","median_calmar","median_trades"]: row[f"{seg.lower()}_{c}"]=float(r[c])
        wide.append(row)
    W=pd.DataFrame(wide)
    if W.empty: return W
    growth_cols=[]; balanced_cols=[]
    for seg in ["train","valid"]:
        for metric in ["median_cagr","p10_cagr","worst_cagr","median_calmar"]:
            c=f"{seg}_{metric}"; rc=f"r_{c}"; W[rc]=W[c].rank(ascending=False,pct=True,method="average"); growth_cols.append(rc); balanced_cols.append(rc)
        for metric in ["median_mdd","worst_mdd"]:
            c=f"{seg}_{metric}"; rc=f"r_{c}"; W[rc]=W[c].rank(ascending=False,pct=True,method="average"); balanced_cols.append(rc)
    W["growth_score"]=W[growth_cols].mean(axis=1); W["balanced_score"]=W[balanced_cols].mean(axis=1)
    return W.sort_values(["balanced_score","growth_score","valid_median_trades"]).reset_index(drop=True)

def contiguous_metrics(d, states):
    periods=[("TRAIN","1997-01-01","2009-12-31"),("VALID","2010-01-01","2016-12-31"),("OOS","2017-01-01",str(d["date"].max().date())),("FULL",str(max(pd.Timestamp("1997-01-01"),d.loc[260,"date"]).date()),str(d["date"].max().date()))]
    rows=[]
    for name,a,b in periods:
        for rule,st in states.items():
            for rf in [0,1]:
                r=eval_path(d,st,a,b,use_rate=bool(rf))
                if r: rows.append({"period":name,"rule":rule,"cash_rate":rf,**r})
    return pd.DataFrame(rows)

def crisis_metrics(d, states, rules):
    crises=[("Asian97_99","1997-01-01","1999-12-31"),("Dotcom00_03","2000-01-01","2003-12-31"),("GFC07_09","2007-01-01","2009-12-31"),("Euro2011","2011-01-01","2011-12-31"),("Covid2020","2020-01-01","2020-12-31"),("Bear2022","2022-01-01","2022-12-31")]
    rows=[]
    for cname,a,b in crises:
        for rule in rules:
            if rule not in states: continue
            for rf in [0,1]:
                r=eval_path(d,states[rule],a,b,use_rate=bool(rf))
                if r: rows.append({"crisis":cname,"rule":rule,"cash_rate":rf,**r})
    return pd.DataFrame(rows)

def etf_confirmation(index_d, index_states, selected_rules):
    etf=attach_rate(yf_ohlc("069500.KS",start="2002-01-01",auto_adjust=True)); ix=index_d[["date"]].copy(); rows=[]
    for rule in selected_rules:
        ix2=ix.copy(); ix2["state"]=index_states[rule]
        e=pd.merge_asof(etf.sort_values("date"),ix2.sort_values("date"),on="date",direction="backward"); st=e["state"].fillna(0).astype(int).to_numpy()
        for period,a,b in [("ETF_FULL","2007-01-01",str(e["date"].max().date())),("ETF_PREOOS","2007-01-01","2016-12-31"),("ETF_OOS","2017-01-01",str(e["date"].max().date()))]:
            for rf in [0,1]:
                r=eval_path(e,st,a,b,use_rate=bool(rf))
                if r: rows.append({"rule":rule,"period":period,"cash_rate":rf,**r})
    return pd.DataFrame(rows), etf

def cost_sensitivity(d, states, rules):
    rows=[]
    for rule in rules:
        for fee in [0.0005,0.001,0.002,0.005]:
            for period,a,b in [("TV","1997-01-01","2016-12-31"),("OOS","2017-01-01",str(d["date"].max().date()))]:
                s=idx_on_or_after(d,a); e=idx_on_or_before(d,b)
                if s is None or e is None or e<=s: continue
                eq,tr=simulate(d,states[rule],s,e,fee=fee,use_rate=True)
                rows.append({"rule":rule,"fee":fee,"period":period,"cagr":cagr(eq[-1],d.loc[s,"date"],d.loc[e,"date"]),"mdd":mdd(eq),"trades":tr})
    return pd.DataFrame(rows)

def main():
    attempts=[]; d=None; ticker_used=None
    for t in ["^KS200","KOSPI200.KS"]:
        try:
            q=yf_ohlc(t,start="1990-01-01",auto_adjust=False); attempts.append({"ticker":t,"rows":len(q),"start":str(q.date.min().date()),"end":str(q.date.max().date())})
            if len(q)>4000: d=q; ticker_used=t; break
        except Exception as ex: attempts.append({"ticker":t,"error":repr(ex)})
    if d is None: raise RuntimeError(f"No sufficiently long KOSPI200 data: {attempts}")
    d=attach_rate(d); states=rule_states(d); d.to_csv(OUT/"kospi200_index_daily.csv",index=False)
    R=rolling_cohorts(d,states); R.to_csv(OUT/"rolling_cohorts.csv",index=False)
    S=summarize_cohorts(R); S.to_csv(OUT/"cohort_summary.csv",index=False)
    rank1=rank_trainvalid(S,cash_rate=1); rank1.to_csv(OUT/"trainvalid_rank_with_cash.csv",index=False)
    rank0=rank_trainvalid(S,cash_rate=0); rank0.to_csv(OUT/"trainvalid_rank_cash0.csv",index=False)
    C=contiguous_metrics(d,states); C.to_csv(OUT/"contiguous_periods.csv",index=False)
    growth = rank1.sort_values(["growth_score","balanced_score"]).iloc[0]["rule"]; balanced = rank1.iloc[0]["rule"]; oos=S[(S.cash_rate==1)&(S.segment=="OOS")].copy(); bh="BH"
    candidates=list(dict.fromkeys([bh,str(growth),str(balanced),"MA100_C5","MA200_C1","M10","M12"]))
    crisis=crisis_metrics(d,states,candidates); crisis.to_csv(OUT/"crisis_metrics.csv",index=False)
    etf,etfraw=etf_confirmation(d,states,candidates); etf.to_csv(OUT/"kodex200_confirmation.csv",index=False); etfraw.to_csv(OUT/"kodex200_adjusted_daily.csv",index=False)
    cost=cost_sensitivity(d,states,candidates); cost.to_csv(OUT/"cost_sensitivity.csv",index=False)
    selected_oos=oos[oos.rule.isin([growth,balanced,bh])].sort_values("rule"); selected_oos.to_csv(OUT/"selected_oos.csv",index=False)
    meta={"ticker_used":ticker_used,"ticker_attempts":attempts,"data_start":str(d.date.min().date()),"data_end":str(d.date.max().date()),"rows":len(d),"fee":FEE,"cash_series":"IR3TIB01KRM156N, 45-day availability lag","train":"annual-start 3Y/5Y cohorts ending by 2009-12-31","valid":"annual-start 3Y/5Y cohorts starting >=2010 and ending by 2016-12-31","oos":"annual-start 3Y/5Y cohorts starting >=2017","growth_selected":str(growth),"balanced_selected":str(balanced),"rules":list(states.keys())}
    (OUT/"meta.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    def fmtrow(df): return df.to_string(index=False) if len(df) else "(none)"
    top=rank1.head(8)[["rule","growth_score","balanced_score","train_median_cagr","train_worst_cagr","train_worst_mdd","valid_median_cagr","valid_worst_cagr","valid_worst_mdd","valid_median_trades"]]
    rep=f"""# KOSPI200 long-horizon trend study\n\nData: {ticker_used}, {meta['data_start']} to {meta['data_end']} ({len(d)} rows)\nCash: Korea 3M interbank rate, 45-day availability lag. Also cash=0 sensitivity.\nCost: {FEE:.4%} per trade side.\nSignal: prior completed close(s); execution next market open.\n\n## Selected without OOS\nGrowth-selected: {growth}\nBalanced-selected: {balanced}\n\n## Train+Validation top rules\n{fmtrow(top)}\n\n## OOS (not used in selection)\n{fmtrow(selected_oos[['rule','cohorts','median_cagr','p10_cagr','worst_cagr','median_mdd','worst_mdd','median_trades']])}\n\n## KODEX200 adjusted-price confirmation\n{fmtrow(etf[(etf.cash_rate==1)&(etf.period.isin(['ETF_FULL','ETF_OOS']))][['rule','period','cagr','mdd','trades']])}\n"""
    (OUT/"REPORT.md").write_text(rep,encoding="utf-8")
    print("META",json.dumps(meta,ensure_ascii=False)); print("\nTOP\n",top.to_string(index=False)); print("\nOOS\n",selected_oos[["rule","cohorts","median_cagr","p10_cagr","worst_cagr","median_mdd","worst_mdd","median_trades"]].to_string(index=False)); print("\nKODEX\n",etf[(etf.cash_rate==1)&(etf.period.isin(["ETF_FULL","ETF_OOS"]))][["rule","period","cagr","mdd","trades"]].to_string(index=False)); print("\nCRISIS\n",crisis[(crisis.cash_rate==1)&(crisis.rule.isin([bh,growth,balanced]))][["crisis","rule","cagr","mdd","trades"]].to_string(index=False))

if __name__=="__main__": main()
