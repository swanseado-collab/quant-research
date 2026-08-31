#!/usr/bin/env python3
from __future__ import annotations
import json, io, requests
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

OUT=Path('results/tqqq_tactical_krw_execution'); OUT.mkdir(parents=True,exist_ok=True)
# Fixed rule from prior study; no re-optimization here.
LOOKBACK=60; LEVELS=[-.25,-.30,-.35]; TP=.10
US_FEE=.0005
FX_COSTS=[.0005,.0010,.0025]  # each conversion
SLEEVE_CAPITALS=[5_000_000,10_000_000,15_000_000,30_000_000,50_000_000,100_000_000]
PERIODS={'TRAIN':('2010-03-01','2015-12-31'),'VALID':('2016-01-01','2020-12-31'),'OOS':('2021-01-01',None)}

def yf_ohlc(t):
    x=yf.download(t,start='2010-01-01',auto_adjust=False,progress=False)
    if isinstance(x.columns,pd.MultiIndex):x.columns=x.columns.get_level_values(0)
    idx=pd.to_datetime(x.index); idx=idx.tz_localize(None) if getattr(idx,'tz',None) else idx
    ratio=(x['Adj Close']/x['Close']).replace([np.inf,-np.inf],np.nan).ffill().bfill()
    d=pd.DataFrame(index=idx)
    for c in ['Open','High','Low','Close']:d[c.lower()]=np.asarray(x[c]*ratio,float)
    return d.dropna().sort_index()

def get_rates():
    u='https://fred.stlouisfed.org/graph/fredgraph.csv?id=IR3TIB01KRM156N'
    r=requests.get(u,timeout=30);r.raise_for_status();d=pd.read_csv(io.StringIO(r.text))
    d.columns=['date','rate'];d['date']=pd.to_datetime(d.date);d['rate']=pd.to_numeric(d.rate,errors='coerce')/100
    # publication-safe approximation: use prior month's observed rate
    d['rate']=d.rate.shift(1)
    return d.dropna().set_index('date').rate

def align():
    q=yf_ohlc('QQQ');t=yf_ohlc('TQQQ');fx=yf_ohlc('KRW=X').close
    idx=t.index.intersection(q.index); t=t.loc[idx].copy();q=q.loc[idx];
    ma=q.close.rolling(250,min_periods=250).mean(); above=q.close>ma;reg=above.rolling(3,min_periods=3).sum().eq(3).shift(1).fillna(False)
    t['regime']=reg.astype(bool)
    rh=t.close.rolling(LOOKBACK,min_periods=LOOKBACK).max();t['dd_prev']=(t.close/rh-1).shift(1)
    # Previous completed FX close only for execution and valuation; no same-day FX lookahead.
    fx=fx.reindex(t.index.union(fx.index)).sort_index().ffill().reindex(t.index).shift(1)
    rates=get_rates(); rates=rates.reindex(t.index.union(rates.index)).sort_index().ffill().reindex(t.index).fillna(0)
    t['fx']=fx;t['kr_rate']=rates
    return t.dropna()

def grow_cash(cash,annual):
    # after 15.4% tax on KOFR-like interest
    net=max(float(annual),0.0)*(1-.154)
    return cash*((1+net)**(1/252))

def sim(d,initial,fx_cost,tax_mode,start,end=None):
    z=d[d.index>=pd.Timestamp(start)]
    if end:z=z[z.index<=pd.Timestamp(end)]
    krw=float(initial); shares=0.; usd_basis_krw=0.; stage=0; armed=True; tranche=0.; entry=None
    realized_by_year={}; tax_paid=0.; fx_paid=0.; trade_cost_krw=0.; trades=[]; eq=[]
    lastyear=None
    def paytax(y,krw_now):
        nonlocal tax_paid
        g=realized_by_year.get(y,0.0)
        if tax_mode=='marginal22': taxable=max(g,0.0)
        elif tax_mode=='deduct250': taxable=max(g-2_500_000,0.0)
        else: taxable=0.0
        tax=.22*taxable; tax=min(tax,krw_now); tax_paid+=tax
        return krw_now-tax
    for dt,row in z.iterrows():
        y=dt.year
        if lastyear is not None and y!=lastyear: krw=paytax(lastyear,krw)
        lastyear=y
        krw=grow_cash(krw,row.kr_rate)
        o,h,c,fx=float(row.open),float(row.high),float(row.close),float(row.fx)
        # regime OFF forces exit at open
        if shares>0 and not bool(row.regime):
            usd_gross=shares*o; usd_net=usd_gross*(1-US_FEE); trade_cost_krw+=usd_gross*US_FEE*fx
            proceeds=usd_net*fx*(1-fx_cost); fx_paid+=usd_net*fx*fx_cost
            gain=proceeds-usd_basis_krw;realized_by_year[y]=realized_by_year.get(y,0)+gain
            ret=proceeds/usd_basis_krw-1;krw+=proceeds;trades.append((entry,dt,ret,stage,'REGIME'))
            shares=0.;usd_basis_krw=0.;stage=0;armed=False;tranche=0.;entry=None
        # TP limit
        if shares>0:
            # avg USD purchase price reconstructed approximately from KRW basis and entry FX is not stable; maintain USD avg via shares basis
            # usd_cost embedded below as attribute-equivalent local variable via basis USD is needed, track separately outside? handled by implied market target using purchase dollars.
            pass
        # We need persistent USD cost for exact TP; initialize lazily on locals map
        if not hasattr(sim,'_dummy'): sim._dummy=True
        # use separate dynamic state initialized via closure-style variable impossible here; replaced below in second pass
        eq.append((dt,krw+shares*c*fx))
    raise RuntimeError('placeholder')

def simulate(d,initial,fx_cost,tax_mode,start,end=None):
    z=d[d.index>=pd.Timestamp(start)]
    if end:z=z[z.index<=pd.Timestamp(end)]
    krw=float(initial);shares=0.;usd_cost=0.;krw_basis=0.;stage=0;armed=True;tranche=0.;entry=None
    realized={};tax_paid=0.;fx_paid=0.;trade_cost=0.;trades=[];curve=[];lastyear=None
    for dt,row in z.iterrows():
        y=dt.year
        if lastyear is not None and y!=lastyear:
            g=realized.get(lastyear,0.0);ded=2_500_000 if tax_mode=='deduct250' else 0.; taxable=max(g-ded,0.) if tax_mode!='none' else 0.;tax=.22*taxable;tax=min(tax,krw);krw-=tax;tax_paid+=tax
        lastyear=y
        krw=grow_cash(krw,row.kr_rate)
        o,h,c,fx=float(row.open),float(row.high),float(row.close),float(row.fx)
        def exitpos(fill,reason):
            nonlocal krw,shares,usd_cost,krw_basis,stage,armed,tranche,entry,fx_paid,trade_cost
            usd_gross=shares*fill; fee=usd_gross*US_FEE;trade_cost+=fee*fx;usd_net=usd_gross-fee
            proceeds=usd_net*fx*(1-fx_cost);fx_paid+=usd_net*fx*fx_cost
            gain=proceeds-krw_basis;realized[y]=realized.get(y,0.)+gain;ret=proceeds/krw_basis-1 if krw_basis else np.nan
            krw+=proceeds;trades.append((entry,dt,ret,stage,reason));shares=0.;usd_cost=0.;krw_basis=0.;stage=0;armed=False;tranche=0.;entry=None
        if shares>0 and not bool(row.regime):exitpos(o,'REGIME')
        if shares>0:
            avg=usd_cost/shares;tgt=avg*(1+TP)
            if h>=tgt:exitpos(max(o,tgt),'TP')
        dd=float(row.dd_prev) if pd.notna(row.dd_prev) else np.nan
        if shares==0 and pd.notna(dd) and dd>-.05:armed=True
        if pd.notna(dd) and bool(row.regime):
            target=sum(dd<=lv for lv in LEVELS)
            if shares==0 and not armed:target=0
            while stage<target and krw>1:
                if stage==0:tranche=krw/len(LEVELS)
                invest=min(krw,tranche);buy_rate=fx*(1+fx_cost);usd=invest/buy_rate;fx_paid+=usd*fx*fx_cost
                fee=usd*US_FEE;trade_cost+=fee*fx;usd_net=usd-fee;sh=usd_net/o
                krw-=invest;shares+=sh;usd_cost+=usd;krw_basis+=invest;stage+=1
                if entry is None:entry=dt
        curve.append((dt,krw+shares*c*fx))
    if shares>0:
        dt=z.index[-1];row=z.iloc[-1];y=dt.year;fx=float(row.fx);fill=float(row.close)
        usd_gross=shares*fill;fee=usd_gross*US_FEE;trade_cost+=fee*fx;usd_net=usd_gross-fee;proceeds=usd_net*fx*(1-fx_cost);fx_paid+=usd_net*fx*fx_cost
        gain=proceeds-krw_basis;realized[y]=realized.get(y,0.)+gain;ret=proceeds/krw_basis-1;krw+=proceeds;trades.append((entry,dt,ret,stage,'END'));curve[-1]=(dt,krw);shares=0
    if lastyear is not None:
        g=realized.get(lastyear,0.);ded=2_500_000 if tax_mode=='deduct250' else 0.;taxable=max(g-ded,0.) if tax_mode!='none' else 0.;tax=.22*taxable;tax=min(tax,krw);krw-=tax;tax_paid+=tax;curve[-1]=(curve[-1][0],krw)
    e=pd.Series(dict(curve));yrs=(e.index[-1]-e.index[0]).days/365.25;cagr=(e.iloc[-1]/initial)**(1/yrs)-1;mdd=(e/e.cummax()-1).min()
    td=pd.DataFrame(trades,columns=['entry','exit','trade_ret','stages','reason'])
    return {'final':krw,'cagr':cagr,'mdd':mdd,'trades':len(td),'win':float((td.trade_ret>0).mean()) if len(td) else np.nan,'worst_trade':float(td.trade_ret.min()) if len(td) else np.nan,'tax_paid':tax_paid,'fx_cost':fx_paid,'trade_cost':trade_cost,'trade_df':td}

def kofr_only(d,initial,start,end=None):
    z=d[d.index>=pd.Timestamp(start)];z=z if end is None else z[z.index<=pd.Timestamp(end)];v=float(initial);curve=[]
    for dt,row in z.iterrows():v=grow_cash(v,row.kr_rate);curve.append((dt,v))
    e=pd.Series(dict(curve));yrs=(e.index[-1]-e.index[0]).days/365.25;return {'final':v,'cagr':(v/initial)**(1/yrs)-1,'mdd':0.0}

def main():
    d=align();rows=[];trade_exports=0
    for cap in SLEEVE_CAPITALS:
      for fx in FX_COSTS:
       for tax in ['marginal22','deduct250','none']:
        for per,(s,e) in PERIODS.items():
            a=simulate(d,cap,fx,tax,s,e);k=kofr_only(d,cap,s,e)
            rows.append({'capital':cap,'fx_each_way':fx,'tax_mode':tax,'period':per,**{f'tac_{x}':a[x] for x in ['final','cagr','mdd','trades','win','worst_trade','tax_paid','fx_cost','trade_cost']},'kofr_final':k['final'],'kofr_cagr':k['cagr'],'cagr_edge':a['cagr']-k['cagr'],'wealth_edge_pct':a['final']/k['final']-1})
            if cap==10_000_000 and fx==.001 and tax=='marginal22':a['trade_df'].to_csv(OUT/f'trades_{per}.csv',index=False)
    R=pd.DataFrame(rows);R.to_csv(OUT/'execution_results.csv',index=False)
    core=R[(R.capital==10_000_000)&(R.fx_each_way==.001)&(R.tax_mode=='marginal22')].copy();
    current={'data_end':str(d.index[-1].date()),'qqq_regime_on':bool(d.regime.iloc[-1]),'tqqq_dd_from_60d_high':float(d.close.iloc[-1]/d.close.rolling(60).max().iloc[-1]-1),'entry_levels':LEVELS,'would_enter_next_open':bool(d.regime.iloc[-1] and any(float(d.dd_prev.iloc[-1])<=x for x in LEVELS))}
    meta={'rule':'QQQ MA250_C3 ON; TQQQ 60d-high DD entries -25/-30/-35, 3 equal tranches; avg-cost +10% TP; regime-off exit','us_fee_each_side':US_FEE,'fx_scenarios_each_conversion':FX_COSTS,'current':current,'rate_proxy':'FRED IR3TIB01KRM156N lagged one month, 15.4% interest tax'}
    (OUT/'meta.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    print('META',json.dumps(meta,ensure_ascii=False));print('\nCORE 10M, FX10BP, MARGINAL22\n',core.to_string(index=False));print('\nOOS SENSITIVITY\n',R[(R.period=='OOS')&(R.tax_mode=='marginal22')].to_string(index=False))

if __name__=='__main__':main()
