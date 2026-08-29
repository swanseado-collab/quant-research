#!/usr/bin/env python3
from __future__ import annotations
import io, json, math, os, requests
from pathlib import Path
import numpy as np
import pandas as pd
from research import execution_taxable_stage1 as v

OUT=Path('results/execution_user_route_kofr'); OUT.mkdir(parents=True,exist_ok=True)
# User-fixed implementation route
v.W={'spy':.15,'qqq':.15,'btc':.35,'k200':.20,'tbill':.15}
v.ROUTES=['DIRECT_US']
v.REBAL_MODES=['MONTHLY','BAND']
v.CRYPTO_TAX=[True,False]
# Base assumptions pending broker-specific fee schedule.
v.COST_BASE['DIRECT_US']={'us_fee':.0007,'kr_fee':.00015,'btc_fee':.0005,'cash_fee':.00015,'fx_fee':.0010}
v.COST_STRESS={
 'LOW': {'us_fee':.0003,'kr_fee':.00010,'btc_fee':.0005,'cash_fee':.00010,'fx_fee':.0005},
 'BASE':{'us_fee':.0007,'kr_fee':.00015,'btc_fee':.0005,'cash_fee':.00015,'fx_fee':.0010},
 'HIGH':{'us_fee':.0025,'kr_fee':.00030,'btc_fee':.0010,'cash_fee':.00030,'fx_fee':.0050},
}
# Actual target products are used as raw execution histories, so no expense-ratio rescaling.
v.PROXY_ER={'spy':.0003,'qqq':.0015,'k200':.00017}
v.TARGET_ER['DIRECT_US']={'spy':.0003,'qqq':.0015,'k200':.00017,'cash':.0005}
KOFR_ER=.0005

# Remap execution-price histories while leaving signal generation unchanged.
_orig_raw=v.raw_history
def raw_history_actual(ticker,start='2017-01-01'):
    mp={'SPY':'VOO','QQQ':'QQQM','069500.KS':'148020.KS'}
    return _orig_raw(mp.get(ticker,ticker),start)
v.raw_history=raw_history_actual

# All inactive/risk-off sleeves and the fixed 15% safe sleeve are KRW KOFR.
# Only active VOO/QQQM positions are USD holdings.
def holding_ccy(a,active):
    return 'USD' if (active and a in ('spy','qqq')) else 'KRW'
v.holding_ccy=holding_ccy

# KODEX KOFR is a KR-listed ETF; use Korean trading-fee assumption for cash sleeve transactions.
def buy_cash(sl,amount,route,cost,led):
    if amount<=0:return
    fee=amount*cost['cash_fee']; sl.cash+=max(amount-fee,0.0); sl.active=False; led['trade_cost']+=fee

def sell_cash(sl,amount,route,cost,led):
    amount=min(amount,sl.cash); fee=amount*cost['cash_fee']; sl.cash-=amount; led['trade_cost']+=fee; return max(amount-fee,0.0)
v.buy_cash=buy_cash; v.sell_cash=sell_cash

# Long-history KOFR return proxy: lagged Korean 3m money-market rate before/around ETF inception.
# 45-day availability lag prevents using unreleased monthly observations.
def add_kr_rate(d):
    u='https://fred.stlouisfed.org/graph/fredgraph.csv?id=IR3TIB01KRM156N'
    r=requests.get(u,timeout=30); r.raise_for_status(); x=pd.read_csv(io.StringIO(r.text)); x.columns=['obs_date','kr_yield']
    x['obs_date']=pd.to_datetime(x.obs_date).astype('datetime64[ns]'); x['kr_yield']=pd.to_numeric(x.kr_yield,errors='coerce'); x=x.dropna().sort_values('obs_date')
    x['available_date']=(x.obs_date+pd.Timedelta(days=45)).astype('datetime64[ns]')
    z=d.copy(); z['date']=pd.to_datetime(z.date).astype('datetime64[ns]'); z=z.sort_values('date')
    z=pd.merge_asof(z,x[['available_date','kr_yield']].sort_values('available_date'),left_on='date',right_on='available_date',direction='backward')
    z['kr_yield']=z.kr_yield.ffill().bfill(); return z.drop(columns='available_date').reset_index(drop=True)

_orig_prepare=v.prepare_data
def prepare_data():
    d,sr,qr=_orig_prepare(); d=add_kr_rate(d); return d,sr,qr
v.prepare_data=prepare_data

# KOFR-like daily accrual in KRW, after 15.4% income tax and 0.05% annual fund expense.
def cash_factor(d,route,i):
    if i==0:return 1.0,0.0
    days=max(1,(d.date.iloc[i]-d.date.iloc[i-1]).days)
    y=max(float(d.kr_yield.iloc[max(i-1,0)]),0.0)/100.0
    gross=(1+y)**(days/365.2425)-1
    net=gross*(1-v.CASH_INCOME_TAX)
    return (1+net)*math.exp(-KOFR_ER*days/365.2425),gross
v.cash_factor=cash_factor

# Rebuild sim to fix terminal currency classification. No intraday FX accrual is applied to inactive cash,
# because every inactive sleeve is KRW KOFR in this user route.
import inspect
src=inspect.getsource(v.sim)
old="""    for a in ASSETS:\n        sl=sleeves[a]\n        if a=='tbill' or not sl.active:\n            proceeds=sell_cash(sl,sl.cash,route,cost,led); sl.cash=0.\n        else:\n            proceeds=sell_qty(sl,sl.qty,px_close[a][e],route,cost,led,yr); sl.active=False\n        if holding_ccy(a,False)=='USD': usd_value+=proceeds\n        else: krw_value+=proceeds\n        final_pool_krw+=proceeds\n"""
new="""    for a in ASSETS:\n        sl=sleeves[a]; old_ccy=holding_ccy(a,sl.active)\n        if a=='tbill' or not sl.active:\n            proceeds=sell_cash(sl,sl.cash,route,cost,led); sl.cash=0.\n        else:\n            proceeds=sell_qty(sl,sl.qty,px_close[a][e],route,cost,led,yr); sl.active=False\n        if old_ccy=='USD': usd_value+=proceeds\n        else: krw_value+=proceeds\n        final_pool_krw+=proceeds\n"""
if old not in src: raise RuntimeError('terminal block not found')
src=src.replace(old,new)
ns={}; exec(src,v.__dict__,ns); v.sim=ns['sim']


def oos_only(d,notional):
    starts=d.groupby(d.date.dt.to_period('M')).head(1).index.tolist(); starts=[s for s in starts if d.date.iloc[s]>=pd.Timestamp('2018-09-01') and s>=260]
    rows=[]
    for h in [3,5]:
      for s in starts:
        e=v.m.end_idx(d,s,h)
        if e is None: continue
        sy=int(d.date.iloc[s].year); seg=v.segment(sy,h)
        if seg!='OOS': continue
        for rb in v.REBAL_MODES:
          for ct in v.CRYPTO_TAX:
            z=v.sim(d,s,e,'DIRECT_US',notional,rb,ct,v.COST_BASE['DIRECT_US'])
            rows.append({'route':'USER_VOO_QQQM_RISE_BTC_KOFR','notional':notional,'rebalance':rb,'crypto_tax':ct,'horizon':h,'start':d.date.iloc[s],'end':d.date.iloc[e],'start_year':sy,'segment':seg,**z})
    return pd.DataFrame(rows)


def fee_stress(d,notional=300_000_000.):
    starts=d.groupby(d.date.dt.to_period('M')).head(1).index.tolist(); starts=[s for s in starts if d.date.iloc[s]>=pd.Timestamp('2018-09-01') and s>=260]
    rows=[]
    for h in [3,5]:
      for s in starts:
        e=v.m.end_idx(d,s,h)
        if e is None: continue
        sy=int(d.date.iloc[s].year)
        if v.segment(sy,h)!='OOS': continue
        for name,cost in v.COST_STRESS.items():
            z=v.sim(d,s,e,'DIRECT_US',notional,'BAND',True,cost)
            rows.append({'cost_profile':name,'notional':notional,'horizon':h,'start':d.date.iloc[s],'end':d.date.iloc[e],**z})
    return pd.DataFrame(rows)


def main():
    notional=float(os.getenv('NOTIONAL','300000000'))
    d,sr,qr=v.prepare_data(); R=oos_only(d,notional); out=OUT/f'{int(notional)}'; out.mkdir(parents=True,exist_ok=True)
    R.to_csv(out/'oos_cohorts.csv',index=False)
    S=R.groupby(['rebalance','crypto_tax']).apply(v.summarize,include_groups=False).reset_index(); S.to_csv(out/'oos_summary.csv',index=False)
    if int(notional)==300_000_000:
        F=fee_stress(d,notional); F.to_csv(out/'fee_stress_cohorts.csv',index=False)
        FS=F.groupby('cost_profile').apply(v.summarize,include_groups=False).reset_index(); FS.to_csv(out/'fee_stress_summary.csv',index=False)
    else: FS=pd.DataFrame()
    meta={'allocation':v.W,'products':{'spy':'VOO','qqq':'QQQM','btc':'KRW spot BTC','k200':'RISE 200 (148020)','cash':'KODEX KOFR Rate Active Synthetic (423160)'},'signals':{'SPY':'MA250_C5','QQQ':'MA250_C3','BTC':'MA150_C3','KOSPI200':'MA100_C3'},'cash_policy':'Fixed 15% safe sleeve and every risk-off sleeve park in KRW KOFR; only active VOO/QQQM are USD.', 'expense_ratios':{'VOO':.0003,'QQQM':.0015,'RISE200':.00017,'KODEX_KOFR':.0005},'base_cost':v.COST_BASE['DIRECT_US'],'crypto_tax_scenarios':[True,False],'kofr_history_note':'Long-history return proxy uses Korean 3m money-market yield with 45-day publication lag, net of 15.4% and 0.05% ER. Actual 423160 began 2022-04-26; exact post-inception splice is a separate validation.','data_end':str(d.date.max().date()),'notional':notional,'auto_spy_rule_ignored':sr,'qqq_rule':qr}
    (out/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    print('META',json.dumps(meta,ensure_ascii=False)); print('\nOOS\n',S.to_string(index=False));
    if len(FS): print('\nFEE STRESS BAND CRYPTO TAX ON\n',FS.to_string(index=False))

if __name__=='__main__': main()
