#!/usr/bin/env python3
from __future__ import annotations
import inspect, io, math, json, requests
from pathlib import Path
import numpy as np
import pandas as pd
from research import execution_taxable_stage1 as v

OUT=Path('results/execution_local_riskoff_stage2'); OUT.mkdir(parents=True,exist_ok=True)
LOCAL_CASH_ER=.0005
LOCAL_CASH_FEE=.00015

def holding_ccy_local(a,active):
    if a in ('btc','k200'): return 'KRW'
    return 'USD'
v.holding_ccy=holding_ccy_local

def add_kr_rate(d):
    u='https://fred.stlouisfed.org/graph/fredgraph.csv?id=IR3TIB01KRM156N'
    r=requests.get(u,timeout=30); r.raise_for_status(); x=pd.read_csv(io.StringIO(r.text)); x.columns=['obs_date','kr_yield']
    x.obs_date=pd.to_datetime(x.obs_date).astype('datetime64[ns]'); x.kr_yield=pd.to_numeric(x.kr_yield,errors='coerce'); x=x.dropna().sort_values('obs_date')
    x['available_date']=(x.obs_date+pd.Timedelta(days=45)).astype('datetime64[ns]')
    z=d.copy(); z['date']=pd.to_datetime(z.date).astype('datetime64[ns]')
    z=pd.merge_asof(z.sort_values('date'),x[['available_date','kr_yield']].sort_values('available_date'),left_on='date',right_on='available_date',direction='backward')
    if z.kr_yield.isna().any(): z.kr_yield=z.kr_yield.ffill().bfill()
    return z.drop(columns='available_date')

orig_prepare=v.prepare_data
def prepare_data_local():
    d,sr,qr=orig_prepare(); return add_kr_rate(d),sr,qr
v.prepare_data=prepare_data_local

def local_cash_factor(d,i):
    if i==0:return 1.0,0.0
    days=max(1,(d.date.iloc[i]-d.date.iloc[i-1]).days); y=max(float(d.kr_yield.iloc[i]),0.0)/100.0
    gross=(1+y)**(days/365.2425)-1; net=gross*(1-v.CASH_INCOME_TAX)
    return (1+net)*math.exp(-LOCAL_CASH_ER*days/365.2425),gross

def buy_cash_local(sl,amount,route,cost,led):
    if amount<=0:return
    fee_rate=LOCAL_CASH_FEE if sl.name in ('btc','k200') else cost['cash_fee']
    fee=amount*fee_rate; sl.cash+=max(amount-fee,0.0); sl.active=False; led['trade_cost']+=fee
v.buy_cash=buy_cash_local

def sell_cash_local(sl,amount,route,cost,led):
    amount=min(amount,sl.cash); fee_rate=LOCAL_CASH_FEE if sl.name in ('btc','k200') else cost['cash_fee']; fee=amount*fee_rate; sl.cash-=amount; led['trade_cost']+=fee; return max(amount-fee,0.0)
v.sell_cash=sell_cash_local

src=inspect.getsource(v.sim)
src=src.replace("""            f,gross_interest=cash_factor(d,route,i)\n            for a,sl in sleeves.items():\n                if not sl.active:\n                    before=sl.cash; sl.cash*=f\n                    # approximate gross USD interest component as financial income, not FX gain.\n                    led['financial_income'][yr]=led['financial_income'].get(yr,0.0)+before*gross_interest\n""","""            for a,sl in sleeves.items():\n                if not sl.active:\n                    before=sl.cash\n                    if a in ('btc','k200'):\n                        f,gross_interest=local_cash_factor(d,i)\n                    else:\n                        days=max(1,(d.date.iloc[i]-d.date.iloc[i-1]).days); y=max(float(d.yk.iloc[i]),0.0)/100.0\n                        gross_interest=(1+y)**(days/365.2425)-1; net_interest=gross_interest*(1-CASH_INCOME_TAX); er=TARGET_ER[route]['cash']\n                        f=(1+net_interest)*math.exp(-er*days/365.2425)*float(d.fx_open.iloc[i]/d.fx_close.iloc[i-1])\n                    sl.cash*=f\n                    led['financial_income'][yr]=led['financial_income'].get(yr,0.0)+before*gross_interest\n""")
oldclose="""        closevals={a:sleeves[a].value(px_close[a][i] if a!='tbill' else 1.) for a in ASSETS}; total=sum(closevals.values()); vals.append(total)\n"""
newclose="""        intraday_fx=float(d.fx_close.iloc[i]/d.fx_open.iloc[i]) if float(d.fx_open.iloc[i])!=0 else 1.0\n        for _a,_sl in sleeves.items():\n            if (not _sl.active) and holding_ccy(_a,False)=='USD': _sl.cash*=intraday_fx\n        closevals={a:sleeves[a].value(px_close[a][i] if a!='tbill' else 1.) for a in ASSETS}; total=sum(closevals.values()); vals.append(total)\n"""
if oldclose not in src: raise RuntimeError('close block not found')
src=src.replace(oldclose,newclose)
oldterm="""    for a in ASSETS:\n        sl=sleeves[a]\n        if a=='tbill' or not sl.active:\n            proceeds=sell_cash(sl,sl.cash,route,cost,led); sl.cash=0.\n        else:\n            proceeds=sell_qty(sl,sl.qty,px_close[a][e],route,cost,led,yr); sl.active=False\n        if holding_ccy(a,False)=='USD': usd_value+=proceeds\n        else: krw_value+=proceeds\n        final_pool_krw+=proceeds\n"""
newterm="""    for a in ASSETS:\n        sl=sleeves[a]; old_ccy=holding_ccy(a,sl.active)\n        if a=='tbill' or not sl.active:\n            proceeds=sell_cash(sl,sl.cash,route,cost,led); sl.cash=0.\n        else:\n            proceeds=sell_qty(sl,sl.qty,px_close[a][e],route,cost,led,yr); sl.active=False\n        if old_ccy=='USD': usd_value+=proceeds\n        else: krw_value+=proceeds\n        final_pool_krw+=proceeds\n"""
if oldterm not in src: raise RuntimeError('terminal block not found')
src=src.replace(oldterm,newterm)
ns={}; exec(src,v.__dict__|{'local_cash_factor':local_cash_factor},ns); v.sim=ns['sim']
v.OUT=OUT
v.ROUTES=['DIRECT_US']; v.SIZES=[100_000_000.,300_000_000.,1_000_000_000.]

if __name__=='__main__':
    v.main()
    meta_path=OUT/'meta.json'; meta=json.loads(meta_path.read_text()); meta['riskoff_cash_policy']='BTC/KOSPI inactive cash in KRW 3m-rate proxy (45-day lag); SPY/QQQ/base T-bill inactive cash in USD SGOV proxy'; meta['kr_rate_series']='IR3TIB01KRM156N'; meta['local_cash_er']=LOCAL_CASH_ER; meta['local_cash_fee']=LOCAL_CASH_FEE; meta_path.write_text(json.dumps(meta,ensure_ascii=False,indent=2))
