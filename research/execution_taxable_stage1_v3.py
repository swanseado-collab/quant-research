#!/usr/bin/env python3
from pathlib import Path
import inspect, math
from research import execution_taxable_stage1 as v

# Cash held in USD may only know today's opening FX before open execution.
def cash_factor_open(d,route,i):
    if i==0: return 1.0,0.0
    days=max(1,(d.date.iloc[i]-d.date.iloc[i-1]).days)
    y=max(float(d.yk.iloc[i]),0.0)/100.0
    gross_interest=(1+y)**(days/365.2425)-1
    net_interest=gross_interest*(1-v.CASH_INCOME_TAX)
    er=v.TARGET_ER[route]['cash']; usd=(1+net_interest)*math.exp(-er*days/365.2425)
    fx=float(d.fx_open.iloc[i]/d.fx_close.iloc[i-1])
    return usd*fx,gross_interest
v.cash_factor=cash_factor_open

src=inspect.getsource(v.sim)
# Fix terminal liquidation currency classification.
old="""    for a in ASSETS:\n        sl=sleeves[a]\n        if a=='tbill' or not sl.active:\n            proceeds=sell_cash(sl,sl.cash,route,cost,led); sl.cash=0.\n        else:\n            proceeds=sell_qty(sl,sl.qty,px_close[a][e],route,cost,led,yr); sl.active=False\n        if holding_ccy(a,False)=='USD': usd_value+=proceeds\n        else: krw_value+=proceeds\n        final_pool_krw+=proceeds\n"""
new="""    for a in ASSETS:\n        sl=sleeves[a]\n        old_ccy=holding_ccy(a,sl.active)\n        if a=='tbill' or not sl.active:\n            proceeds=sell_cash(sl,sl.cash,route,cost,led); sl.cash=0.\n        else:\n            proceeds=sell_qty(sl,sl.qty,px_close[a][e],route,cost,led,yr); sl.active=False\n        if old_ccy=='USD': usd_value+=proceeds\n        else: krw_value+=proceeds\n        final_pool_krw+=proceeds\n"""
if old not in src: raise RuntimeError('terminal block not found')
src=src.replace(old,new)
# After all open executions/rebalance and distributions, mark remaining USD cash from today's open FX to close FX.
old2="""        closevals={a:sleeves[a].value(px_close[a][i] if a!='tbill' else 1.) for a in ASSETS}; total=sum(closevals.values()); vals.append(total)\n"""
new2="""        intraday_fx=float(d.fx_close.iloc[i]/d.fx_open.iloc[i]) if float(d.fx_open.iloc[i])!=0 else 1.0\n        for _a,_sl in sleeves.items():\n            if not _sl.active:\n                _sl.cash*=intraday_fx\n        closevals={a:sleeves[a].value(px_close[a][i] if a!='tbill' else 1.) for a in ASSETS}; total=sum(closevals.values()); vals.append(total)\n"""
if old2 not in src: raise RuntimeError('close valuation block not found')
src=src.replace(old2,new2)
ns={}; exec(src,v.__dict__,ns); v.sim=ns['sim']
v.OUT=Path('results/execution_taxable_stage1_v3'); v.OUT.mkdir(parents=True,exist_ok=True)
if __name__=='__main__': v.main()
