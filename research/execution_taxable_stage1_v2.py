#!/usr/bin/env python3
from pathlib import Path
import inspect
from research import execution_taxable_stage1 as v

# Patch the already-reviewed V1 engine without duplicating the full research file:
# final liquidation must classify proceeds by the currency of the holding BEFORE liquidation.
src=inspect.getsource(v.sim)
old="""    for a in ASSETS:\n        sl=sleeves[a]\n        if a=='tbill' or not sl.active:\n            proceeds=sell_cash(sl,sl.cash,route,cost,led); sl.cash=0.\n        else:\n            proceeds=sell_qty(sl,sl.qty,px_close[a][e],route,cost,led,yr); sl.active=False\n        if holding_ccy(a,False)=='USD': usd_value+=proceeds\n        else: krw_value+=proceeds\n        final_pool_krw+=proceeds\n"""
new="""    for a in ASSETS:\n        sl=sleeves[a]\n        old_ccy=holding_ccy(a,sl.active)\n        if a=='tbill' or not sl.active:\n            proceeds=sell_cash(sl,sl.cash,route,cost,led); sl.cash=0.\n        else:\n            proceeds=sell_qty(sl,sl.qty,px_close[a][e],route,cost,led,yr); sl.active=False\n        if old_ccy=='USD': usd_value+=proceeds\n        else: krw_value+=proceeds\n        final_pool_krw+=proceeds\n"""
if old not in src:
    raise RuntimeError('Expected terminal-FX block not found')
ns={}
exec(src.replace(old,new),v.__dict__,ns)
v.sim=ns['sim']
v.OUT=Path('results/execution_taxable_stage1_v2'); v.OUT.mkdir(parents=True,exist_ok=True)

if __name__=='__main__':
    v.main()
