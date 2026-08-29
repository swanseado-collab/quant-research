#!/usr/bin/env python3
from research import execution_taxable_stage1_v3 as patched
from research import execution_taxable_stage1 as v
import pandas as pd

def main():
    d,_,_=v.prepare_data(); starts=d.groupby(d.date.dt.to_period('M')).head(1).index.tolist(); s=next(x for x in starts if d.date.iloc[x]>=pd.Timestamp('2021-01-01') and x>=260); e=v.m.end_idx(d,s,3)
    for route in v.ROUTES:
      for rb in v.REBAL_MODES:
        z=v.sim(d,s,e,route,300_000_000.,rb,True,v.COST_BASE[route])
        print(route,rb,{k:round(z[k],6) if isinstance(z[k],float) else z[k] for k in ['cagr_liquidated','cagr_mtm','mdd','tax_paid','trade_cost','fx_cost','max_annual_financial_income','rebalances']})
if __name__=='__main__': main()
