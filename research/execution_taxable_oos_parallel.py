#!/usr/bin/env python3
import os, json
from pathlib import Path
import pandas as pd
from research import execution_taxable_stage1_v3 as patched
from research import execution_taxable_stage1 as v

route=os.environ['ROUTE']; notional=float(os.environ['NOTIONAL'])
OUT=Path(f"results/execution_taxable_oos_parallel/{route}_{int(notional)}"); OUT.mkdir(parents=True,exist_ok=True)

def main():
    d,_,_=v.prepare_data(); starts=d.groupby(d.date.dt.to_period('M')).head(1).index.tolist(); starts=[s for s in starts if d.date.iloc[s]>=pd.Timestamp('2018-09-01') and s>=260]
    rows=[]
    for h in [3,5]:
      for s in starts:
        e=v.m.end_idx(d,s,h)
        if e is None: continue
        sy=int(d.date.iloc[s].year); seg=v.segment(sy,h)
        if seg!='OOS': continue
        for rb in v.REBAL_MODES:
          for ct in v.CRYPTO_TAX:
            z=v.sim(d,s,e,route,notional,rb,ct,v.COST_BASE[route]); rows.append({'route':route,'notional':notional,'rebalance':rb,'crypto_tax':ct,'horizon':h,'start':d.date.iloc[s],'end':d.date.iloc[e],**z})
    R=pd.DataFrame(rows); R.to_csv(OUT/'oos_cohorts.csv',index=False)
    S=R.groupby(['rebalance','crypto_tax']).apply(v.summarize,include_groups=False).reset_index(); S.to_csv(OUT/'oos_summary.csv',index=False)
    print('CONFIG',route,notional); print(S.to_string(index=False))
if __name__=='__main__': main()
