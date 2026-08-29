#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
from research import multi_asset_6way_kospi_trend_compare as kt
from research import btc_return_stress_6asset as stress
from research import btc_qqq_joint_robust as j

OUT=Path('results/btc_qqq_joint_robust_kospi_expand'); OUT.mkdir(parents=True,exist_ok=True)

def expanded_weights():
    # Keep the original broad universe and add a focused 5pp refinement allowing KOSPI up to 40%.
    arr=[x.tolist() for x in stress.candidate_weights()]
    for spy in np.arange(.05,.301,.05):
      for qqq in np.arange(.10,.351,.05):
       for btc in np.arange(.10,.451,.05):
        for eth in [.05,.10]:
         for k in np.arange(.10,.401,.05):
          tb=1-spy-qqq-btc-eth-k
          if .05-1e-9 <= tb <= .30+1e-9:
            arr.append([spy,qqq,btc,eth,k,round(tb,10)])
    return np.unique(np.round(np.asarray(arr,float),10),axis=0)

def main():
    W=expanded_weights(); d,sr,qr=kt.prepare(j.K_RULE); err=stress.validate_fast(d,W); cache=stress.make_cache(d)
    be,qe=j.calibrate_edges(cache); S,O=j.evaluate(cache,W,be,qe)
    old=j.GOALS; j.GOALS=[-.30]
    sel,reg=j.select_robust(S,W); oo=j.oos_selected(sel,O); opt=j.scenario_optima(S)
    j.GOALS=old
    sel.to_csv(OUT/'robust_selected.csv',index=False); reg.to_csv(OUT/'robust_trainvalid_by_scenario.csv',index=False); oo.to_csv(OUT/'robust_oos_by_scenario.csv',index=False); opt.to_csv(OUT/'scenario_specific_optima.csv',index=False)
    meta={'candidate_count':len(W),'kospi_max':.40,'goal':-.30,'btc_alphas':j.BTC_ALPHAS,'qqq_alphas':j.QQQ_ALPHAS,'spy_rule':sr,'qqq_rule':qr,'k_rule':j.K_RULE,'btc_edge':be,'qqq_edge':qe,'engine_validation':err,'data_end':str(d.date.max().date())}
    (OUT/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    (OUT/'REPORT.md').write_text('# Expanded KOSPI robust refinement\n\n```json\n'+json.dumps(meta,ensure_ascii=False,indent=2)+'\n```\n\n## Robust\n'+sel.to_markdown(index=False)+'\n\n## OOS\n'+oo.to_markdown(index=False))
    print('META',json.dumps(meta,ensure_ascii=False)); print('\nROBUST\n',sel.to_string(index=False)); print('\nOOS\n',oo.to_string(index=False))
if __name__=='__main__': main()
