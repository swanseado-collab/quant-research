#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
from research import multi_asset_6way_kospi_trend_compare as kt
from research import btc_return_stress_6asset as stress
from research import btc_qqq_joint_robust as j
from research import spy_rule_robust_stage as spy_stage
from research import fx_regime_robust_stage as fx_stage

OUT=Path('results/final_retirement_robust_selection'); OUT.mkdir(parents=True,exist_ok=True)
KEY=j.KEY; GOAL=-.30; REG_W=j.REG_W
SPY_RULES=['MA150_C5','MA250_C5']
# Representative adversarial economic worlds; SPY-rule sensitivity doubles these to 28 model worlds.
ECON=[
 ('BASE',1.0,1.0,1.0,'HIST'),
 ('BTC75',.75,1.0,1.0,'HIST'),
 ('BTC50',.50,1.0,1.0,'HIST'),
 ('BTC25',.25,1.0,1.0,'HIST'),
 ('QQQ50',1.0,.50,1.0,'HIST'),
 ('QQQ0',1.0,0.0,1.0,'HIST'),
 ('ETH0',1.0,1.0,0.0,'HIST'),
 ('JOINT_MID',.50,.50,.50,'HIST'),
 ('JOINT_LOW',.25,0.0,0.0,'HIST'),
 ('MID_FLATFX',.50,.50,.50,'FLAT'),
 ('LOW_FLATFX',.25,0.0,0.0,'FLAT'),
 ('BASE_KRWSTRONG',1.0,1.0,1.0,'KRW_STRONG'),
 ('MID_KRWSTRONG',.50,.50,.50,'KRW_STRONG'),
 ('LOW_KRWSTRONG',.25,0.0,0.0,'KRW_STRONG'),
]
CURRENT=np.array([.10,.25,.30,.10,.20,.05])

def candidate_weights():
    arr=[]
    for spy in np.arange(.05,.301,.05):
      for qqq in np.arange(.10,.351,.05):
       for btc in np.arange(.10,.401,.05):
        for eth in [0,.05,.10]:
         for k in np.arange(0,.301,.05):
          tb=1-spy-qqq-btc-eth-k
          if .05-1e-9 <= tb <= .30+1e-9:
            arr.append([spy,qqq,btc,eth,k,round(tb,10)])
    return np.unique(np.round(np.asarray(arr,float),10),axis=0)

def years(s,e): return (pd.Timestamp(e)-pd.Timestamp(s)).days/365.2425

def drift_curve(curve,alpha,edge,dates):
    c=np.asarray(curve,float); t=(pd.to_datetime(dates)-pd.Timestamp(dates[0])).days.to_numpy(float)/365.2425
    return c*np.exp(-(1-alpha)*edge*t)

def calibrate(cache):
    b=[]; q=[]; e=[]; f=[]
    for x in cache:
      if x['segment'] not in ('TRAIN','VALID'): continue
      y=years(x['start'],x['end']); c=x['curves']
      b.append(np.log(c[-1,2]/x['tbill'][-1])/y)
      q.append(np.log(c[-1,1]/c[-1,0])/y)
      e.append(np.log(c[-1,3]/x['tbill'][-1])/y)
      f.append(np.log(x['fx'][-1]/x['fx'][0])/y)
    return {'btc':float(np.median(b)),'qqq':float(np.median(q)),'eth':float(np.median(e)),'fx':float(np.median(f))}

def fx_adjust(fx,dates,hist_edge,regime):
    if regime=='HIST': return np.asarray(fx,float).copy()
    target={'FLAT':0.0,'KRW_STRONG':-0.02}[regime]
    f=np.asarray(fx,float); t=(pd.to_datetime(dates)-pd.Timestamp(dates[0])).days.to_numpy(float)/365.2425
    return f*np.exp((target-hist_edge)*t)

def build_rule_cache(d0,rule):
    d=spy_stage.override_spy(d0,rule); return d,stress.make_cache(d)

def evaluate_world(cache,W,edges,ba,qa,ea,fxr):
    tvc=[]; tvm=[]; ooc=[]; oom=[]
    for x in cache:
      c=x['curves'].copy()
      c[:,2]=drift_curve(c[:,2],ba,edges['btc'],x['dates'])
      c[:,1]=drift_curve(c[:,1],qa,edges['qqq'],x['dates'])
      c[:,3]=drift_curve(c[:,3],ea,edges['eth'],x['dates'])
      fx=fx_adjust(x['fx'],x['dates'],edges['fx'],fxr)
      _,_,fk,mk,_=stress.port_matrix_krw_fast(c,W,x['dates'],fx); y=years(x['start'],x['end']); cg=np.power(np.maximum(fk,1e-15),1/y)-1
      if x['segment'] in ('TRAIN','VALID'): tvc.append(cg); tvm.append(mk)
      else: ooc.append(cg); oom.append(mk)
    tvc=np.asarray(tvc); tvm=np.asarray(tvm); ooc=np.asarray(ooc); oom=np.asarray(oom)
    return {'median_cagr':np.median(tvc,0),'p10_cagr':np.quantile(tvc,.1,axis=0),'worst_cagr':np.min(tvc,0),'worst_mdd':np.min(tvm,0)},(ooc,oom)

def select_robust(worlds,W):
    feasible=np.ones(len(W),bool); bench={}
    for name,z in worlds.items():
      feasible &= z['worst_mdd']>=GOAL
      ok=z['worst_mdd']>=GOAL; bench[name]={c:float(np.max(z[c][ok])) for c in REG_W}
    ids=np.flatnonzero(feasible)
    if not len(ids): raise RuntimeError('No candidate satisfies MDD goal in every world')
    mx=np.full(len(ids),-np.inf); av=np.zeros(len(ids)); mxmed=np.full(len(ids),-np.inf)
    for name,z in worlds.items():
      comp=np.zeros(len(ids)); medreg=bench[name]['median_cagr']-z['median_cagr'][ids]
      for c,w in REG_W.items(): comp+=w*(bench[name][c]-z[c][ids])
      mx=np.maximum(mx,comp); av+=comp/len(worlds); mxmed=np.maximum(mxmed,medreg)
    order=np.lexsort((av,mx)); rankrows=[]
    for r,pos in enumerate(order[:30]):
      ix=int(ids[pos]); row={'rank':r+1,'candidate_index':ix,'max_composite_regret':float(mx[pos]),'avg_composite_regret':float(av[pos]),'max_median_cagr_regret':float(mxmed[pos])}; row.update({KEY[i]:float(W[ix,i]) for i in range(6)}); rankrows.append(row)
    return pd.DataFrame(rankrows),bench

def oos_for_candidate(ix,W,oos_store):
    rows=[]
    for name,(c,m,meta) in oos_store.items():
      cg=c[:,ix]; mm=m[:,ix]; rows.append({**meta,'world':name,'median_cagr':float(np.median(cg)),'p10_cagr':float(np.quantile(cg,.1)),'worst_cagr':float(np.min(cg)),'median_mdd':float(np.median(mm)),'worst_mdd':float(np.min(mm)),**{KEY[i]:float(W[ix,i]) for i in range(6)}})
    return pd.DataFrame(rows)

def find_index(W,w):
    z=np.where(np.all(np.isclose(W,w[None,:]),axis=1))[0]; return int(z[0]) if len(z) else None

def main():
    W=candidate_weights(); d0,sr_auto,qr=kt.prepare(j.K_RULE); worlds={}; store={}; validations={}; edge_meta={}
    for rule in SPY_RULES:
      d,cache=build_rule_cache(d0,rule); validations[rule]=stress.validate_fast(d,W); edges=calibrate(cache); edge_meta[rule]=edges
      for econ,ba,qa,ea,fxr in ECON:
        z,o=evaluate_world(cache,W,edges,ba,qa,ea,fxr); name=f'{rule}|{econ}'; worlds[name]=z; store[name]=(o[0],o[1],{'spy_rule':rule,'econ':econ,'btc_alpha':ba,'qqq_alpha':qa,'eth_alpha':ea,'fx_regime':fxr})
    TOP,bench=select_robust(worlds,W); ix=int(TOP.iloc[0].candidate_index); OO=oos_for_candidate(ix,W,store)
    ci=find_index(W,CURRENT); CUR=oos_for_candidate(ci,W,store) if ci is not None else pd.DataFrame()
    # Aggregate adversarial OOS summaries across model worlds.
    def agg(df,label):
      return {'profile':label,'min_world_median_cagr':float(df.median_cagr.min()),'min_world_p10_cagr':float(df.p10_cagr.min()),'min_world_worst_cagr':float(df.worst_cagr.min()),'worst_world_mdd':float(df.worst_mdd.min()),'max_world_median_cagr':float(df.median_cagr.max())}
    AGG=pd.DataFrame([agg(OO,'final_robust'),agg(CUR,'current_v1')])
    # Nearby plateau: all candidates within +0.005 absolute max composite regret of best, top 30 already retained for readability.
    best=float(TOP.iloc[0].max_composite_regret); TOP['within_50bp_regret_plateau']=TOP.max_composite_regret<=best+.005
    TOP.to_csv(OUT/'top_robust_candidates.csv',index=False); OO.to_csv(OUT/'selected_oos_worlds.csv',index=False); CUR.to_csv(OUT/'current_v1_oos_worlds.csv',index=False); AGG.to_csv(OUT/'adversarial_oos_summary.csv',index=False)
    meta={'goal_mdd':GOAL,'candidate_count':len(W),'economic_worlds':len(ECON),'model_worlds':len(worlds),'econ_definitions':ECON,'spy_rules':SPY_RULES,'auto_selected_spy_rule':sr_auto,'qqq_rule':qr,'k_rule':j.K_RULE,'edges_by_spy_rule':edge_meta,'engine_validation':validations,'data_end':str(d0.date.max().date()),'regret_weights':REG_W}
    (OUT/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)); (OUT/'REPORT.md').write_text('# Final retirement robust selection\n\n```json\n'+json.dumps(meta,ensure_ascii=False,indent=2)+'\n```\n\n## Top robust candidates\n'+TOP.to_markdown(index=False)+'\n\n## Adversarial OOS aggregate\n'+AGG.to_markdown(index=False)+'\n\n## Selected OOS worlds\n'+OO.to_markdown(index=False)+'\n\n## Current V1 OOS worlds\n'+CUR.to_markdown(index=False))
    print('META',json.dumps(meta,ensure_ascii=False)); print('\nTOP\n',TOP.to_string(index=False)); print('\nAGG\n',AGG.to_string(index=False)); print('\nSELECTED OOS\n',OO.to_string(index=False)); print('\nCURRENT V1 OOS\n',CUR.to_string(index=False))
if __name__=='__main__': main()
