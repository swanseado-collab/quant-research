#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
from research import multi_asset_5way_allocation as m
from research import multi_asset_6way_kospi200_v2 as base
from research import multi_asset_6way_kospi_trend_compare as kt
from research import btc_return_stress_6asset as stress
from research import btc_qqq_joint_robust as j
from research import spy_rule_robust_stage as spy_stage
from research import gold_rule_pre2018 as gold_rules

OUT=Path('results/final_retirement_robust_with_gold'); OUT.mkdir(parents=True,exist_ok=True)
KEY=['spy','qqq','btc','eth','kospi200','gold','tbill']; GOAL=-.30; REG_W=j.REG_W
SPY_RULES=['MA150_C5','MA250_C5']; GOLD_RULE='MA200_C5'
# Gold alpha is sleeve excess-return retention vs T-bill.
ECON=[
 ('BASE',1.0,1.0,1.0,1.0,'HIST'),
 ('GOLD50',1.0,1.0,1.0,.50,'HIST'),
 ('GOLD0',1.0,1.0,1.0,0.0,'HIST'),
 ('BTC75',.75,1.0,1.0,1.0,'HIST'),
 ('BTC50',.50,1.0,1.0,1.0,'HIST'),
 ('BTC25',.25,1.0,1.0,1.0,'HIST'),
 ('QQQ50',1.0,.50,1.0,1.0,'HIST'),
 ('QQQ0',1.0,0.0,1.0,1.0,'HIST'),
 ('ETH0',1.0,1.0,0.0,1.0,'HIST'),
 ('JOINT_MID',.50,.50,.50,.50,'HIST'),
 ('JOINT_LOW',.25,0.0,0.0,0.0,'HIST'),
 ('MID_FLATFX',.50,.50,.50,.50,'FLAT'),
 ('LOW_FLATFX',.25,0.0,0.0,0.0,'FLAT'),
 ('BASE_KRWSTRONG',1.0,1.0,1.0,1.0,'KRW_STRONG'),
 ('MID_KRWSTRONG',.50,.50,.50,.50,'KRW_STRONG'),
 ('LOW_KRWSTRONG',.25,0.0,0.0,0.0,'KRW_STRONG'),
]
CURRENT=np.array([.15,.15,.35,0.,.20,0.,.15])

def candidate_weights():
    arr=[]
    for spy in np.arange(.05,.251,.05):
      for qqq in np.arange(.05,.251,.05):
       for btc in np.arange(.25,.401,.05):
        for eth in [0,.05]:
         for k in np.arange(.05,.301,.05):
          for gold in np.arange(0,.251,.05):
           tb=1-spy-qqq-btc-eth-k-gold
           if .05-1e-9 <= tb <= .30+1e-9:
            arr.append([spy,qqq,btc,eth,k,gold,round(tb,10)])
    return np.unique(np.round(np.asarray(arr,float),10),axis=0)

def years(s,e): return (pd.Timestamp(e)-pd.Timestamp(s)).days/365.2425

def drift_curve(curve,alpha,edge,dates):
    c=np.asarray(curve,float); t=(pd.to_datetime(dates)-pd.Timestamp(dates[0])).days.to_numpy(float)/365.2425
    return c*np.exp(-(1-alpha)*edge*t)

def fx_adjust(fx,dates,hist_edge,regime):
    if regime=='HIST': return np.asarray(fx,float).copy()
    target={'FLAT':0.0,'KRW_STRONG':-0.02}[regime]; f=np.asarray(fx,float); t=(pd.to_datetime(dates)-pd.Timestamp(dates[0])).days.to_numpy(float)/365.2425
    return f*np.exp((target-hist_edge)*t)

def add_gold(d):
    g=m.eqdata('GLD').copy(); g['state']=gold_rules.rule_state(g,GOLD_RULE); g['trade_day']=1
    g=g.rename(columns={'open':'gold_open','close':'gold_close','state':'gold_state','trade_day':'gold_trade_day'})[['date','gold_open','gold_close','gold_state','gold_trade_day']]
    z=d.merge(g,on='date',how='left'); z[['gold_close','gold_state']]=z[['gold_close','gold_state']].ffill(); z['gold_trade_day']=z.gold_trade_day.fillna(0); z['gold_open']=z.gold_open.fillna(z.gold_close)
    return z.dropna(subset=['gold_close']).reset_index(drop=True)

def make_cache7(d):
    starts=d.groupby(d.date.dt.to_period('M')).head(1).index.tolist(); starts=[s for s in starts if d.loc[s,'date']>=pd.Timestamp('2018-09-01') and s>=260]; cache=[]
    for h in [3,5]:
      for s in starts:
        e=m.end_idx(d,s,h)
        if e is None: continue
        tb=m.tbill_sleeve(d,s,e)
        curves=np.vstack([
          m.sleeve_simple(d,s,e,'spy','spy_state','spy_trade_day'),
          m.sleeve_simple(d,s,e,'qqq','qqq_state','qqq_trade_day'),
          m.sleeve_simple(d,s,e,'btc','btc_state',None),
          m.eth_sleeve(d,s,e),
          m.sleeve_simple(d,s,e,'k200','k200_state','k200_trade_day'),
          m.sleeve_simple(d,s,e,'gold','gold_state','gold_trade_day'),
          tb]).T
        sy=int(d.loc[s,'date'].year); cache.append({'horizon':h,'start':d.loc[s,'date'],'end':d.loc[e,'date'],'start_year':sy,'segment':m.segment(sy,h),'curves':curves,'tbill':tb,'dates':d.loc[s:e,'date'].to_numpy(),'fx':d.loc[s:e,'fx_close'].to_numpy(float)})
    return cache

def override_and_cache(d0,rule):
    d=spy_stage.override_spy(d0,rule); return d,make_cache7(d)

def calibrate(cache):
    b=[]; q=[]; e=[]; g=[]; f=[]
    for x in cache:
      if x['segment'] not in ('TRAIN','VALID'): continue
      y=years(x['start'],x['end']); c=x['curves']
      b.append(np.log(c[-1,2]/x['tbill'][-1])/y); q.append(np.log(c[-1,1]/c[-1,0])/y); e.append(np.log(c[-1,3]/x['tbill'][-1])/y); g.append(np.log(c[-1,5]/x['tbill'][-1])/y); f.append(np.log(x['fx'][-1]/x['fx'][0])/y)
    return {'btc':float(np.median(b)),'qqq':float(np.median(q)),'eth':float(np.median(e)),'gold':float(np.median(g)),'fx':float(np.median(f))}

def evaluate_world(cache,W,edges,ba,qa,ea,ga,fxr):
    tvc=[]; tvm=[]; ooc=[]; oom=[]
    for x in cache:
      c=x['curves'].copy(); c[:,2]=drift_curve(c[:,2],ba,edges['btc'],x['dates']); c[:,1]=drift_curve(c[:,1],qa,edges['qqq'],x['dates']); c[:,3]=drift_curve(c[:,3],ea,edges['eth'],x['dates']); c[:,5]=drift_curve(c[:,5],ga,edges['gold'],x['dates'])
      fx=fx_adjust(x['fx'],x['dates'],edges['fx'],fxr); _,_,fk,mk,_=stress.port_matrix_krw_fast(c,W,x['dates'],fx); y=years(x['start'],x['end']); cg=np.power(np.maximum(fk,1e-15),1/y)-1
      if x['segment'] in ('TRAIN','VALID'): tvc.append(cg); tvm.append(mk)
      else: ooc.append(cg); oom.append(mk)
    tvc=np.asarray(tvc); tvm=np.asarray(tvm); ooc=np.asarray(ooc); oom=np.asarray(oom)
    return {'median_cagr':np.median(tvc,0),'p10_cagr':np.quantile(tvc,.1,axis=0),'worst_cagr':np.min(tvc,0),'worst_mdd':np.min(tvm,0)},(ooc,oom)

def robust_stats(worlds,W,subset=None):
    if subset is None: subset=np.arange(len(W))
    feasible=np.ones(len(subset),bool); bench={}
    for name,z in worlds.items():
      feasible &= z['worst_mdd'][subset]>=GOAL
      ok=z['worst_mdd']>=GOAL; bench[name]={c:float(np.max(z[c][ok])) for c in REG_W}
    ids=subset[feasible]
    if not len(ids): return None,None,None
    mx=np.full(len(ids),-np.inf); av=np.zeros(len(ids)); mxmed=np.full(len(ids),-np.inf)
    for name,z in worlds.items():
      comp=np.zeros(len(ids)); med=bench[name]['median_cagr']-z['median_cagr'][ids]
      for c,w in REG_W.items(): comp+=w*(bench[name][c]-z[c][ids])
      mx=np.maximum(mx,comp); av+=comp/len(worlds); mxmed=np.maximum(mxmed,med)
    order=np.lexsort((av,mx)); return ids,mx,av

def top_table(worlds,W):
    ids,mx,av=robust_stats(worlds,W); order=np.lexsort((av,mx)); rows=[]
    for r,pos in enumerate(order[:30]):
      ix=int(ids[pos]); row={'rank':r+1,'candidate_index':ix,'max_composite_regret':float(mx[pos]),'avg_composite_regret':float(av[pos])}; row.update({KEY[i]:float(W[ix,i]) for i in range(7)}); rows.append(row)
    return pd.DataFrame(rows)

def gold_frontier(worlds,W):
    rows=[]
    for gv in sorted(np.unique(W[:,5])):
      subset=np.flatnonzero(np.isclose(W[:,5],gv)); ids,mx,av=robust_stats(worlds,W,subset)
      if ids is None: continue
      p=int(np.lexsort((av,mx))[0]); ix=int(ids[p]); rows.append({'gold':float(gv),'candidate_index':ix,'max_composite_regret':float(mx[p]),'avg_composite_regret':float(av[p]),**{KEY[i]:float(W[ix,i]) for i in range(7)}})
    return pd.DataFrame(rows)

def oos_for(ix,W,store):
    rows=[]
    for name,(c,mx,meta) in store.items():
      cg=c[:,ix]; mm=mx[:,ix]; rows.append({**meta,'world':name,'median_cagr':float(np.median(cg)),'p10_cagr':float(np.quantile(cg,.1)),'worst_cagr':float(np.min(cg)),'median_mdd':float(np.median(mm)),'worst_mdd':float(np.min(mm)),**{KEY[i]:float(W[ix,i]) for i in range(7)}})
    return pd.DataFrame(rows)

def find_index(W,w):
    z=np.where(np.all(np.isclose(W,w[None,:]),axis=1))[0]; return int(z[0]) if len(z) else None

def validate(d,W):
    cache=make_cache7(d); x=cache[0]; w=W[:min(8,len(W))]
    a=base.port_matrix_krw(x['curves'],w,x['dates'],x['fx']); b=stress.port_matrix_krw_fast(x['curves'],w,x['dates'],x['fx']); return max(float(np.max(np.abs(np.asarray(u)-np.asarray(v)))) for u,v in zip(a,b))

def main():
    W=candidate_weights(); d0,sr_auto,qr=kt.prepare(j.K_RULE); d0=add_gold(d0); worlds={}; store={}; edges_by={}; validations={}
    for rule in SPY_RULES:
      d,cache=override_and_cache(d0,rule); validations[rule]=validate(d,W); ed=calibrate(cache); edges_by[rule]=ed
      for econ,ba,qa,ea,ga,fxr in ECON:
        z,o=evaluate_world(cache,W,ed,ba,qa,ea,ga,fxr); name=f'{rule}|{econ}'; worlds[name]=z; store[name]=(o[0],o[1],{'spy_rule':rule,'econ':econ,'btc_alpha':ba,'qqq_alpha':qa,'eth_alpha':ea,'gold_alpha':ga,'fx_regime':fxr})
    TOP=top_table(worlds,W); ix=int(TOP.iloc[0].candidate_index); best=float(TOP.iloc[0].max_composite_regret); TOP['within_50bp_regret_plateau']=TOP.max_composite_regret<=best+.005
    FRONT=gold_frontier(worlds,W); OO=oos_for(ix,W,store); ci=find_index(W,CURRENT); CUR=oos_for(ci,W,store)
    def agg(df,label): return {'profile':label,'min_world_median_cagr':float(df.median_cagr.min()),'min_world_p10_cagr':float(df.p10_cagr.min()),'min_world_worst_cagr':float(df.worst_cagr.min()),'worst_world_mdd':float(df.worst_mdd.min()),'max_world_median_cagr':float(df.median_cagr.max())}
    AGG=pd.DataFrame([agg(OO,'gold_robust'),agg(CUR,'prior_final_no_gold')])
    TOP.to_csv(OUT/'top_robust_candidates.csv',index=False); FRONT.to_csv(OUT/'gold_share_frontier.csv',index=False); OO.to_csv(OUT/'selected_oos_worlds.csv',index=False); CUR.to_csv(OUT/'prior_final_oos_worlds.csv',index=False); AGG.to_csv(OUT/'adversarial_oos_summary.csv',index=False); pd.DataFrame(W,columns=KEY).to_csv(OUT/'candidate_weights.csv',index=False)
    meta={'goal_mdd':GOAL,'candidate_count':len(W),'economic_worlds':len(ECON),'model_worlds':len(worlds),'gold_rule':GOLD_RULE,'gold_rule_selected_pre2018':True,'gold_range':[0,.25],'spy_rules':SPY_RULES,'auto_selected_spy_rule':sr_auto,'qqq_rule':qr,'k_rule':j.K_RULE,'econ_definitions':ECON,'edges_by_spy_rule':edges_by,'engine_validation':validations,'data_end':str(d0.date.max().date()),'regret_weights':REG_W}
    (OUT/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)); (OUT/'REPORT.md').write_text('# Final robust retirement portfolio with gold\n\n```json\n'+json.dumps(meta,ensure_ascii=False,indent=2)+'\n```\n\n## Top\n'+TOP.to_markdown(index=False)+'\n\n## Gold frontier\n'+FRONT.to_markdown(index=False)+'\n\n## OOS aggregate\n'+AGG.to_markdown(index=False)+'\n\n## Selected worlds\n'+OO.to_markdown(index=False))
    print('META',json.dumps(meta,ensure_ascii=False)); print('\nTOP\n',TOP.to_string(index=False)); print('\nGOLD FRONTIER\n',FRONT.to_string(index=False)); print('\nAGG\n',AGG.to_string(index=False)); print('\nSELECTED OOS\n',OO.to_string(index=False))
if __name__=='__main__': main()
