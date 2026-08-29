#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research import multi_asset_5way_allocation as m
from research import multi_asset_6way_kospi200_v2 as base
from research import multi_asset_6way_kospi_trend_compare as kt

OUT=Path('results/btc_return_stress_6asset'); OUT.mkdir(parents=True,exist_ok=True)
FEE=m.FEE
ALPHAS=[1.00,0.75,0.50,0.25,0.00]
GOALS=[-.30,-.35,-.40,-.45]
KEY=['spy','qqq','btc','eth','kospi200','tbill']
K_RULE='MA100_C3'


def candidate_weights():
    arr=[]
    for spy in np.arange(.05,.251,.05):
      for qqq in np.arange(.10,.351,.05):
       for btc in np.arange(0,.601,.05):
        for eth in [0,.05,.10]:
         for k in np.arange(0,.201,.05):
          tb=1-spy-qqq-btc-eth-k
          if .05-1e-9 <= tb <= .60+1e-9:
            arr.append([spy,qqq,btc,eth,k,round(tb,10)])
    # Add equity-heavy stress candidates so BTC can lose its role without an artificial equity ceiling.
    for spy in np.arange(.10,.501,.10):
      for qqq in np.arange(.10,.501,.10):
       for btc in np.arange(0,.401,.10):
        for eth in [0,.05,.10]:
         for k in [0,.05,.10,.15,.20]:
          tb=1-spy-qqq-btc-eth-k
          if .05-1e-9 <= tb <= .70+1e-9:
            arr.append([spy,qqq,btc,eth,k,round(tb,10)])
    return np.unique(np.round(np.asarray(arr,float),10),axis=0)


def summarize(g):
    if len(g)==0:
        return pd.Series({'cohorts':0,'median_cagr':np.nan,'p10_cagr':np.nan,'worst_cagr':np.nan,'median_mdd':np.nan,'worst_mdd':np.nan,'median_turnover':np.nan})
    return pd.Series({'cohorts':len(g),'median_cagr':g.cagr_krw.median(),'p10_cagr':g.cagr_krw.quantile(.1),'worst_cagr':g.cagr_krw.min(),'median_mdd':g.mdd_krw.median(),'worst_mdd':g.mdd_krw.min(),'median_turnover':g.turnover.median()})


def stress_btc_curve(btc_curve,tbill_curve,alpha):
    """Compress only positive BTC excess return vs T-bill; keep negative excess return unchanged."""
    b=np.asarray(btc_curve,float); r=np.asarray(tbill_curve,float)
    gb=np.empty_like(b); gr=np.empty_like(r)
    gb[0]=b[0]; gr[0]=r[0]
    gb[1:]=b[1:]/b[:-1]; gr[1:]=r[1:]/r[:-1]
    excess=gb/np.maximum(gr,1e-15)-1.0
    stressed=np.where(excess>0,alpha*excess,excess)
    gs=gr*(1.0+stressed)
    return np.cumprod(gs)


def port_matrix_krw_fast(curves,W,dates,fx_close):
    """Exact monthly-rebalance equivalent of base.port_matrix_krw, vectorized by month."""
    curves=np.asarray(curves,float); W=np.asarray(W,float); dates=pd.to_datetime(np.asarray(dates)); fx=np.asarray(fx_close,float)
    G=np.empty_like(curves); G[0]=curves[0]; G[1:]=curves[1:]/curves[:-1]
    k=len(W); pos=W.copy(); turns=np.zeros(k); peak_usd=np.ones(k); worst_usd=np.zeros(k); peak_krw=np.ones(k); worst_krw=np.zeros(k)
    fx0=float(fx[0]); per=dates.to_period('M'); starts=np.r_[0,np.flatnonzero(per[1:]!=per[:-1])+1]; ends=np.r_[starts[1:]-1,len(dates)-1]
    total=np.ones(k)
    for segi,(a,b) in enumerate(zip(starts,ends)):
        if segi>0:
            tot=pos.sum(1); target=tot[:,None]*W; traded=np.abs(target-pos).sum(1); tot2=tot-traded*FEE; pos=tot2[:,None]*W; turns+=traded
        cum=np.cumprod(G[a:b+1],axis=0)  # day x asset
        path=cum @ pos.T                 # day x candidate
        pk=np.maximum.accumulate(np.vstack([peak_usd[None,:],path]),axis=0)[1:]
        worst_usd=np.minimum(worst_usd,np.min(path/np.maximum(pk,1e-15)-1,axis=0)); peak_usd=np.maximum(peak_usd,path[-1])
        krw_path=path*(fx[a:b+1,None]/fx0)
        pkk=np.maximum.accumulate(np.vstack([peak_krw[None,:],krw_path]),axis=0)[1:]
        worst_krw=np.minimum(worst_krw,np.min(krw_path/np.maximum(pkk,1e-15)-1,axis=0)); peak_krw=np.maximum(peak_krw,krw_path[-1])
        pos=pos*cum[-1][None,:]; total=pos.sum(1)
    return total,worst_usd,total*float(fx[-1])/fx0,worst_krw,turns


def validate_fast(d,W):
    # One 3Y cohort, a few weights: require numerical equivalence with prior engine.
    starts=d.groupby(d.date.dt.to_period('M')).head(1).index.tolist(); s=next(x for x in starts if d.loc[x,'date']>=pd.Timestamp('2019-01-01') and x>=260); e=m.end_idx(d,s,3)
    curves=np.vstack([m.sleeve_simple(d,s,e,'spy','spy_state','spy_trade_day'),m.sleeve_simple(d,s,e,'qqq','qqq_state','qqq_trade_day'),m.sleeve_simple(d,s,e,'btc','btc_state',None),m.eth_sleeve(d,s,e),m.sleeve_simple(d,s,e,'k200','k200_state','k200_trade_day'),m.tbill_sleeve(d,s,e)]).T
    dates=d.loc[s:e,'date'].to_numpy(); fx=d.loc[s:e,'fx_close'].to_numpy(float); w=W[:min(8,len(W))]
    a=base.port_matrix_krw(curves,w,dates,fx); b=port_matrix_krw_fast(curves,w,dates,fx)
    dif=max(float(np.max(np.abs(np.asarray(x)-np.asarray(y)))) for x,y in zip(a,b))
    if dif>1e-9: raise RuntimeError(f'fast engine mismatch {dif}')
    return dif


def make_cache(d):
    starts=d.groupby(d.date.dt.to_period('M')).head(1).index.tolist(); starts=[s for s in starts if d.loc[s,'date']>=pd.Timestamp('2018-09-01') and s>=260]
    cache=[]
    for h in [3,5]:
      for s in starts:
        e=m.end_idx(d,s,h)
        if e is None: continue
        tb=m.tbill_sleeve(d,s,e)
        curves=np.vstack([m.sleeve_simple(d,s,e,'spy','spy_state','spy_trade_day'),m.sleeve_simple(d,s,e,'qqq','qqq_state','qqq_trade_day'),m.sleeve_simple(d,s,e,'btc','btc_state',None),m.eth_sleeve(d,s,e),m.sleeve_simple(d,s,e,'k200','k200_state','k200_trade_day'),tb]).T
        sy=int(d.loc[s,'date'].year)
        cache.append({'horizon':h,'start':d.loc[s,'date'],'end':d.loc[e,'date'],'start_year':sy,'segment':m.segment(sy,h),'curves':curves,'tbill':tb,'dates':d.loc[s:e,'date'].to_numpy(),'fx':d.loc[s:e,'fx_close'].to_numpy(float)})
    return cache


def eval_all(cache,W):
    rows=[]; btcrows=[]
    for item in cache:
      basecur=item['curves']; tb=item['tbill']; s=item['start']; e=item['end']
      for alpha in ALPHAS:
        curves=basecur.copy(); curves[:,2]=stress_btc_curve(basecur[:,2],tb,alpha)
        fu,mu,fk,mk,to=port_matrix_krw_fast(curves,W,item['dates'],item['fx'])
        # Standalone stressed BTC sleeve metrics in USD; useful for interpreting what "50%" produced.
        b=curves[:,2]
        btcrows.append({'alpha':alpha,'horizon':item['horizon'],'start':s,'end':e,'start_year':item['start_year'],'segment':item['segment'],'cagr':m.cagr(float(b[-1]),s,e),'mdd':m.mdd(b)})
        years=(e-s).days/365.2425
        cagr_usd=np.power(np.maximum(fu,1e-15),1/years)-1; cagr_krw=np.power(np.maximum(fk,1e-15),1/years)-1
        for j,w in enumerate(W):
          rec={'alpha':alpha,'horizon':item['horizon'],'start':s,'end':e,'start_year':item['start_year'],'segment':item['segment'],'cagr_usd':cagr_usd[j],'mdd_usd':mu[j],'cagr_krw':cagr_krw[j],'mdd_krw':mk[j],'turnover':to[j]}
          rec.update({KEY[i]:w[i] for i in range(6)}); rows.append(rec)
    return pd.DataFrame(rows),pd.DataFrame(btcrows)


def select_profiles(R):
    tv=R[R.segment.isin(['TRAIN','VALID'])]; summaries=[]; sels=[]
    for alpha in ALPHAS:
      z=tv[np.isclose(tv.alpha,alpha)]
      A=z.groupby(KEY).apply(summarize,include_groups=False).reset_index(); A['alpha']=alpha; summaries.append(A)
      for goal in GOALS:
        q=A[A.worst_mdd>=goal].copy()
        if q.empty: continue
        for c in ['median_cagr','p10_cagr','worst_cagr']: q['r_'+c]=q[c].rank(ascending=False,pct=True,method='average')
        q['score']=q[['r_median_cagr','r_p10_cagr','r_worst_cagr']].mean(1)
        best=q.sort_values(['score','worst_mdd'],ascending=[True,False]).iloc[0].to_dict(); best['alpha']=alpha; best['goal_mdd']=goal; sels.append(best)
    return pd.concat(summaries,ignore_index=True),pd.DataFrame(sels)


def oos_selected(R,SEL):
    out=[]
    for _,q in SEL.iterrows():
      mask=R.segment.eq('OOS') & np.isclose(R.alpha,q.alpha)
      for c in KEY: mask &= np.isclose(R[c],q[c])
      s=summarize(R[mask]).to_dict(); s.update({'alpha':q.alpha,'goal_mdd':q.goal_mdd,**{c:q[c] for c in KEY}}); out.append(s)
    return pd.DataFrame(out)


def fixed_profiles(R):
    profiles={
      'growth_no_kospi':{'spy':.05,'qqq':.20,'btc':.60,'eth':.10,'kospi200':0.,'tbill':.05},
      'growth_kospi5':{'spy':.05,'qqq':.15,'btc':.60,'eth':.10,'kospi200':.05,'tbill':.05},
      'older_template':{'spy':.10,'qqq':.10,'btc':.60,'eth':.10,'kospi200':0.,'tbill':.10},
    }
    rows=[]
    for name,p in profiles.items():
      for alpha in ALPHAS:
       for seg in ['TRAIN','VALID','OOS']:
        mask=R.segment.eq(seg)&np.isclose(R.alpha,alpha)
        for c,v in p.items(): mask &= np.isclose(R[c],v)
        if mask.any(): x=summarize(R[mask]).to_dict(); x.update({'profile':name,'alpha':alpha,'segment':seg,**p}); rows.append(x)
    return pd.DataFrame(rows)


def btc_effect(B):
    return B.groupby(['alpha','segment','horizon']).agg(cohorts=('cagr','size'),median_cagr=('cagr','median'),p10_cagr=('cagr',lambda x:x.quantile(.1)),worst_cagr=('cagr','min'),median_mdd=('mdd','median'),worst_mdd=('mdd','min')).reset_index()


def main():
    W=candidate_weights(); pd.DataFrame(W,columns=['SPY','QQQ','BTC','ETH','KOSPI200','TBILL']).to_csv(OUT/'candidate_weights.csv',index=False)
    d,sr,qr=kt.prepare(K_RULE); err=validate_fast(d,W); cache=make_cache(d)
    R,B=eval_all(cache,W); R.to_csv(OUT/'cohorts.csv',index=False); B.to_csv(OUT/'btc_sleeve_cohorts.csv',index=False)
    A,SEL=select_profiles(R); A.to_csv(OUT/'trainvalid_summary.csv',index=False); SEL.to_csv(OUT/'selected_trainvalid.csv',index=False)
    O=oos_selected(R,SEL); O.to_csv(OUT/'selected_oos.csv',index=False)
    F=fixed_profiles(R); F.to_csv(OUT/'fixed_profile_stress.csv',index=False)
    E=btc_effect(B); E.to_csv(OUT/'btc_stress_effect.csv',index=False)
    meta={'stress':'Positive BTC trend-sleeve excess return over T-bill multiplied by alpha; negative excess returns unchanged.','alphas':ALPHAS,'goals':GOALS,'k_rule':K_RULE,'spy_rule':sr,'qqq_rule':qr,'btc_rule':'MA150_C3','eth_rule':'MA200_C1 + 40/5x12','fee':FEE,'candidate_count':len(W),'fast_engine_validation_max_abs_error':err,'cohorts':len(cache),'data_end':str(d.date.max().date())}
    (OUT/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    report='# BTC return-stress study: six-asset retirement-growth portfolio\n\n```json\n'+json.dumps(meta,ensure_ascii=False,indent=2)+'\n```\n\n## Train+Validation selections\n'+SEL.to_markdown(index=False)+'\n\n## OOS\n'+O.to_markdown(index=False)+'\n\n## Fixed profiles\n'+F.to_markdown(index=False)+'\n\n## BTC sleeve stress effect\n'+E.to_markdown(index=False)
    (OUT/'REPORT.md').write_text(report)
    print('META',json.dumps(meta,ensure_ascii=False)); print('\nSELECTED TV\n',SEL.to_string(index=False)); print('\nSELECTED OOS\n',O.to_string(index=False)); print('\nFIXED OOS\n',F[F.segment=='OOS'].to_string(index=False)); print('\nBTC EFFECT\n',E.to_string(index=False))

if __name__=='__main__': main()
