#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from research import multi_asset_5way_allocation as m
from research import multi_asset_6way_kospi200_v2 as base

OUT=Path('results/multi_asset_6way_kospi_trend'); OUT.mkdir(parents=True,exist_ok=True)
FEE=m.FEE
K_RULES=['MA100_C1','MA100_C3','M12']
GOALS=[-.30,-.35,-.40,-.45]
KEY=['spy','qqq','btc','eth','kospi200','tbill']


def monthly_state(k,months=12):
    x=k[['date','close']].copy(); x['period']=x.date.dt.to_period('M')
    last=x.groupby('period',sort=True).tail(1).copy()
    last['sma']=last.close.rolling(months,min_periods=months).mean()
    last['sig']=(last.close>last.sma).astype(int)
    mp=dict(zip(last.date,last.sig))
    st=np.zeros(len(k),dtype=int); cur=0
    for i,dt in enumerate(k.date):
        if dt in mp: cur=int(mp[dt])
        st[i]=cur
    return st


def add_state(k,rule):
    k=k.copy()
    if rule.startswith('MA'):
        a,b=rule.replace('MA','').split('_C'); w=int(a); c=int(b)
        ma=k.close.rolling(w,min_periods=w).mean(); k['state']=m.state_from(k.close,ma,c)
    elif rule=='M12':
        k['state']=monthly_state(k,12)
    else: raise ValueError(rule)
    k['trade_day']=1
    return k


def prepare(rule):
    d,sr,qr=m.prepare(); utc_today=pd.Timestamp.now('UTC').tz_localize(None).normalize(); d=d[d.date<utc_today].copy().reset_index(drop=True)
    k=base.yfdata('069500.KS','2002-01-01'); k=add_state(k,rule)
    fx=base.yfdata('KRW=X','2002-01-01').rename(columns={'open':'fx_open','close':'fx_close'})[['date','fx_open','fx_close']]
    k=k.merge(fx,on='date',how='left').dropna(subset=['fx_open','fx_close'])
    k['k200_open']=k.open/k.fx_open; k['k200_close']=k.close/k.fx_close
    k=k.rename(columns={'state':'k200_state','trade_day':'k200_trade_day'})[['date','k200_open','k200_close','k200_state','k200_trade_day']]
    d=d.merge(k,on='date',how='left').merge(fx,on='date',how='left')
    d[['fx_open','fx_close']]=d[['fx_open','fx_close']].ffill()
    d[['k200_close','k200_state']]=d[['k200_close','k200_state']].ffill()
    d['k200_trade_day']=d.k200_trade_day.fillna(0); d['k200_open']=d.k200_open.fillna(d.k200_close)
    return d.dropna(subset=['fx_close','k200_close']).reset_index(drop=True),sr,qr


def summarize(g):
    return pd.Series({'cohorts':len(g),'median_cagr':g.cagr_krw.median(),'p10_cagr':g.cagr_krw.quantile(.1),'worst_cagr':g.cagr_krw.min(),'median_mdd':g.mdd_krw.median(),'worst_mdd':g.mdd_krw.min(),'median_cagr_usd':g.cagr_usd.median(),'median_mdd_usd':g.mdd_usd.median(),'median_turnover':g.turnover.median()})


def run_rule(rule,W):
    d,sr,qr=prepare(rule); starts=d.groupby(d.date.dt.to_period('M')).head(1).index.tolist(); starts=[s for s in starts if d.loc[s,'date']>=pd.Timestamp('2018-09-01') and s>=260]
    rows=[]
    for h in [3,5]:
      for s in starts:
        e=m.end_idx(d,s,h)
        if e is None: continue
        curves=np.vstack([
          m.sleeve_simple(d,s,e,'spy','spy_state','spy_trade_day'),
          m.sleeve_simple(d,s,e,'qqq','qqq_state','qqq_trade_day'),
          m.sleeve_simple(d,s,e,'btc','btc_state',None),
          m.eth_sleeve(d,s,e),
          m.sleeve_simple(d,s,e,'k200','k200_state','k200_trade_day'),
          m.tbill_sleeve(d,s,e)]).T
        dates=d.loc[s:e,'date'].to_numpy(); fx=d.loc[s:e,'fx_close'].to_numpy(float)
        fu,mu,fk,mk,to=base.port_matrix_krw(curves,W,dates,fx); sy=int(d.loc[s,'date'].year); seg=m.segment(sy,h)
        for j,w in enumerate(W):
            rec={'k_rule':rule,'horizon':h,'start':d.loc[s,'date'],'end':d.loc[e,'date'],'start_year':sy,'segment':seg,
                 'cagr_usd':m.cagr(fu[j],d.loc[s,'date'],d.loc[e,'date']),'mdd_usd':mu[j],
                 'cagr_krw':m.cagr(fk[j],d.loc[s,'date'],d.loc[e,'date']),'mdd_krw':mk[j],'turnover':to[j]}
            rec.update({KEY[x]:w[x] for x in range(6)}); rows.append(rec)
    return pd.DataFrame(rows),d,sr,qr


def select_profiles(R):
    tv=R[R.segment.isin(['TRAIN','VALID'])]
    A=tv.groupby(['k_rule']+KEY).apply(summarize,include_groups=False).reset_index()
    out=[]
    for goal in GOALS:
      z=A[A.worst_mdd>=goal].copy()
      if z.empty: continue
      for c in ['median_cagr','p10_cagr','worst_cagr']: z['r_'+c]=z[c].rank(ascending=False,pct=True,method='average')
      z['score']=z[['r_median_cagr','r_p10_cagr','r_worst_cagr']].mean(axis=1)
      q=z.sort_values(['score','worst_mdd'],ascending=[True,False]).iloc[0].to_dict(); q['goal_mdd']=goal; q['constraint']='KOSPI_CAN_BE_ZERO'; out.append(q)
      z2=z[z.kospi200>=.05].copy()
      if not z2.empty:
        q=z2.sort_values(['score','worst_mdd'],ascending=[True,False]).iloc[0].to_dict(); q['goal_mdd']=goal; q['constraint']='KOSPI_MIN_5'; out.append(q)
    return A,pd.DataFrame(out)


def oos_for(R,SEL):
    rows=[]
    for _,q in SEL.iterrows():
      mask=R.segment.eq('OOS') & R.k_rule.eq(q.k_rule)
      for c in KEY: mask &= np.isclose(R[c],q[c])
      s=summarize(R[mask]).to_dict(); s.update({'goal_mdd':q.goal_mdd,'constraint':q.constraint,'k_rule':q.k_rule,**{c:q[c] for c in KEY}}); rows.append(s)
    return pd.DataFrame(rows)


def frontier(R):
    tv=R[R.segment.isin(['TRAIN','VALID'])]; oo=R[R.segment.eq('OOS')]; rows=[]
    for rule in K_RULES:
      for ks in [0,.05,.10,.15,.20]:
        z=tv[(tv.k_rule==rule)&np.isclose(tv.kospi200,ks)]
        if z.empty: continue
        A=z.groupby(KEY).apply(summarize,include_groups=False).reset_index(); A=A[A.worst_mdd>=-.40].copy()
        if A.empty: continue
        for c in ['median_cagr','p10_cagr','worst_cagr']: A['r_'+c]=A[c].rank(ascending=False,pct=True,method='average')
        A['score']=A[['r_median_cagr','r_p10_cagr','r_worst_cagr']].mean(axis=1); q=A.sort_values('score').iloc[0]
        mask=oo.k_rule.eq(rule)
        for c in KEY: mask &= np.isclose(oo[c],q[c])
        os=summarize(oo[mask]); rows.append({'k_rule':rule,'kospi200':ks,**{c:float(q[c]) for c in KEY if c!='kospi200'},
          'tv_median_cagr':q.median_cagr,'tv_p10_cagr':q.p10_cagr,'tv_worst_cagr':q.worst_cagr,'tv_worst_mdd':q.worst_mdd,
          'oos_median_cagr':os.median_cagr,'oos_p10_cagr':os.p10_cagr,'oos_worst_cagr':os.worst_cagr,'oos_worst_mdd':os.worst_mdd})
    return pd.DataFrame(rows)


def main():
    W=base.candidate_weights(); pd.DataFrame(W,columns=['SPY','QQQ','BTC','ETH','KOSPI200','TBILL']).to_csv(OUT/'candidate_weights.csv',index=False)
    allr=[]; states={}
    for rule in K_RULES:
      R,d,sr,qr=run_rule(rule,W); allr.append(R); last=d.iloc[-1]; states[rule]={'data_end':str(last.date.date()),'SPY':int(last.spy_state),'QQQ':int(last.qqq_state),'BTC':int(last.btc_state),'ETH':int(last.eth_state),'KOSPI200':int(last.k200_state),'spy_rule':sr,'qqq_rule':qr}
    R=pd.concat(allr,ignore_index=True); R.to_csv(OUT/'cohorts.csv',index=False)
    A,SEL=select_profiles(R); A.to_csv(OUT/'trainvalid_summary.csv',index=False); SEL.to_csv(OUT/'selected_trainvalid.csv',index=False)
    O=oos_for(R,SEL); O.to_csv(OUT/'selected_oos.csv',index=False)
    F=frontier(R); F.to_csv(OUT/'kospi_rule_share_frontier_mdd40.csv',index=False)
    # Controlled replacement of BTC by KOSPI in user's growth template
    ctrl=[]
    for rule in K_RULES:
      for ks in [0,.05,.10,.15,.20]:
        ww={'spy':.10,'qqq':.10,'btc':.60-ks,'eth':.10,'kospi200':ks,'tbill':.10}
        for seg in ['TRAIN','VALID','OOS']:
          mask=R.segment.eq(seg)&R.k_rule.eq(rule)
          for c,v in ww.items(): mask &= np.isclose(R[c],v)
          if mask.any(): a=summarize(R[mask]).to_dict(); a.update({'k_rule':rule,'segment':seg,**ww}); ctrl.append(a)
    C=pd.DataFrame(ctrl); C.to_csv(OUT/'controlled_btc_to_kospi.csv',index=False)
    meta={'fee':FEE,'rules':K_RULES,'candidate_count':len(W),'states':states,'note':'KOSPI B&H excluded; rules selected from prior long-history study and compared without using OOS for selection.'}; (OUT/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    report='# Six-asset portfolio with non-BH KOSPI200 rules\n\n'+json.dumps(meta,ensure_ascii=False,indent=2)+'\n\n## Selected Train+Validation\n'+SEL.to_markdown(index=False)+'\n\n## OOS\n'+O.to_markdown(index=False)+'\n\n## -40% frontier\n'+F.to_markdown(index=False)
    (OUT/'REPORT.md').write_text(report)
    print('SELECTED\n',SEL.to_string(index=False)); print('\nOOS\n',O.to_string(index=False)); print('\nFRONTIER\n',F.to_string(index=False)); print('\nCONTROLLED OOS\n',C[C.segment=='OOS'].to_string(index=False)); print('\nSTATES',json.dumps(states,ensure_ascii=False))

if __name__=='__main__': main()
