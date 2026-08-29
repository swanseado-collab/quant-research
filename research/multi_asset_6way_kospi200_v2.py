#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from research import multi_asset_5way_allocation as m

OUT=Path('results/multi_asset_6way_kospi200'); OUT.mkdir(parents=True,exist_ok=True)
FEE=m.FEE
GOALS=[-.30,-.35,-.40,-.45]


def yfdata(t,start='2002-01-01'):
    d=yf.download(t,start=start,auto_adjust=True,progress=False,threads=False)
    if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
    d=d.reset_index().rename(columns={'Date':'date','Open':'open','High':'high','Low':'low','Close':'close'})
    d['date']=pd.to_datetime(d.date).dt.tz_localize(None)
    return d[['date','open','high','low','close']].dropna().drop_duplicates('date').sort_values('date').reset_index(drop=True)


def choose_kospi_rule(k,y):
    z=m.add_rf(k.copy(),y)
    for w in m.MA_WINDOWS: z[f'ma{w}']=z.close.rolling(w,min_periods=w).mean()
    rules=[('BH',None,None)]+[(f'MA{w}_C{c}',w,c) for w in m.MA_WINDOWS for c in m.CONFIRMS]
    starts=z.groupby(z.date.dt.year).head(1).index.tolist(); raw=[]
    for name,w,c in rules:
        st=np.ones(len(z),int) if name=='BH' else m.state_from(z.close,z[f'ma{w}'],c)
        for h in [3,5]:
            for s in starts:
                if s<260: continue
                target=z.loc[s,'date']+pd.DateOffset(years=h)
                if target>pd.Timestamp('2017-12-31'): continue
                e=int(np.searchsorted(z.date.values,target.to_datetime64(),'right')-1)
                if e<=s or (target-z.loc[e,'date']).days>5: continue
                eq,tr=m.sim_trend(z,st,s,e)
                raw.append({'rule':name,'h':h,'start':z.loc[s,'date'],'start_year':int(z.loc[s,'date'].year),'cagr':m.cagr(eq[-1],z.loc[s,'date'],z.loc[e,'date']),'mdd':m.mdd(eq),'trades':tr})
    R=pd.DataFrame(raw)
    if R.empty: raise RuntimeError(f'No KOSPI pre-2018 cohorts. data={z.date.min()}..{z.date.max()} rows={len(z)}')
    yrs=sorted(R.start_year.unique()); cut=yrs[max(0,int(len(yrs)*0.55)-1)]
    R['seg']=np.where(R.start_year<=cut,'TRAIN','VALID')
    S=R.groupby(['rule','seg']).agg(cohorts=('cagr','size'),median_cagr=('cagr','median'),p10_cagr=('cagr',lambda x:x.quantile(.1)),worst_cagr=('cagr','min'),median_mdd=('mdd','median'),worst_mdd=('mdd','min'),median_trades=('trades','median')).reset_index()
    tr=S[S.seg=='TRAIN'].drop(columns='seg'); va=S[S.seg=='VALID'].drop(columns='seg'); q=tr.merge(va,on='rule',suffixes=('_tr','_va'))
    if q.empty: raise RuntimeError(f'KOSPI split empty: years={yrs}, cut={cut}')
    for col in ['median_cagr_tr','p10_cagr_tr','worst_cagr_tr','median_mdd_tr','worst_mdd_tr','median_cagr_va','p10_cagr_va','worst_cagr_va','median_mdd_va','worst_mdd_va']:
        q['r_'+col]=q[col].rank(ascending=False,pct=True,method='average')
    q['score']=q[[c for c in q if c.startswith('r_')]].mean(axis=1)
    q=q.sort_values(['score','median_trades_va'])
    R.to_csv(OUT/'kospi200_rule_cohorts.csv',index=False); q.to_csv(OUT/'kospi200_rule_rank.csv',index=False)
    return str(q.iloc[0].rule),int(cut),str(z.date.min().date())


def add_native_state(d,rule):
    d=d.copy()
    if rule=='BH': d['state']=1
    else:
        a,b=rule.replace('MA','').split('_C'); w=int(a); c=int(b)
        d['ma']=d.close.rolling(w,min_periods=w).mean(); d['state']=m.state_from(d.close,d.ma,c)
    d['trade_day']=1; return d


def prepare6():
    d,sr,qr=m.prepare(); utc_today=pd.Timestamp.now('UTC').tz_localize(None).normalize(); d=d[d.date<utc_today].copy().reset_index(drop=True)
    y=m.fred(); k=yfdata('069500.KS','2002-01-01'); kr,kcut,kstart=choose_kospi_rule(k,y); k=add_native_state(k,kr)
    fx=yfdata('KRW=X','2002-01-01').rename(columns={'open':'fx_open','close':'fx_close'})[['date','fx_open','fx_close']]
    k=k.merge(fx,on='date',how='left').dropna(subset=['fx_open','fx_close']); k['k200_open']=k.open/k.fx_open; k['k200_close']=k.close/k.fx_close
    k=k.rename(columns={'state':'k200_state','trade_day':'k200_trade_day'})[['date','k200_open','k200_close','k200_state','k200_trade_day']]
    d=d.merge(k,on='date',how='left').merge(fx,on='date',how='left'); d[['fx_open','fx_close']]=d[['fx_open','fx_close']].ffill(); d[['k200_close','k200_state']]=d[['k200_close','k200_state']].ffill(); d['k200_trade_day']=d.k200_trade_day.fillna(0); d['k200_open']=d.k200_open.fillna(d.k200_close)
    return d.dropna(subset=['fx_close','k200_close']).reset_index(drop=True),sr,qr,kr,kcut,kstart


def candidate_weights():
    arr=[]
    for spy in [5,10,15]:
      for qqq in [5,10,15,20,25]:
       for btc in [30,35,40,45,50,55,60]:
        for eth in [5,10]:
         for k in [0,5,10,15,20]:
          tb=100-spy-qqq-btc-eth-k
          if 5<=tb<=40: arr.append([spy,qqq,btc,eth,k,tb])
    for k in [0,5,10,15,20]: arr.append([10,10,60-k,10,k,10])
    return np.unique(np.asarray(arr,float),axis=0)/100.


def port_matrix_krw(curves,W,dates,fx_close):
    n=len(dates); k=len(W); pos=W.copy(); peak_usd=np.ones(k); worst_usd=np.zeros(k); peak_krw=np.ones(k); worst_krw=np.zeros(k); turns=np.zeros(k)
    G=np.empty_like(curves); G[0]=curves[0]; G[1:]=curves[1:]/curves[:-1]; fx0=float(fx_close[0]); prev_m=pd.Timestamp(dates[0]).month; total=np.ones(k)
    for j in range(n):
        if j>0:
            dt=pd.Timestamp(dates[j])
            if dt.month!=prev_m:
                tot=pos.sum(1); target=tot[:,None]*W; traded=np.abs(target-pos).sum(1); tot2=tot-traded*FEE; pos=tot2[:,None]*W; turns+=traded; prev_m=dt.month
        pos*=G[j][None,:]; total=pos.sum(1); peak_usd=np.maximum(peak_usd,total); worst_usd=np.minimum(worst_usd,total/peak_usd-1)
        krw=total*float(fx_close[j])/fx0; peak_krw=np.maximum(peak_krw,krw); worst_krw=np.minimum(worst_krw,krw/peak_krw-1)
    return total,worst_usd,total*float(fx_close[-1])/fx0,worst_krw,turns


def summarize(g):
    return pd.Series({'cohorts':len(g),'median_cagr':g.cagr_krw.median(),'p10_cagr':g.cagr_krw.quantile(.1),'worst_cagr':g.cagr_krw.min(),'median_mdd':g.mdd_krw.median(),'worst_mdd':g.mdd_krw.min(),'median_cagr_usd':g.cagr_usd.median(),'median_mdd_usd':g.mdd_usd.median(),'median_turnover':g.turnover.median()})


def main():
    d,sr,qr,kr,kcut,kstart=prepare6(); d.to_csv(OUT/'daily_inputs.csv',index=False); W=candidate_weights(); names=['SPY','QQQ','BTC','ETH','KOSPI200','TBILL']; key=[x.lower() for x in names]; pd.DataFrame(W,columns=names).to_csv(OUT/'candidate_weights.csv',index=False)
    starts=d.groupby(d.date.dt.to_period('M')).head(1).index.tolist(); starts=[s for s in starts if d.loc[s,'date']>=pd.Timestamp('2018-09-01') and s>=260]; rows=[]
    for h in [3,5]:
      for s in starts:
        e=m.end_idx(d,s,h)
        if e is None: continue
        curves=np.vstack([m.sleeve_simple(d,s,e,'spy','spy_state','spy_trade_day'),m.sleeve_simple(d,s,e,'qqq','qqq_state','qqq_trade_day'),m.sleeve_simple(d,s,e,'btc','btc_state',None),m.eth_sleeve(d,s,e),m.sleeve_simple(d,s,e,'k200','k200_state','k200_trade_day'),m.tbill_sleeve(d,s,e)])
        dates=d.loc[s:e,'date'].to_numpy(); fx=d.loc[s:e,'fx_close'].to_numpy(float); fu,mu,fk,mk,to=port_matrix_krw(curves,W,dates,fx); sy=int(d.loc[s,'date'].year); seg=m.segment(sy,h)
        for j,w in enumerate(W):
            rec={'horizon':h,'start':d.loc[s,'date'],'end':d.loc[e,'date'],'start_year':sy,'segment':seg,'cagr_usd':m.cagr(fu[j],d.loc[s,'date'],d.loc[e,'date']),'mdd_usd':mu[j],'cagr_krw':m.cagr(fk[j],d.loc[s,'date'],d.loc[e,'date']),'mdd_krw':mk[j],'turnover':to[j]}; rec.update({key[x]:w[x] for x in range(6)}); rows.append(rec)
    R=pd.DataFrame(rows); R.to_csv(OUT/'cohorts.csv',index=False); S=R.groupby(key+['horizon','segment']).apply(summarize,include_groups=False).reset_index(); S.to_csv(OUT/'segment_summary.csv',index=False)
    tv=R[R.segment.isin(['TRAIN','VALID'])]; A=tv.groupby(key).apply(summarize,include_groups=False).reset_index(); selected=[]
    for goal in GOALS:
        z=A[A.worst_mdd>=goal].copy()
        if z.empty: continue
        for c in ['median_cagr','p10_cagr','worst_cagr']: z['r_'+c]=z[c].rank(ascending=False,pct=True,method='average')
        z['score']=z[['r_median_cagr','r_p10_cagr','r_worst_cagr']].mean(1); q=z.sort_values(['score','worst_mdd'],ascending=[True,False]).iloc[0].to_dict(); q['goal_mdd']=goal; selected.append(q)
    SEL=pd.DataFrame(selected); SEL.to_csv(OUT/'selected_trainvalid.csv',index=False); oo=[]
    for _,q in SEL.iterrows():
        mask=(R.segment=='OOS')
        for c in key: mask &= np.isclose(R[c],q[c])
        a=summarize(R[mask]).to_dict(); a.update({'goal_mdd':q.goal_mdd,**{c:q[c] for c in key}}); oo.append(a)
    O=pd.DataFrame(oo); O.to_csv(OUT/'selected_oos.csv',index=False)
    by=[]
    for ks in [0,.05,.10,.15,.20]:
        z=A[np.isclose(A.kospi200,ks)&(A.worst_mdd>=-.40)].copy()
        if z.empty: continue
        for c in ['median_cagr','p10_cagr','worst_cagr']: z['r_'+c]=z[c].rank(ascending=False,pct=True,method='average')
        z['score']=z[['r_median_cagr','r_p10_cagr','r_worst_cagr']].mean(1); q=z.sort_values('score').iloc[0]; mask=(R.segment=='OOS')
        for c in key: mask &= np.isclose(R[c],q[c])
        os=summarize(R[mask]); by.append({'kospi200':ks,**{c:float(q[c]) for c in key if c!='kospi200'},'tv_median_cagr':q.median_cagr,'tv_p10_cagr':q.p10_cagr,'tv_worst_cagr':q.worst_cagr,'tv_worst_mdd':q.worst_mdd,'oos_median_cagr':os.median_cagr,'oos_p10_cagr':os.p10_cagr,'oos_worst_cagr':os.worst_cagr,'oos_worst_mdd':os.worst_mdd})
    pd.DataFrame(by).to_csv(OUT/'kospi_share_frontier_mdd40.csv',index=False)
    repl=[]
    for ks in [0,.05,.10,.15,.20]:
        ww={'spy':.10,'qqq':.10,'btc':.60-ks,'eth':.10,'kospi200':ks,'tbill':.10}
        for seg in ['TRAIN','VALID','OOS']:
            mask=(R.segment==seg)
            for c,v in ww.items(): mask &= np.isclose(R[c],v)
            if not R[mask].empty:
                a=summarize(R[mask]).to_dict(); a.update({'segment':seg,**ww}); repl.append(a)
    pd.DataFrame(repl).to_csv(OUT/'controlled_kospi_from_btc.csv',index=False)
    ann=[]
    for _,q in SEL.iterrows():
        mask=R.start.dt.month.eq(1)
        for c in key: mask &= np.isclose(R[c],q[c])
        z=R[mask].copy(); z['goal_mdd']=q.goal_mdd; ann.append(z)
    if ann: pd.concat(ann,ignore_index=True).to_csv(OUT/'annual_start_selected.csv',index=False)
    last=d.iloc[-1]; state={'data_end':str(last.date.date()),'spy_rule':sr,'qqq_rule':qr,'kospi200_rule':kr,'kospi_rule_train_cut':kcut,'kospi_data_start':kstart,'btc_rule':'MA150_C3','eth_rule':'MA200_C1 + 40/5x12','fee':FEE,'candidates':len(W),'signals':{'SPY':int(last.spy_state),'QQQ':int(last.qqq_state),'BTC':int(last.btc_state),'ETH':int(last.eth_state),'KOSPI200':int(last.k200_state)}}; (OUT/'state.json').write_text(json.dumps(state,indent=2,ensure_ascii=False)); (OUT/'README.md').write_text('# Six-asset KOSPI200 backtest\n\n'+json.dumps(state,ensure_ascii=False)+'\n\n## Train+Validation\n'+SEL.to_markdown(index=False)+'\n\n## OOS\n'+O.to_markdown(index=False)); print('STATE',json.dumps(state,ensure_ascii=False)); print('\nSELECTED\n',SEL.to_string(index=False)); print('\nOOS\n',O.to_string(index=False)); print('\nKOSPI FRONTIER\n',pd.DataFrame(by).to_string(index=False)); print('\nCONTROLLED\n',pd.DataFrame(repl).to_string(index=False))

if __name__=='__main__': main()
