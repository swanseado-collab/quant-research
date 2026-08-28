#!/usr/bin/env python3
from pathlib import Path
import numpy as np, pandas as pd, json
from research.multi_asset_5way_allocation import prepare,weights10,sleeve_simple,eth_sleeve,tbill_sleeve,port_matrix,end_idx,segment,summarize,GOALS
OUT=Path('results/multi_asset_5way_fast'); OUT.mkdir(parents=True,exist_ok=True)

def main():
    d,sr,qr=prepare(); W=weights10(); names=['SPY','QQQ','BTC','ETH','TBILL']
    first=d.groupby(d.date.dt.to_period('Q')).head(1).index.tolist(); starts=[s for s in first if d.loc[s,'date']>=pd.Timestamp('2018-09-01') and s>=260]
    rows=[]
    for h in [3,5]:
      for s in starts:
        e=end_idx(d,s,h)
        if e is None: continue
        curves=np.column_stack([sleeve_simple(d,s,e,'spy','spy_state','spy_trade_day'),sleeve_simple(d,s,e,'qqq','qqq_state','qqq_trade_day'),sleeve_simple(d,s,e,'btc','btc_state'),eth_sleeve(d,s,e),tbill_sleeve(d,s,e)])
        fin,dd,to=port_matrix(curves,W,d.loc[s:e,'date'].to_numpy(),'QUARTERLY',False); years=(d.loc[e,'date']-d.loc[s,'date']).days/365.2425; cg=fin**(1/years)-1; sy=d.loc[s,'date'].year; sg=segment(sy,h)
        for ix,w in enumerate(W): rows.append({'horizon':h,'start':d.loc[s,'date'],'segment':sg,'wid':ix,**{names[j].lower():w[j] for j in range(5)},'cagr':cg[ix],'mdd':dd[ix],'turnover':to[ix]})
    R=pd.DataFrame(rows);R.to_csv(OUT/'cohorts.csv',index=False);tv=R[R.segment.isin(['TRAIN','VALID'])];A=tv.groupby(['wid','spy','qqq','btc','eth','tbill']).apply(summarize,include_groups=False).reset_index();picks=[]
    for goal in GOALS:
        q=A[A.worst_mdd>=goal].copy()
        for c in ['median_cagr','p10_cagr','worst_cagr','median_mdd']:q['r_'+c]=q[c].rank(ascending=False,pct=True,method='average')
        q['score']=q[[c for c in q if c.startswith('r_')]].mean(axis=1);z=q.sort_values(['score','worst_cagr'],ascending=[True,False]).iloc[0].to_dict();z['goal_mdd']=goal;picks.append(z)
    P=pd.DataFrame(picks);P.to_csv(OUT/'selected_trainvalid.csv',index=False);oo=[]
    for _,p in P.iterrows():
        z=R[(R.segment=='OOS')&(R.wid==int(p.wid))];a=summarize(z).to_dict();a.update({'goal_mdd':p.goal_mdd,'spy':p.spy,'qqq':p.qqq,'btc':p.btc,'eth':p.eth,'tbill':p.tbill});oo.append(a)
    O=pd.DataFrame(oo);O.to_csv(OUT/'selected_oos.csv',index=False);state={'spy_rule':sr,'qqq_rule':qr,'rows':len(R)};(OUT/'state.json').write_text(json.dumps(state,indent=2));print('STATE',state);print(P.to_string(index=False));print(O.to_string(index=False))
if __name__=='__main__':main()
