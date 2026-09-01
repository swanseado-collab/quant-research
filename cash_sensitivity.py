import pandas as pd, numpy as np
import stale_entry_backtest as b

YIELDS=[0.0,0.03,0.05]

def sim(d, obs_i, protocol, horizon, ann_y):
    start_i=obs_i+1; end_i=start_i+horizon-1
    if start_i>=len(d) or end_i>=len(d): return None
    cash,shares=1.0,0.0
    normal=(protocol=='LUMP'); waiting=(protocol=='WAIT'); current_phase=True
    targets={'DCA3':[0,21,42],'DCA6':[0,21,42,63,84,105]}.get(protocol,[])
    done=0; ntr=len(targets); vals=[1.0]
    daily=(1+ann_y)**(1/252)-1
    for j in range(start_i,end_i+1):
        o=float(d.at[j,'o']); c=float(d.at[j,'c'])
        on=bool(d.at[j-1,'on_flip']); off=bool(d.at[j-1,'off_flip']); prior=d.at[j-1,'state']
        if normal:
            if off: cash,shares=b.sell_all(cash,shares,o)
            elif on: cash,shares=b.buy_all(cash,shares,o)
            elif j==start_i and prior==1 and protocol=='LUMP': cash,shares=b.buy_all(cash,shares,o)
        elif waiting:
            if on and not current_phase:
                cash,shares=b.buy_all(cash,shares,o); normal=True; waiting=False
            if off: current_phase=False
        elif protocol=='HALF':
            if j==start_i: cash,shares=b.apply_buy(cash,shares,o,0.5)
            if off:
                cash,shares=b.sell_all(cash,shares,o); current_phase=False; waiting=True
        else:
            rel=j-start_i
            if off:
                cash,shares=b.sell_all(cash,shares,o); current_phase=False; waiting=True
            elif current_phase and done<ntr and rel==targets[done]:
                amt=min(cash,1.0/ntr)
                if done==ntr-1: amt=cash
                shares += amt*(1-b.COST)/o; cash-=amt; done+=1
            if done>=ntr and current_phase: normal=True
        if protocol in ('HALF','DCA3','DCA6') and waiting and not normal and on and not current_phase:
            cash,shares=b.buy_all(cash,shares,o); normal=True; waiting=False
        cash *= (1+daily)
        vals.append(cash+shares*c)
    s=pd.Series(vals); dd=s/s.cummax()-1
    return float(s.iloc[-1]-1),float(dd.min())

def main():
    rows=[]
    for t,conf in b.SPECS.items():
        d=b.build_state(b.load_data(t),conf)
        for eid in sorted(int(x) for x in d.episode.dropna().unique()):
            inds=d.index[d.episode.eq(eid)].tolist()
            if not inds: continue
            oi=inds[0]+325
            if oi>inds[-1] or oi>=len(d)-1: continue
            gap=float(d.at[oi,'gap'])
            for y in YIELDS:
                for hname,h in [('1y',252),('3y',756)]:
                    for p in b.PROTOCOLS:
                        r=sim(d,oi,p,h,y)
                        if r is not None: rows.append([t,eid,d.at[oi,'date'],gap,y,hname,p,*r])
    x=pd.DataFrame(rows,columns=['ticker','episode','date','gap','cash_yield','horizon','protocol','ret','mdd'])
    out=x.groupby(['ticker','cash_yield','horizon','protocol']).agg(n=('ret','size'),median_ret=('ret','median'),p10_ret=('ret',lambda z:z.quantile(.1)),worst_ret=('ret','min'),median_mdd=('mdd','median'),worst_mdd=('mdd','min')).reset_index()
    print('=== AGE 325 CASH-YIELD SENSITIVITY ===')
    print(out.to_csv(index=False))
    z=x[(x.gap>=.05)&(x.gap<=.15)]
    out2=z.groupby(['ticker','cash_yield','horizon','protocol']).agg(n=('ret','size'),median_ret=('ret','median'),p10_ret=('ret',lambda q:q.quantile(.1)),worst_ret=('ret','min'),median_mdd=('mdd','median'),worst_mdd=('mdd','min')).reset_index()
    print('=== AGE 325 + GAP 5-15% (INDEPENDENT EPISODES) ===')
    print(out2.to_csv(index=False))
if __name__=='__main__': main()
