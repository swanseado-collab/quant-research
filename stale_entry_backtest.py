from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path('results/stale_entry_20260901')
OUT.mkdir(parents=True, exist_ok=True)
COST = 0.0005
AGES = [150,200,250,300,325,350,400]
HORIZONS = {'6m':126,'1y':252,'2y':504,'3y':756}
PROTOCOLS = ['LUMP','HALF','WAIT','DCA3','DCA6']
SPECS = {'SPY':5,'QQQ':3}
URLS = {
    'SPY':'https://raw.githubusercontent.com/moolobi/notification/main/data/tickers/SPY.csv',
    'QQQ':'https://raw.githubusercontent.com/moolobi/notification/main/data/tickers/QQQ.csv',
}
# Exact transition dates recovered from the user's prior 2017-2026 research cache.
EXPECTED = {
'SPY': [
('2018-10-30',0),('2018-11-07',1),('2018-11-26',0),('2019-02-14',1),
('2020-03-12',0),('2020-05-29',1),('2022-03-10',0),('2022-03-23',1),
('2022-04-27',0),('2023-02-01',1),('2023-03-02',0),('2023-03-30',1),
('2025-04-09',0),('2025-05-14',1)],
'QQQ': [
('2018-10-30',0),('2018-11-02',1),('2018-11-14',0),('2019-02-06',1),
('2019-02-11',0),('2019-02-14',1),('2020-03-13',0),('2020-04-08',1),
('2022-01-25',0),('2022-02-02',1),('2022-02-15',0),('2022-03-30',1),
('2022-04-07',0),('2023-02-06',1),('2023-02-23',0),('2023-03-16',1),
('2025-03-12',0),('2025-03-26',1),('2025-03-31',0),('2025-05-12',1)]}

def load_data(ticker:str)->pd.DataFrame:
    d = pd.read_csv(URLS[ticker])
    d['date'] = pd.to_datetime(d['date'])
    d = d.sort_values('date').drop_duplicates('date').reset_index(drop=True)
    for c in ['o','c']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.dropna(subset=['o','c']).reset_index(drop=True)
    # Source is adjusted OHLC (cross-validated below against prior transition dates).
    d['ma250'] = d['c'].rolling(250, min_periods=250).mean()
    d['above'] = d['c'] > d['ma250']
    return d

def build_state(d:pd.DataFrame, confirm:int)->pd.DataFrame:
    d=d.copy()
    state=np.full(len(d), np.nan)
    side_streak=0
    last_side=None
    cur=np.nan
    for i,row in d.iterrows():
        if pd.isna(row['ma250']):
            continue
        side=bool(row['above'])
        if last_side is None or side != last_side:
            side_streak=1
            last_side=side
        else:
            side_streak += 1
        if side_streak >= confirm:
            cur = 1.0 if side else 0.0
        state[i]=cur
    d['state']=state
    # fresh ON/OFF confirmation at the completed close of this row.
    prev=d['state'].shift(1)
    d['on_flip']=(d['state'].eq(1) & prev.eq(0))
    d['off_flip']=(d['state'].eq(0) & prev.eq(1))
    # Episode id and age: age=0 on fresh ON confirmation close.
    ep=np.full(len(d), np.nan)
    age=np.full(len(d), np.nan)
    eid=-1; current_on=False; a=0
    for i in range(len(d)):
        st=d.at[i,'state']
        if d.at[i,'on_flip']:
            eid += 1; current_on=True; a=0
        elif d.at[i,'off_flip']:
            current_on=False
        if current_on and st==1 and eid>=0:
            ep[i]=eid; age[i]=a; a+=1
    d['episode']=ep
    d['on_age']=age
    d['gap']=d['c']/d['ma250']-1
    return d

def validate_transitions(ticker:str,d:pd.DataFrame):
    actual=[(x.date.strftime('%Y-%m-%d'),int(x.state)) for x in d.loc[(d.date>=pd.Timestamp('2018-01-01')) & (d.date<=pd.Timestamp('2025-12-31')) & (d.on_flip|d.off_flip), ['date','state']].itertuples(index=False)]
    exp=EXPECTED[ticker]
    return {'ticker':ticker,'pass':actual==exp,'expected':exp,'actual':actual}

def apply_buy(cash:float, shares:float, price:float, frac_of_equity:float):
    eq=cash+shares*price
    gross=max(0.0, min(cash, eq*frac_of_equity))
    # spend gross from cash, cost embedded in acquired shares value
    if gross<=0: return cash,shares
    shares += gross*(1-COST)/price
    cash -= gross
    return cash,shares

def buy_all(cash:float,shares:float,price:float):
    if cash<=0:return cash,shares
    gross=cash
    shares += gross*(1-COST)/price
    cash=0.0
    return cash,shares

def sell_all(cash:float,shares:float,price:float):
    if shares<=0:return cash,shares
    cash += shares*price*(1-COST)
    shares=0.0
    return cash,shares

def simulate_from_observation(d:pd.DataFrame, obs_i:int, protocol:str, horizon:int):
    # Decision is made after obs_i close. First possible trade is next trading-day open.
    start_i=obs_i+1
    end_i=start_i+horizon-1
    if start_i>=len(d) or end_i>=len(d): return None
    current_ep=d.at[obs_i,'episode']
    if pd.isna(current_ep): return None
    cash,shares=1.0,0.0
    normal=False
    waiting_for_fresh=False
    dca_targets=[]
    if protocol=='LUMP':
        normal=True  # enter at first open because prior close state is ON
    elif protocol=='HALF':
        pass
    elif protocol=='WAIT':
        waiting_for_fresh=True
    elif protocol=='DCA3':
        dca_targets=[0,21,42]
    elif protocol=='DCA6':
        dca_targets=[0,21,42,63,84,105]
    else: raise ValueError(protocol)

    vals=[1.0]
    first_entry_idx=None
    fresh_wait_idx=None
    current_phase=True
    ntr=len(dca_targets)
    tranche_done=0

    for j in range(start_i,end_i+1):
        pxo=float(d.at[j,'o']); pxc=float(d.at[j,'c'])
        prior_state=d.at[j-1,'state']
        prior_on_flip=bool(d.at[j-1,'on_flip'])
        prior_off_flip=bool(d.at[j-1,'off_flip'])

        if normal:
            # Follow close-confirmed state at this open.
            if prior_off_flip:
                cash,shares=sell_all(cash,shares,pxo)
            elif prior_on_flip:
                cash,shares=buy_all(cash,shares,pxo)
                if first_entry_idx is None: first_entry_idx=j
            elif j==start_i and prior_state==1 and protocol=='LUMP':
                cash,shares=buy_all(cash,shares,pxo)
                if first_entry_idx is None:first_entry_idx=j
        elif waiting_for_fresh:
            # Must see current stale ON finish, then wait for a later 0->1.
            if prior_on_flip and not current_phase:
                cash,shares=buy_all(cash,shares,pxo)
                if first_entry_idx is None:first_entry_idx=j
                fresh_wait_idx=j
                normal=True; waiting_for_fresh=False
            if prior_off_flip:
                current_phase=False
        elif protocol=='HALF':
            if j==start_i:
                # Allocate 50% of starting equity now.
                cash,shares=apply_buy(cash,shares,pxo,0.5)
                if first_entry_idx is None:first_entry_idx=j
            if prior_off_flip:
                cash,shares=sell_all(cash,shares,pxo)
                current_phase=False
                waiting_for_fresh=True
        else: # DCA current stale episode
            rel=j-start_i
            # Exit all if current ON ended; cancel remaining tranches and wait for fresh ON.
            if prior_off_flip:
                cash,shares=sell_all(cash,shares,pxo)
                current_phase=False
                waiting_for_fresh=True
            elif current_phase and tranche_done<ntr and rel==dca_targets[tranche_done]:
                # Each scheduled tranche spends 1/n of initial capital; last can use residual.
                amount=min(cash,1.0/ntr)
                if tranche_done==ntr-1: amount=cash
                if amount>0:
                    shares += amount*(1-COST)/pxo; cash-=amount
                    if first_entry_idx is None:first_entry_idx=j
                tranche_done+=1
            if tranche_done>=ntr and current_phase:
                # Fully deployed; from here follow current signal normally.
                normal=True

        # Generic waiting logic for HALF/DCA after current episode ends on subsequent days.
        if protocol in ('HALF','DCA3','DCA6') and waiting_for_fresh and not normal:
            # If fresh ON was confirmed yesterday (cannot coincide with same off flip), enter full now.
            if prior_on_flip and not current_phase:
                cash,shares=buy_all(cash,shares,pxo)
                fresh_wait_idx=j
                normal=True; waiting_for_fresh=False

        vals.append(cash+shares*pxc)

    s=pd.Series(vals)
    dd=s/s.cummax()-1
    return {
        'ret':float(s.iloc[-1]-1),
        'mdd':float(dd.min()),
        'first_entry_wait': None if first_entry_idx is None else int(first_entry_idx-start_i),
        'fresh_wait': None if fresh_wait_idx is None else int(fresh_wait_idx-start_i),
    }

def episode_wait_to_next_fresh(d:pd.DataFrame,obs_i:int):
    # Wait from first executable open after observation to first executable open after a future fresh ON.
    start=obs_i+1
    if start>=len(d): return None,True
    off_seen=False
    for j in range(start,len(d)):
        if bool(d.at[j-1,'off_flip']): off_seen=True
        if off_seen and bool(d.at[j-1,'on_flip']):
            return j-start,False
    return len(d)-start,True

def run_ticker(ticker:str,confirm:int):
    d=build_state(load_data(ticker),confirm)
    validation=validate_transitions(ticker,d)
    obs=[]
    perf=[]
    waits=[]
    # Only complete, observed 0->1 episodes are eligible by construction episode>=0.
    eps=sorted(int(x) for x in d['episode'].dropna().unique())
    for eid in eps:
        inds=d.index[d.episode.eq(eid)].tolist()
        if not inds: continue
        on0=inds[0]
        on_end=inds[-1]
        for age in AGES:
            oi=on0+age
            if oi>on_end or oi>=len(d)-1: continue
            row=d.loc[oi]
            obs.append({'ticker':ticker,'episode':eid,'age':age,'obs_date':row.date,'close':row.c,'ma250':row.ma250,'gap':row.gap})
            w,cens=episode_wait_to_next_fresh(d,oi)
            waits.append({'ticker':ticker,'episode':eid,'age':age,'obs_date':row.date,'wait_td':w,'censored':cens})
            for hname,h in HORIZONS.items():
                for p in PROTOCOLS:
                    r=simulate_from_observation(d,oi,p,h)
                    if r is None: continue
                    perf.append({'ticker':ticker,'episode':eid,'age':age,'obs_date':row.date,'gap':row.gap,'horizon':hname,'protocol':p,**r})
    last=d.iloc[-1]
    current_age=None
    if last.state==1 and pd.notna(last.episode): current_age=int(last.on_age)
    cur={'ticker':ticker,'last_date':last.date.strftime('%Y-%m-%d'),'close':float(last.c),'ma250':float(last.ma250),'gap':float(last.gap),'state':None if pd.isna(last.state) else int(last.state),'on_age':current_age}
    trans=d.loc[d.on_flip|d.off_flip,['date','state','c','ma250','gap']].copy(); trans.insert(0,'ticker',ticker)
    return d,pd.DataFrame(obs),pd.DataFrame(perf),pd.DataFrame(waits),validation,cur,trans

def summarize(perf:pd.DataFrame):
    if perf.empty:return pd.DataFrame()
    base=perf[perf.protocol.eq('LUMP')][['ticker','episode','age','horizon','ret']].rename(columns={'ret':'lump_ret'})
    x=perf.merge(base,on=['ticker','episode','age','horizon'],how='left')
    x['regret_vs_lump']=x['lump_ret']-x['ret']  # positive = protocol underperformed lump
    out=[]
    for keys,g in x.groupby(['ticker','age','horizon','protocol'],sort=True):
        out.append({
            'ticker':keys[0],'age':keys[1],'horizon':keys[2],'protocol':keys[3],
            'n':len(g),'median_ret':g.ret.median(),'mean_ret':g.ret.mean(),
            'p10_ret':g.ret.quantile(.1),'worst_ret':g.ret.min(),
            'median_mdd':g.mdd.median(),'worst_mdd':g.mdd.min(),
            'median_regret_vs_lump':g.regret_vs_lump.median(),
            'worst_regret_vs_lump':g.regret_vs_lump.max(),
            'beat_lump_rate':float((g.ret>g.lump_ret+1e-12).mean()) if keys[3]!='LUMP' else 0.0,
        })
    return pd.DataFrame(out)

def summarize_wait(waits):
    out=[]
    for keys,g in waits.groupby(['ticker','age']):
        unc=g[~g.censored]
        out.append({'ticker':keys[0],'age':keys[1],'n':len(g),'uncensored_n':len(unc),'censored_n':int(g.censored.sum()),
                    'median_wait_td':unc.wait_td.median() if len(unc) else np.nan,
                    'p90_wait_td':unc.wait_td.quantile(.9) if len(unc) else np.nan,
                    'max_wait_td':unc.wait_td.max() if len(unc) else np.nan})
    return pd.DataFrame(out)

def current_like(perf,obs):
    # Pre-specified subset: stale age 250-400 and price/MA250 gap +5% to +15%.
    x=perf[(perf.age>=250)&(perf.age<=400)&(perf.gap>=.05)&(perf.gap<=.15)].copy()
    if x.empty:return pd.DataFrame()
    base=x[x.protocol.eq('LUMP')][['ticker','episode','age','horizon','ret']].rename(columns={'ret':'lump_ret'})
    x=x.merge(base,on=['ticker','episode','age','horizon'],how='left')
    x['regret_vs_lump']=x.lump_ret-x.ret
    out=[]
    for keys,g in x.groupby(['ticker','horizon','protocol']):
        out.append({'ticker':keys[0],'horizon':keys[1],'protocol':keys[2],'n':len(g),
                    'median_ret':g.ret.median(),'p10_ret':g.ret.quantile(.1),'worst_ret':g.ret.min(),
                    'median_mdd':g.mdd.median(),'worst_mdd':g.mdd.min(),
                    'median_regret_vs_lump':g.regret_vs_lump.median(),
                    'worst_regret_vs_lump':g.regret_vs_lump.max(),
                    'beat_lump_rate':float((g.ret>g.lump_ret+1e-12).mean()) if keys[2]!='LUMP' else 0.0})
    return pd.DataFrame(out)

def main():
    allobs=[]; allperf=[]; allwait=[]; vals=[]; curs=[]; trans=[]
    data_meta=[]
    for t,c in SPECS.items():
        d,o,p,w,v,cur,tr=run_ticker(t,c)
        allobs.append(o); allperf.append(p); allwait.append(w); vals.append(v); curs.append(cur); trans.append(tr)
        data_meta.append({'ticker':t,'rows':len(d),'start':d.date.min().strftime('%Y-%m-%d'),'end':d.date.max().strftime('%Y-%m-%d')})
        d[['date','o','c','ma250','state','on_flip','off_flip','episode','on_age','gap']].to_csv(OUT/f'{t.lower()}_signal_daily.csv',index=False)
    obs=pd.concat(allobs,ignore_index=True); perf=pd.concat(allperf,ignore_index=True); waits=pd.concat(allwait,ignore_index=True); transitions=pd.concat(trans,ignore_index=True)
    summ=summarize(perf); wsum=summarize_wait(waits); cl=current_like(perf,obs)
    obs.to_csv(OUT/'observations.csv',index=False)
    perf.to_csv(OUT/'episode_protocol_results.csv',index=False)
    summ.to_csv(OUT/'summary_by_age.csv',index=False)
    waits.to_csv(OUT/'waits.csv',index=False)
    wsum.to_csv(OUT/'wait_summary.csv',index=False)
    cl.to_csv(OUT/'current_like_summary.csv',index=False)
    transitions.to_csv(OUT/'transitions.csv',index=False)
    meta={'cost_one_way':COST,'cash_return':'0% (entry-timing isolation)','ages':AGES,'horizons_td':HORIZONS,
          'protocol_definition':{
              'LUMP':'100% at next open, then normal trend engine',
              'HALF':'50% at next open; exit that half when current stale ON ends; 100% at next fresh ON; then normal',
              'WAIT':'cash until current stale ON ends and next fresh ON confirms; then 100%; then normal',
              'DCA3':'1/3 at next open and +21/+42 trading days only while current stale ON survives; if it ends, exit/cancel; 100% next fresh ON',
              'DCA6':'1/6 at next open and +21/+42/+63/+84/+105 trading days only while current stale ON survives; if it ends, exit/cancel; 100% next fresh ON'},
          'data':data_meta,'validation':vals,'current':curs}
    (OUT/'metadata.json').write_text(json.dumps(meta,indent=2,default=str))

    lines=['# Stale-entry backtest','',f'One-way cost: {COST:.2%}; cash yield: 0% (timing isolation).','']
    lines.append('## Data and validation')
    for m,v,cur in zip(data_meta,vals,curs):
        lines.append(f"- {m['ticker']}: {m['start']} to {m['end']}, {m['rows']} rows; transition validation={v['pass']}; latest state={cur['state']}, age={cur['on_age']}, gap={cur['gap']:.2%}")
    lines += ['', '## Episode counts by stale age']
    cnt=obs.groupby(['ticker','age']).size().unstack(0).fillna(0).astype(int)
    lines.append(cnt.to_markdown())
    lines += ['', '## 1-year summary']
    one=summ[summ.horizon.eq('1y')].copy()
    for t in SPECS:
        lines += ['',f'### {t}']
        z=one[one.ticker.eq(t)][['age','protocol','n','median_ret','p10_ret','worst_ret','median_mdd','worst_mdd','median_regret_vs_lump','worst_regret_vs_lump','beat_lump_rate']]
        lines.append(z.to_markdown(index=False,floatfmt='.4f'))
    lines += ['', '## 3-year summary']
    three=summ[summ.horizon.eq('3y')].copy()
    for t in SPECS:
        lines += ['',f'### {t}']
        z=three[three.ticker.eq(t)][['age','protocol','n','median_ret','p10_ret','worst_ret','median_mdd','worst_mdd','median_regret_vs_lump','worst_regret_vs_lump','beat_lump_rate']]
        lines.append(z.to_markdown(index=False,floatfmt='.4f'))
    lines += ['', '## WAIT time']
    lines.append(wsum.to_markdown(index=False,floatfmt='.1f'))
    lines += ['', '## Current-like subset (age 250-400, gap +5% to +15%)']
    lines.append(cl.to_markdown(index=False,floatfmt='.4f') if len(cl) else 'No qualifying observations.')
    (OUT/'SUMMARY.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:80]))
    print('\nRESULT_DIR',OUT)

if __name__=='__main__':main()
