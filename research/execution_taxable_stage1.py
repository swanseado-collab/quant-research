#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from research import multi_asset_5way_allocation as m
from research import multi_asset_6way_kospi_trend_compare as kt
from research import spy_rule_robust_stage as spy_stage

OUT=Path('results/execution_taxable_stage1'); OUT.mkdir(parents=True,exist_ok=True)
W={'spy':.15,'qqq':.15,'btc':.35,'k200':.20,'tbill':.15}
ASSETS=['spy','qqq','btc','k200','tbill']
SIZES=[100_000_000.,300_000_000.,1_000_000_000.]
REBAL_MODES=['MONTHLY','BAND']
CRYPTO_TAX=[True,False]
ROUTES=['DIRECT_US','DOMESTIC_ETF']
# Base executable-cost assumptions, not broker promises. Sensitivity is run separately.
COST_BASE={
 'DIRECT_US': {'us_fee':.0007,'kr_fee':.00015,'btc_fee':.0005,'cash_fee':.0007,'fx_fee':.0010},
 'DOMESTIC_ETF': {'us_fee':0.0,'kr_fee':.00015,'btc_fee':.0005,'cash_fee':.00015,'fx_fee':0.0},
}
COST_STRESS={
 'LOW':  {'us_fee':.0005,'kr_fee':.00010,'btc_fee':.0005,'cash_fee':.0005,'fx_fee':.0005},
 'BASE': {'us_fee':.0007,'kr_fee':.00015,'btc_fee':.0005,'cash_fee':.0007,'fx_fee':.0010},
 'HIGH': {'us_fee':.0025,'kr_fee':.00030,'btc_fee':.0005,'cash_fee':.0025,'fx_fee':.0050},
}
# Proxy -> implementation ongoing annual cost. Raw proxy NAV already contains proxy fee, so we scale by fee difference.
PROXY_ER={'spy':.000945,'qqq':.0018,'k200':.0015}
TARGET_ER={
 'DIRECT_US': {'spy':.0003,'qqq':.0015,'k200':.00052,'cash':.0009}, # VOO, QQQM, RISE200 all-in approx, SGOV
 'DOMESTIC_ETF': {'spy':.001083,'qqq':.001561,'k200':.00052,'cash':.0015}, # KODEX S&P/Nasdaq effective-cost approx, RISE200, SOFR
}
DIV_TAX={'DIRECT_US': {'spy':.15,'qqq':.15,'k200':.154}, 'DOMESTIC_ETF': {'spy':.154,'qqq':.154,'k200':.154}}
CASH_INCOME_TAX=.154
CGT_RATE=.22; CGT_DEDUCTION=2_500_000.
DOMESTIC_OVERSEAS_ETF_TAX=.154
FIN_INCOME_THRESHOLD=20_000_000.


def raw_history(ticker,start='2017-01-01'):
    x=yf.Ticker(ticker).history(start=start,auto_adjust=False,actions=True)
    if x.empty: raise RuntimeError(f'No raw history {ticker}')
    x=x.reset_index(); x.columns=[str(c).lower() for c in x.columns]
    dc='date' if 'date' in x.columns else x.columns[0]
    x=x.rename(columns={dc:'date'}); x['date']=pd.to_datetime(x.date).dt.tz_localize(None).dt.normalize()
    if 'dividends' not in x: x['dividends']=0.0
    return x[['date','open','close','dividends']].dropna(subset=['open','close']).sort_values('date').reset_index(drop=True)


def prepare_data():
    d0,sr_auto,qr=kt.prepare('MA100_C3'); d=spy_stage.override_spy(d0,'MA250_C5').copy()
    if qr!='MA250_C3':
        raise RuntimeError(f'QQQ rule drifted: {qr}')
    # Only completed daily bars.
    utc_today=pd.Timestamp.now('UTC').tz_localize(None).normalize(); d=d[d.date<utc_today].copy().reset_index(drop=True)
    raws={'spy':raw_history('SPY'),'qqq':raw_history('QQQ'),'k200':raw_history('069500.KS')}
    for a,x in raws.items():
        x=x.rename(columns={'open':f'{a}_raw_open','close':f'{a}_raw_close','dividends':f'{a}_div'})
        d=d.merge(x,on='date',how='left')
        d[f'{a}_raw_close']=d[f'{a}_raw_close'].ffill()
        d[f'{a}_raw_open']=d[f'{a}_raw_open'].fillna(d[f'{a}_raw_close'])
        d[f'{a}_div']=d[f'{a}_div'].fillna(0.0)
    # BTC raw USD price from Binance data already in d. K200 raw local price is present from Yahoo merge above.
    d['btc_raw_open']=d.btc_open; d['btc_raw_close']=d.btc_close
    d['fx_open']=d.fx_open.ffill(); d['fx_close']=d.fx_close.ffill()
    return d,sr_auto,qr


def scaled_price(d,a,route):
    if a=='btc':
        return d.btc_raw_open.to_numpy(float)*d.fx_open.to_numpy(float), d.btc_raw_close.to_numpy(float)*d.fx_close.to_numpy(float), np.zeros(len(d))
    base_open=d[f'{a}_raw_open'].to_numpy(float); base_close=d[f'{a}_raw_close'].to_numpy(float); div=d[f'{a}_div'].to_numpy(float)
    t=(d.date-d.date.iloc[0]).dt.days.to_numpy(float)/365.2425
    diff=PROXY_ER[a]-TARGET_ER[route][a]; scale=np.exp(diff*t)
    base_open=base_open*scale; base_close=base_close*scale; div=div*scale
    if a in ('spy','qqq'):
        fxop=d.fx_open.to_numpy(float); fxcl=d.fx_close.to_numpy(float)
        return base_open*fxop,base_close*fxcl,div*fxcl
    return base_open,base_close,div


def cash_factor(d,route,i):
    if i==0: return 1.0,0.0
    days=max(1,(d.date.iloc[i]-d.date.iloc[i-1]).days)
    y=max(float(d.yk.iloc[i]),0.0)/100.0
    gross_interest=(1+y)**(days/365.2425)-1
    net_interest=gross_interest*(1-CASH_INCOME_TAX)
    er=TARGET_ER[route]['cash']; usd=(1+net_interest)*math.exp(-er*days/365.2425)
    fx=float(d.fx_close.iloc[i]/d.fx_close.iloc[i-1])
    return usd*fx,gross_interest


def trade_day(d,a,i):
    if a=='btc': return True
    return bool(d[f'{a}_trade_day'].iloc[i]) if a!='k200' else bool(d.k200_trade_day.iloc[i])


def desired_state(d,a,i):
    if a=='tbill': return 0
    j=max(i-1,0)
    col={'spy':'spy_state','qqq':'qqq_state','btc':'btc_state','k200':'k200_state'}[a]
    return int(d[col].iloc[j])


def asset_fee(route,a,cost):
    if a in ('spy','qqq'): return cost['us_fee'] if route=='DIRECT_US' else cost['kr_fee']
    if a=='btc': return cost['btc_fee']
    if a=='k200': return cost['kr_fee']
    return cost['cash_fee']


def holding_ccy(a,active):
    if not active or a=='tbill': return 'USD'
    return 'USD' if a in ('spy','qqq') else 'KRW'


def is_domestic_overseas_taxable(route,a): return route=='DOMESTIC_ETF' and a in ('spy','qqq')
def is_direct_foreign(route,a): return route=='DIRECT_US' and a in ('spy','qqq')


class Sleeve:
    def __init__(self,name):
        self.name=name; self.active=False; self.qty=0.0; self.cost=0.0; self.cash=0.0
    def value(self,px): return self.qty*px if self.active else self.cash


def sell_qty(sl: Sleeve, qty, px, route, cost, led, year):
    if qty<=0 or sl.qty<=0: return 0.0
    qty=min(qty,sl.qty); frac=qty/sl.qty; basis=sl.cost*frac; gross=qty*px; fee=gross*asset_fee(route,sl.name,cost); net=gross-fee; gain=net-basis
    if is_domestic_overseas_taxable(route,sl.name):
        taxable=max(gain,0.0); tax=taxable*DOMESTIC_OVERSEAS_ETF_TAX; net-=tax; led['tax_paid']+=tax; led['financial_income'][year]=led['financial_income'].get(year,0.0)+taxable
    elif is_direct_foreign(route,sl.name):
        led['foreign_gain'][year]=led['foreign_gain'].get(year,0.0)+gain
    elif sl.name=='btc':
        led['crypto_gain'][year]=led['crypto_gain'].get(year,0.0)+gain
    led['trade_cost']+=fee; sl.qty-=qty; sl.cost-=basis
    if sl.qty<1e-12: sl.qty=0.; sl.cost=0.
    return net


def buy_asset(sl: Sleeve, amount, px, route, cost, led):
    if amount<=0: return
    fee=amount*asset_fee(route,sl.name,cost); invest=max(amount-fee,0.0); q=invest/max(px,1e-15)
    sl.qty+=q; sl.cost+=amount; sl.active=True; led['trade_cost']+=fee


def buy_cash(sl: Sleeve, amount, route, cost, led):
    if amount<=0:return
    fee=amount*cost['cash_fee']; sl.cash+=max(amount-fee,0.0); sl.active=False; led['trade_cost']+=fee


def sell_cash(sl: Sleeve, amount, route, cost, led):
    amount=min(amount,sl.cash); fee=amount*cost['cash_fee']; sl.cash-=amount; led['trade_cost']+=fee; return max(amount-fee,0.0)


def apply_fx_cost(amount,cost,led):
    if amount<=0 or cost['fx_fee']<=0:return amount
    fee=amount*cost['fx_fee']; led['fx_cost']+=fee; return max(amount-fee,0.0)


def deduct_tax_from_portfolio(sleeves,amount,prices):
    if amount<=0:return
    tb=sleeves['tbill']; x=min(tb.cash,amount); tb.cash-=x; amount-=x
    if amount<=1e-9:return
    vals=np.array([sleeves[a].value(prices[a]) for a in ASSETS],float); tot=vals.sum()
    if tot<=0:return
    scale=max((tot-amount)/tot,0.0)
    for a in ASSETS:
        sl=sleeves[a]
        if sl.active: sl.qty*=scale; sl.cost*=scale
        else: sl.cash*=scale


def settle_year(sleeves,route,year,crypto_tax,led,prices):
    tax=0.0
    if route=='DIRECT_US': tax+=max(led['foreign_gain'].get(year,0.0)-CGT_DEDUCTION,0.0)*CGT_RATE
    if crypto_tax: tax+=max(led['crypto_gain'].get(year,0.0)-CGT_DEDUCTION,0.0)*CGT_RATE
    if tax>0: led['tax_paid']+=tax; deduct_tax_from_portfolio(sleeves,tax,prices)


def sim(d,s,e,route,notional,rebal_mode,crypto_tax,cost):
    px_open={}; px_close={}; div={}
    for a in ('spy','qqq','btc','k200'):
        op,cl,dv=scaled_price(d,a,route); px_open[a]=op; px_close[a]=cl; div[a]=dv
    sleeves={a:Sleeve(a) for a in ASSETS}; led={'tax_paid':0.,'trade_cost':0.,'fx_cost':0.,'foreign_gain':{},'crypto_gain':{},'financial_income':{},'turnovers':0.,'rebalances':0}
    # Initial allocation at start open. Direct-US route converts only USD holdings; domestic route has no external FX conversion.
    desired={a:desired_state(d,a,s) for a in ASSETS}
    usd_initial=sum(W[a] for a in ASSETS if holding_ccy(a,bool(desired[a]))=='USD')*notional
    fx_init=usd_initial*cost['fx_fee'] if route=='DIRECT_US' else 0.; led['fx_cost']+=fx_init
    capital=notional-fx_init
    for a in ASSETS:
        amt=capital*W[a]
        if a=='tbill' or not desired[a]: buy_cash(sleeves[a],amt,route,cost,led)
        else: buy_asset(sleeves[a],amt,px_open[a][s],route,cost,led)
    last_year=int(d.date.iloc[s].year); peak=0.; worst=0.; vals=[]; fin_over=False
    prev_period=d.date.iloc[s].to_period('M')
    for i in range(s,e+1):
        yr=int(d.date.iloc[i].year)
        # Settle prior calendar-year tax before new year's first trades.
        if yr!=last_year:
            prevpx={a:(px_close[a][i-1] if a!='tbill' else 1.) for a in ASSETS}; settle_year(sleeves,route,last_year,crypto_tax,led,prevpx)
            if led['financial_income'].get(last_year,0.)>FIN_INCOME_THRESHOLD: fin_over=True
            last_year=yr
        # Accrue USD cash sleeves from prior close to current close; active assets are marked by market price instead.
        if i>s:
            f,gross_interest=cash_factor(d,route,i)
            for a,sl in sleeves.items():
                if not sl.active:
                    before=sl.cash; sl.cash*=f
                    # approximate gross USD interest component as financial income, not FX gain.
                    led['financial_income'][yr]=led['financial_income'].get(yr,0.0)+before*gross_interest
        # Signal transitions at today's open.
        for a in ('spy','qqq','btc','k200'):
            if not trade_day(d,a,i): continue
            want=bool(desired_state(d,a,i)); sl=sleeves[a]
            if want==sl.active: continue
            old_ccy=holding_ccy(a,sl.active); new_ccy=holding_ccy(a,want)
            if sl.active:
                proceeds=sell_qty(sl,sl.qty,px_open[a][i],route,cost,led,yr)
                if route=='DIRECT_US' and old_ccy!=new_ccy: proceeds=apply_fx_cost(proceeds,cost,led)
                buy_cash(sl,proceeds,route,cost,led)
            else:
                proceeds=sell_cash(sl,sl.cash,route,cost,led)
                if route=='DIRECT_US' and old_ccy!=new_ccy: proceeds=apply_fx_cost(proceeds,cost,led)
                sl.cash=0.; buy_asset(sl,proceeds,px_open[a][i],route,cost,led)
        # Monthly/band rebalance at first observation of a new month, after signal switches.
        per=d.date.iloc[i].to_period('M')
        if i>s and per!=prev_period:
            openvals={a:sleeves[a].value(px_open[a][i] if a!='tbill' else 1.) for a in ASSETS}; total=sum(openvals.values())
            cw={a:openvals[a]/max(total,1e-15) for a in ASSETS}; do=True
            if rebal_mode=='BAND':
                do=any(abs(cw[a]-W[a])>max(.025,.25*W[a]) for a in ASSETS)
            if do:
                led['rebalances']+=1
                target={a:total*W[a] for a in ASSETS}; current_usd=sum(openvals[a] for a in ASSETS if holding_ccy(a,sleeves[a].active)=='USD'); target_usd=sum(target[a] for a in ASSETS if holding_ccy(a,sleeves[a].active)=='USD')
                fx_cross=abs(target_usd-current_usd) if route=='DIRECT_US' else 0.0
                pool=0.; deficits={}
                for a in ASSETS:
                    delta=openvals[a]-target[a]
                    if delta>1e-8:
                        sl=sleeves[a]
                        if sl.active:
                            q=delta/max(px_open[a][i],1e-15); pool+=sell_qty(sl,q,px_open[a][i],route,cost,led,yr)
                        else: pool+=sell_cash(sl,delta,route,cost,led)
                        led['turnovers']+=delta
                    elif delta< -1e-8: deficits[a]=-delta
                if fx_cross>0:
                    fxfee=min(pool,fx_cross)*cost['fx_fee']; pool-=fxfee; led['fx_cost']+=fxfee
                denom=sum(deficits.values())
                for a,need in deficits.items():
                    x=pool*need/max(denom,1e-15); sl=sleeves[a]
                    if sl.active: buy_asset(sl,x,px_open[a][i],route,cost,led)
                    else: buy_cash(sl,x,route,cost,led)
                    led['turnovers']+=x
            prev_period=per
        # Dividends/distributions, net of withholding, reinvested without commission at close.
        for a in ('spy','qqq','k200'):
            sl=sleeves[a]
            if sl.active and div[a][i]>0 and sl.qty>0:
                gross=sl.qty*div[a][i]
                tax=gross*DIV_TAX[route][a]; net=gross-tax; led['tax_paid']+=tax; led['financial_income'][yr]=led['financial_income'].get(yr,0.0)+gross
                sl.qty+=net/max(px_close[a][i],1e-15)
                # Reinvested distribution becomes new basis for capital-gains calculations.
                sl.cost+=net
        closevals={a:sleeves[a].value(px_close[a][i] if a!='tbill' else 1.) for a in ASSETS}; total=sum(closevals.values()); vals.append(total)
        peak=max(peak,total); worst=min(worst,total/max(peak,1e-15)-1 if peak>0 else 0.)
    # Mark-to-market before liquidation and current-year tax settlement.
    mtm=sum(sleeves[a].value(px_close[a][e] if a!='tbill' else 1.) for a in ASSETS)
    # Liquidate to KRW at final close for a fully comparable after-tax terminal value.
    yr=int(d.date.iloc[e].year); final_pool_krw=0.; usd_value=0.; krw_value=0.
    for a in ASSETS:
        sl=sleeves[a]
        if a=='tbill' or not sl.active:
            proceeds=sell_cash(sl,sl.cash,route,cost,led); sl.cash=0.
        else:
            proceeds=sell_qty(sl,sl.qty,px_close[a][e],route,cost,led,yr); sl.active=False
        if holding_ccy(a,False)=='USD': usd_value+=proceeds
        else: krw_value+=proceeds
        final_pool_krw+=proceeds
    if route=='DIRECT_US' and usd_value>0:
        fxfee=usd_value*cost['fx_fee']; final_pool_krw-=fxfee; led['fx_cost']+=fxfee
    # Settlement after final liquidation. Tax is subtracted directly from terminal pool.
    final_tax=0.
    if route=='DIRECT_US': final_tax+=max(led['foreign_gain'].get(yr,0.0)-CGT_DEDUCTION,0.0)*CGT_RATE
    if crypto_tax: final_tax+=max(led['crypto_gain'].get(yr,0.0)-CGT_DEDUCTION,0.0)*CGT_RATE
    final_pool_krw-=final_tax; led['tax_paid']+=final_tax
    if led['financial_income'].get(yr,0.)>FIN_INCOME_THRESHOLD: fin_over=True
    years=(d.date.iloc[e]-d.date.iloc[s]).days/365.2425
    return {
      'final_liquidated':max(final_pool_krw,1e-9),'final_mtm':mtm,'cagr_liquidated':(max(final_pool_krw,1e-9)/notional)**(1/years)-1,
      'cagr_mtm':(max(mtm,1e-9)/notional)**(1/years)-1,'mdd':worst,'tax_paid':led['tax_paid'],'trade_cost':led['trade_cost'],'fx_cost':led['fx_cost'],
      'financial_income_over20m':fin_over,'max_annual_financial_income':max(led['financial_income'].values()) if led['financial_income'] else 0.,'rebalances':led['rebalances'],'turnover_krw':led['turnovers']
    }


def segment(sy,h): return m.segment(sy,h)
def summarize(g):
    return pd.Series({'cohorts':len(g),'median_cagr':g.cagr_liquidated.median(),'p10_cagr':g.cagr_liquidated.quantile(.1),'worst_cagr':g.cagr_liquidated.min(),'median_mdd':g.mdd.median(),'worst_mdd':g.mdd.min(),'median_tax_pct_initial':(g.tax_paid/g.notional).median(),'median_trade_cost_pct_initial':(g.trade_cost/g.notional).median(),'median_fx_cost_pct_initial':(g.fx_cost/g.notional).median(),'fin_income_20m_hit_rate':g.financial_income_over20m.mean(),'median_rebalances':g.rebalances.median()})


def run_main(d):
    starts=d.groupby(d.date.dt.to_period('M')).head(1).index.tolist(); starts=[s for s in starts if d.date.iloc[s]>=pd.Timestamp('2018-09-01') and s>=260]
    rows=[]
    for h in [3,5]:
      for s in starts:
        e=m.end_idx(d,s,h)
        if e is None: continue
        sy=int(d.date.iloc[s].year); seg=segment(sy,h)
        for route in ROUTES:
          for notional in SIZES:
            for rb in REBAL_MODES:
              for ct in CRYPTO_TAX:
                z=sim(d,s,e,route,notional,rb,ct,COST_BASE[route]); rows.append({'route':route,'notional':notional,'rebalance':rb,'crypto_tax':ct,'horizon':h,'start':d.date.iloc[s],'end':d.date.iloc[e],'start_year':sy,'segment':seg,**z})
    return pd.DataFrame(rows)


def run_fee_stress(d):
    starts=d.groupby(d.date.dt.to_period('M')).head(1).index.tolist(); starts=[s for s in starts if d.date.iloc[s]>=pd.Timestamp('2018-09-01') and s>=260]
    rows=[]
    for h in [3,5]:
      for s in starts:
        e=m.end_idx(d,s,h)
        if e is None: continue
        sy=int(d.date.iloc[s].year); seg=segment(sy,h)
        for name,cost in COST_STRESS.items():
          z=sim(d,s,e,'DIRECT_US',300_000_000.,'BAND',True,cost); rows.append({'cost_profile':name,'route':'DIRECT_US','notional':300_000_000.,'rebalance':'BAND','crypto_tax':True,'horizon':h,'start':d.date.iloc[s],'end':d.date.iloc[e],'start_year':sy,'segment':seg,**z})
    return pd.DataFrame(rows)


def main():
    d,sr_auto,qr=prepare_data(); R=run_main(d); F=run_fee_stress(d)
    R.to_csv(OUT/'cohorts_taxable_routes.csv',index=False); F.to_csv(OUT/'fee_stress_direct_band_300m.csv',index=False)
    S=R.groupby(['route','notional','rebalance','crypto_tax','segment']).apply(summarize,include_groups=False).reset_index(); S.to_csv(OUT/'summary_by_segment.csv',index=False)
    O=S[S.segment=='OOS'].copy(); O.to_csv(OUT/'oos_summary.csv',index=False)
    FS=F.groupby(['cost_profile','segment']).apply(summarize,include_groups=False).reset_index(); FS.to_csv(OUT/'fee_stress_summary.csv',index=False)
    meta={'allocation':W,'fixed_rules':{'SPY':'MA250_C5','QQQ':'MA250_C3','BTC':'MA150_C3','KOSPI200':'MA100_C3'},'routes':ROUTES,'sizes':SIZES,'rebal_modes':REBAL_MODES,'crypto_tax_scenarios':CRYPTO_TAX,'base_costs':COST_BASE,'cost_stress':COST_STRESS,'target_er':TARGET_ER,'tax':{'foreign_stock_cgt_rate':CGT_RATE,'foreign_stock_annual_deduction':CGT_DEDUCTION,'crypto_rate_when_on':CGT_RATE,'crypto_deduction':CGT_DEDUCTION,'domestic_overseas_etf_positive_realized_gain_withholding':DOMESTIC_OVERSEAS_ETF_TAX,'cash_income_tax':CASH_INCOME_TAX,'financial_income_threshold':FIN_INCOME_THRESHOLD},'tax_model_note':'Domestic overseas ETF sale tax uses positive realized price gain as a conservative proxy for min(actual gain,tax-base-price increase). Direct foreign stock and crypto taxes settle yearly; liquidation value includes final tax. Financial-income comprehensive-rate uplift above KRW20m is flagged but not imposed because user marginal income-tax rate is unknown.','cash_policy':'All risk-off sleeves preserve prior research structure by parking in USD short-rate exposure; DIRECT_US pays FX conversion when BTC/KOSPI switch between KRW risk assets and USD cash.','proxy_note':'Signals use SPY/QQQ/KODEX200 history; execution return paths are raw proxy prices/dividends scaled for target-product expense differences.','auto_selected_spy_rule_ignored':sr_auto,'qqq_rule_asserted':qr,'data_end':str(d.date.max().date())}
    (OUT/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)); (OUT/'REPORT.md').write_text('# Taxable execution stage 1\n\n```json\n'+json.dumps(meta,ensure_ascii=False,indent=2)+'\n```\n\n## OOS summary\n'+O.to_markdown(index=False)+'\n\n## Direct fee stress\n'+FS.to_markdown(index=False))
    print('META',json.dumps(meta,ensure_ascii=False)); print('\nOOS\n',O.to_string(index=False)); print('\nFEE STRESS\n',FS.to_string(index=False))

if __name__=='__main__': main()
