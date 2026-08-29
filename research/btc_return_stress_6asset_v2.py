#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
import pandas as pd
from research import btc_return_stress_6asset as s


def port_matrix_krw_fast(curves,W,dates,fx_close):
    """Monthly-vectorized implementation numerically equivalent to legacy daily engine."""
    curves=np.asarray(curves,float); W=np.asarray(W,float); dates=pd.to_datetime(np.asarray(dates)); fx=np.asarray(fx_close,float)
    G=np.empty_like(curves); G[0]=curves[0]; G[1:]=curves[1:]/curves[:-1]
    k=len(W); pos=W.copy(); turns=np.zeros(k); peak_usd=np.ones(k); worst_usd=np.zeros(k); peak_krw=np.ones(k); worst_krw=np.zeros(k)
    fx0=float(fx[0]); per=dates.to_period('M'); starts=np.r_[0,np.flatnonzero(per[1:]!=per[:-1])+1]; ends=np.r_[starts[1:]-1,len(dates)-1]
    total=np.ones(k)
    for segi,(a,b) in enumerate(zip(starts,ends)):
        if segi>0:
            tot=pos.sum(1); target=tot[:,None]*W; traded=np.abs(target-pos).sum(1); tot2=tot-traded*s.FEE; pos=tot2[:,None]*W; turns+=traded
        cum=np.cumprod(G[a:b+1],axis=0)
        path=cum @ pos.T
        pk=np.maximum.accumulate(np.vstack([peak_usd[None,:],path]),axis=0)[1:]
        worst_usd=np.minimum(worst_usd,np.min(path/np.maximum(pk,1e-15)-1,axis=0))
        peak_usd=np.maximum(peak_usd,np.max(path,axis=0))
        krw_path=path*(fx[a:b+1,None]/fx0)
        pkk=np.maximum.accumulate(np.vstack([peak_krw[None,:],krw_path]),axis=0)[1:]
        worst_krw=np.minimum(worst_krw,np.min(krw_path/np.maximum(pkk,1e-15)-1,axis=0))
        peak_krw=np.maximum(peak_krw,np.max(krw_path,axis=0))
        pos=pos*cum[-1][None,:]; total=pos.sum(1)
    return total,worst_usd,total*float(fx[-1])/fx0,worst_krw,turns


s.port_matrix_krw_fast=port_matrix_krw_fast

if __name__=='__main__':
    s.main()
