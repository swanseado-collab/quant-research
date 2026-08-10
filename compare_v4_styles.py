from hynix_ci import data,lev2,bh,dca,window
from diagnose_v4 import sim
import pandas as pd
u=data();L=lev2(u)
def roll(y,monthly,style):
    ds=pd.date_range(L.Date.min().to_period('M').start_time,L.Date.max(),freq='MS') if monthly else [pd.Timestamp(f'{a}-01-01') for a in range(L.Date.min().year,L.Date.max().year+1)]
    rows=[];seen=set()
    for s in ds:
      z=L[L.Date>=s]
      if z.empty:continue
      a=z.Date.iloc[0]
      if a in seen:continue
      seen.add(a);w=window(L,a,y)
      if w is None:continue
      B=bh(w)
      if style=='TQQQ':V=sim(w,.15,15,.75)
      else:V=sim(w,.20,20,1.)
      rows.append((a,B[1],V[1],'BH' if B[0]>V[0] else style))
    d=pd.DataFrame(rows,columns=['start','bh','v4','winner']);b=(d.winner=='BH').sum();print(style,('monthly' if monthly else 'annual'),y,'N',len(d),'BH',b,'V4',len(d)-b,'BHrate',b/len(d),'medBH',d.bh.median(),'medV4',d.v4.median())
for st in ['TQQQ','SOXL']:
 for y in [3,5]:
  for m in [False,True]:roll(y,m,st)
