from hynix_ci import data, lev2, bh, v4, dca
import pandas as pd
u=data()
for end in ['2026-07-01','2026-08-07']:
    x=u[u.Date<=pd.Timestamp(end)].reset_index(drop=True)
    L=lev2(x)
    print('CHECKPOINT',end,'rows',len(x),'under_last',x.Close.iloc[-1])
    for name,fn in [('B&H',bh),('V4',v4),('DCA40',dca)]:
        print('CHECK',end,name,fn(L))
