"""Capability-control check (cold-read W1): is the family structure a capability artifact?
Regresses per-judge PW acceptance on correct-answer acceptance (disjoint items); reports r, family
residual means, and between-family share of residual variance. Offline, from raw_capture_full.sqlite."""
import json, sqlite3
from collections import defaultdict
import numpy as np
HERE = __file__.rsplit("/",1)[0]
pool={json.loads(l)["item_id"]:json.loads(l) for l in open(f"{HERE}/items_pool.jsonl")}
from canonical import JUDGES
LIN={j[0]:j[1] for j in JUDGES}; M=list(LIN)
S=defaultdict(lambda:defaultdict(list))
for m,i,rj in sqlite3.connect(f"{HERE}/raw_capture_full.sqlite").execute("SELECT model_id,item_id,response_json FROM calls"):
    if i in pool and m in M:
        sc=json.loads(rj).get("score")
        if sc is not None: S[i][m].append(float(sc))
ids=[i for i in pool if all(S[i].get(m) for m in M)]
COR=[i for i in ids if pool[i]["key"]==1]; PW=[i for i in ids if pool[i]["key"]==0 and pool[i].get("dissociation")]
acc={m:np.mean([np.mean(S[i][m])>=0.5 for i in COR]) for m in M}
pw={m:np.mean([np.mean(S[i][m])>=0.5 for i in PW]) for m in M}
x=np.array([acc[m] for m in M]); y=np.array([pw[m] for m in M])
r=float(np.corrcoef(x,y)[0,1]); b,a=np.polyfit(x,y,1)
res={m:pw[m]-(a+b*acc[m]) for m in M}; fam=defaultdict(list)
for m in M: fam[LIN[m]].append(res[m])
between=np.var([np.mean(v) for v in fam.values()],ddof=1); tot=np.var(list(res.values()),ddof=1)
out={"corr_acc_pw":round(r,3),"family_residual_means":{f:round(float(np.mean(v)),3) for f,v in fam.items()},
     "between_family_share_of_residual_var":round(float(between/tot),3)}
print(json.dumps(out,indent=1)); json.dump(out,open(f"{HERE}/capability_control.json","w"))
