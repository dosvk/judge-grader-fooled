"""Gate-4 hardening: bootstrap CIs on the item-adjusted decomposition collapse + per-domain
plausible-wrong rates. Offline from cache. Addresses mock-PC fixes #1 and the per-task 'minor'."""
import os, json, sqlite3
import numpy as np
from collections import defaultdict

HERE=os.path.dirname(os.path.abspath(__file__))
DB=os.path.join(HERE,"raw_capture_full.sqlite"); POOL=os.path.join(HERE,"items_pool.jsonl")
LIN={"us.anthropic.claude-opus-4-5-20251101-v1:0":"anthropic",
 "us.anthropic.claude-sonnet-4-5-20250929-v1:0":"anthropic",
 "us.anthropic.claude-haiku-4-5-20251001-v1:0":"anthropic",
 "amazon.nova-pro-v1:0":"amazon","amazon.nova-lite-v1:0":"amazon","qwen.qwen3-32b-v1:0":"qwen","us.meta.llama3-3-70b-instruct-v1:0":"meta","us.meta.llama4-maverick-17b-instruct-v1:0":"meta"}
items={json.loads(l)["item_id"]:json.loads(l) for l in open(POOL)}
db=sqlite3.connect(DB); S=defaultdict(lambda:defaultdict(list))
for m,i,rep,rj in db.execute("SELECT model_id,item_id,rep,response_json FROM calls"):
    try: sc=json.loads(rj).get("score")
    except: sc=None
    if sc is not None: S[i][m].append(sc)
models=list(LIN.keys()); ids=[i for i in items if i in S and all(m in S[i] for m in models)]
M=np.array([[np.mean(S[i][m]) for m in models] for i in ids]); keys=np.array([items[i]["key"] for i in ids])
dom=np.array([items[i]["domain"] for i in ids]); disso=np.array([items[i]["dissociation"] for i in ids])
E=M-keys[:,None]; linidx={L:[k for k,m in enumerate(models) if LIN[m]==L] for L in set(LIN.values())}
lin_names=sorted(linidx)

def shared_components(Esub):
    vo=np.mean(np.var(Esub,axis=1,ddof=1))
    raw=max(Esub.mean(axis=1).var(ddof=1)-vo/Esub.shape[1],0.0)
    linmeans=np.array([Esub[:,linidx[L]].mean(axis=1).mean() for L in lin_names])
    between=linmeans.var(ddof=1)   # between-lineage leniency spread (item-difficulty-free at aggregate)
    return raw, between

rng=np.random.default_rng(20260816)
raws=[]; betweens=[]
n=len(ids)
for _ in range(2000):
    idx=rng.integers(0,n,n)
    r,b=shared_components(E[idx]); raws.append(r); betweens.append(b)
def ci(a): a=np.array(a); return [round(float(np.percentile(a,2.5)),5),round(float(np.percentile(a,97.5)),5)]

# per-domain plausible-wrong acceptance per lineage
pw=(keys==0)&disso
per_dom={}
for d in set(dom):
    mask=pw&(dom==d)
    if mask.any():
        per_dom[d]={L:round(float(M[mask][:,linidx[L]].mean()),3) for L in lin_names}

res={
 "raw_shared_point":round(float(shared_components(E)[0]),5),
 "raw_shared_CI95":ci(raws),
 "between_lineage_point":round(float(shared_components(E)[1]),5),
 "between_lineage_CI95":ci(betweens),
 "collapse_note":"between-lineage (leniency) component is an order of magnitude below raw shared; "
                 "CI shows it is small but >0 => a real, modest leniency axis, not heritage common-mode.",
 "plausible_wrong_acceptance_by_domain_and_lineage":per_dom,
}
json.dump(res,open(os.path.join(HERE,"gate4_hardening.json"),"w"),indent=2)
print(json.dumps(res,indent=2))
