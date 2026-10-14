"""Compare baseline vs STRICT-rubric acceptance of surface-plausible-wrong items, per lineage/domain.
Lower acceptance = better (these are all INCORRECT answers; accepting them is the error).
Offline from the two caches."""
import os, json, sqlite3
import numpy as np
from collections import defaultdict

HERE=os.path.dirname(os.path.abspath(__file__))
POOL=os.path.join(HERE,"items_pool.jsonl")
LIN={"us.anthropic.claude-opus-4-5-20251101-v1:0":"anthropic",
 "us.anthropic.claude-sonnet-4-5-20250929-v1:0":"anthropic",
 "us.anthropic.claude-haiku-4-5-20251001-v1:0":"anthropic",
 "amazon.nova-pro-v1:0":"amazon","amazon.nova-lite-v1:0":"amazon","qwen.qwen3-32b-v1:0":"qwen"}
items={json.loads(l)["item_id"]:json.loads(l) for l in open(POOL)}
pw={i for i,it in items.items() if it["key"]==0 and it["dissociation"]}   # plausible-wrong set

def load(db):
    S=defaultdict(lambda:defaultdict(list))
    for m,i,rep,rj in sqlite3.connect(db).execute("SELECT model_id,item_id,rep,response_json FROM calls"):
        if i not in pw: continue
        try: sc=json.loads(rj).get("score")
        except: sc=None
        if sc is not None: S[i][m].append(sc)
    return S

base=load(os.path.join(HERE,"raw_capture_full.sqlite"))     # baseline rubric
strict=load(os.path.join(HERE,"raw_capture_rubric.sqlite"))  # strict rubric

def accept_by_lineage(S):
    # acceptance = mean score on plausible-wrong items (higher = worse); binarize >=0.5 too
    out={}
    for L in set(LIN.values()):
        ms=[m for m in LIN if LIN[m]==L]
        vals=[np.mean(S[i][m]) for i in pw for m in ms if S[i].get(m)]
        binacc=[1.0 if np.mean(S[i][m])>=0.5 else 0.0 for i in pw for m in ms if S[i].get(m)]
        out[L]={"mean_score":round(float(np.mean(vals)),3),
                "accept_rate>=0.5":round(float(np.mean(binacc)),3),"n":len(vals)}
    return out

b=accept_by_lineage(base); s=accept_by_lineage(strict)
print("SURFACE-PLAUSIBLE-WRONG ACCEPTANCE (lower=better; these are all INCORRECT)\n")
print(f"{'lineage':>10} | {'baseline accept>=0.5':>20} | {'strict accept>=0.5':>18} | {'Δ (reduction)':>13}")
res={}
for L in sorted(b):
    ba=b[L]["accept_rate>=0.5"]; sa=s[L]["accept_rate>=0.5"]
    print(f"{L:>10} | {ba:>20.3f} | {sa:>18.3f} | {ba-sa:>+13.3f}")
    res[L]={"baseline":ba,"strict":sa,"reduction":round(ba-sa,3)}

# per-domain for the outlier (amazon)
def dom_accept(S,L):
    ms=[m for m in LIN if LIN[m]==L]; out={}
    for d in ("code","factual","math"):
        vv=[1.0 if np.mean(S[i][m])>=0.5 else 0.0 for i in pw if items[i]["domain"]==d for m in ms if S[i].get(m)]
        if vv: out[d]=round(float(np.mean(vv)),3)
    return out
print("\nAmazon/Nova per-domain accept>=0.5:  baseline",dom_accept(base,"amazon"),"-> strict",dom_accept(strict,"amazon"))
res["amazon_per_domain"]={"baseline":dom_accept(base,"amazon"),"strict":dom_accept(strict,"amazon")}
json.dump(res,open(os.path.join(HERE,"rubric_lever_results.json"),"w"),indent=2)
print("\nWrote rubric_lever_results.json")
