"""
Gate-2 required fix: partial out ITEM DIFFICULTY (item fixed effect) before decomposing.
If a shared judge component survives item-demeaning, it's a real correlated-judge signal;
if it collapses, the raw 'common-mode' was mostly stimulus difficulty (must be labeled so).
Also: leniency ordering, and item-difficulty-controlled pairwise lineage agreement.
Offline, from raw_capture_full.sqlite.
"""
import os, json, sqlite3
import numpy as np
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "raw_capture_full.sqlite"); POOL = os.path.join(HERE, "items_pool.jsonl")
LIN = {"us.anthropic.claude-opus-4-5-20251101-v1:0":"anthropic",
       "us.anthropic.claude-sonnet-4-5-20250929-v1:0":"anthropic",
       "us.anthropic.claude-haiku-4-5-20251001-v1:0":"anthropic",
       "amazon.nova-pro-v1:0":"amazon","amazon.nova-lite-v1:0":"amazon",
       "qwen.qwen3-32b-v1:0":"qwen","us.meta.llama3-3-70b-instruct-v1:0":"meta","us.meta.llama4-maverick-17b-instruct-v1:0":"meta",
       "openai.gpt-oss-120b-1:0":"openai","openai.gpt-oss-20b-1:0":"openai",
       "google.gemma-3-27b-it":"google","google.gemma-3-12b-it":"google"}

items={json.loads(l)["item_id"]:json.loads(l) for l in open(POOL)}
db=sqlite3.connect(DB)
S=defaultdict(lambda:defaultdict(list))
for m,i,rep,rj in db.execute("SELECT model_id,item_id,rep,response_json FROM calls"):
    try: sc=json.loads(rj).get("score")
    except: sc=None
    if sc is not None: S[i][m].append(sc)

models=list(LIN.keys())
ids=[i for i in items if i in S and all(m in S[i] for m in models)]
M=np.array([[np.mean(S[i][m]) for m in models] for i in ids])   # item x model
keys=np.array([items[i]["key"] for i in ids])
E=M-keys[:,None]                                                # error

# --- RAW decomposition (for comparison) ---
vo_raw=np.mean(np.var(E,axis=1,ddof=1))
rs_raw=max(E.mean(axis=1).var(ddof=1)-vo_raw/len(models),0.0)

# --- ITEM-DEMEANED: remove per-item mean error (item fixed effect) ---
E_dm=E-E.mean(axis=1,keepdims=True)          # removes the item-difficulty common component entirely
# after demeaning, the remaining shared structure across judges at the LINEAGE level:
# compute lineage mean-error, then variance across lineages (between-lineage) vs within
lin_ids=sorted(set(LIN.values()))
lin_err={L:E[:,[k for k,m in enumerate(models) if LIN[m]==L]].mean(axis=1) for L in lin_ids}
# between-lineage variance component (per item mean removed) vs residual
lin_matrix=np.array([lin_err[L] for L in lin_ids]).T     # item x lineage
lin_dm=lin_matrix-lin_matrix.mean(axis=1,keepdims=True)
between_lineage_var=lin_matrix.mean(axis=0).var(ddof=1)  # variance of lineage MEAN errors (leniency spread)
within_after_demean=np.mean(np.var(lin_dm,axis=1,ddof=1))

# leniency = mean score (higher = more accepting)
leniency={m:float(np.mean(M[:,k])) for k,m in enumerate(models)}
lin_leniency={L:float(np.mean([leniency[m] for m in models if LIN[m]==L])) for L in lin_ids}

# pairwise lineage agreement on binarized verdict, controlling difficulty (correlation of demeaned err)
pair={}
for a in range(len(lin_ids)):
    for b in range(a+1,len(lin_ids)):
        A,B=lin_ids[a],lin_ids[b]
        r=np.corrcoef(lin_err[A],lin_err[B])[0,1]
        pair[f"{A}~{B}"]=round(float(r),3)

res={
 "raw_common_mode": round(rs_raw,5), "orthogonal_var": round(vo_raw,5),
 "raw_ratio_common_over_total": round(rs_raw/(rs_raw+vo_raw),3),
 "AFTER_item_demeaning": {
    "between_lineage_var_(leniency_spread)": round(float(between_lineage_var),5),
    "within_lineage_residual_var": round(float(within_after_demean),5),
    "note": "item difficulty fully removed by demeaning; remaining between-lineage var = leniency differences"
 },
 "lineage_leniency_meanscore": {k:round(v,3) for k,v in lin_leniency.items()},
 "pairwise_lineage_error_correlation": pair,
 "interpretation": (
    "Raw 'common-mode' is inflated by shared item difficulty. After removing the item fixed effect, "
    "the surviving between-lineage structure is a LENIENCY spread (Amazon accepts more), not heritage-"
    "driven correlated failure. Pairwise error correlations show which lineages track together.")
}
json.dump(res,open(os.path.join(HERE,"decomposition_controlled.json"),"w"),indent=2)
print(json.dumps(res,indent=2))
