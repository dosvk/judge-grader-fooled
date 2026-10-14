"""D4: null/refusal analysis + explicit handling rule + recomputed rubric lever.
Rule (frozen): a (judge,item) cell is usable only if >=2 of K=3 reps parsed; acceptance computed over
usable cells; per-judge null rates reported for BOTH arms so differential missingness is visible.
Also D7 check: strict-rubric false-rejection cost needs correct-item data (collected separately).
Writes d4_null_analysis.json."""
import os, json, sqlite3
import numpy as np
from collections import defaultdict

HERE=os.path.dirname(os.path.abspath(__file__))
JUDGES=[('us.anthropic.claude-opus-4-5-20251101-v1:0','anthropic','Opus-4.5'),
 ('us.anthropic.claude-sonnet-4-5-20250929-v1:0','anthropic','Sonnet-4.5'),
 ('us.anthropic.claude-haiku-4-5-20251001-v1:0','anthropic','Haiku-4.5'),
 ('amazon.nova-pro-v1:0','amazon','Nova-Pro'),('amazon.nova-lite-v1:0','amazon','Nova-Lite'),
 ('qwen.qwen3-32b-v1:0','qwen','Qwen3-32B'),
 ('us.meta.llama3-3-70b-instruct-v1:0','meta','Llama-3.3-70B'),
 ('us.meta.llama4-maverick-17b-instruct-v1:0','meta','Llama-4-Maverick')]
LIN={j[0]:j[1] for j in JUDGES}; SHORT={j[0]:j[2] for j in JUDGES}
items={json.loads(l)['item_id']:json.loads(l) for l in open(os.path.join(HERE,'items_pool.jsonl'))}
PW=[i for i,it in items.items() if it['key']==0 and it['dissociation']]

def load(db, restrict=None):
    """returns S[item][model]=list of parsed scores, and nulls[model]=count of null reps"""
    S=defaultdict(lambda:defaultdict(list)); nulls=defaultdict(int); tot=defaultdict(int)
    for m,i,rep,rj in sqlite3.connect(db).execute('SELECT model_id,item_id,rep,response_json FROM calls'):
        if restrict and i not in restrict: continue
        if m not in LIN: continue
        tot[m]+=1
        try: sc=json.loads(rj).get('score')
        except Exception: sc=None
        if sc is None: nulls[m]+=1
        else: S[i][m].append(float(sc))
    return S,nulls,tot

base_S,base_nulls,base_tot=load(os.path.join(HERE,'raw_capture_full.sqlite'),restrict=set(PW))
strict_S,strict_nulls,strict_tot=load(os.path.join(HERE,'raw_capture_rubric.sqlite'),restrict=set(PW))

def usable(S,i,m): return len(S[i].get(m,[]))>=2       # frozen rule: >=2 of 3 reps parsed
def accept(S,i,m): return 1.0 if np.mean(S[i][m])>=0.5 else 0.0

res={"handling_rule":"cell usable iff >=2 of K=3 reps parsed; acceptance over usable cells only",
     "per_judge_null_rates":{},"lever_recomputed":{}}
print(f"{'judge':18} | {'base null%':>10} | {'strict null%':>12} | {'base PW-acc':>11} | {'strict PW-acc':>13} | usable(strict)")
for mid,_,short in JUDGES:
    bn=base_nulls[mid]/max(base_tot[mid],1)*100
    sn=strict_nulls[mid]/max(strict_tot[mid],1)*100
    b=[accept(base_S,i,mid) for i in PW if usable(base_S,i,mid)]
    s=[accept(strict_S,i,mid) for i in PW if usable(strict_S,i,mid)]
    res["per_judge_null_rates"][short]={"base_pct":round(bn,1),"strict_pct":round(sn,1),
        "strict_usable_cells":len(s),"strict_total_cells":len(PW)}
    res["lever_recomputed"][short]={"base":round(float(np.mean(b)),3) if b else None,
        "strict":round(float(np.mean(s)),3) if s else None}
    print(f"{short:18} | {bn:>9.1f}% | {sn:>11.1f}% | {np.mean(b) if b else float('nan'):>11.3f} | {np.mean(s) if s else float('nan'):>13.3f} | {len(s)}/{len(PW)}")

# lineage-level recomputed lever
for L in ['anthropic','qwen','amazon','meta']:
    ms=[j[0] for j in JUDGES if j[1]==L]
    b=[accept(base_S,i,m) for i in PW for m in ms if usable(base_S,i,m)]
    s=[accept(strict_S,i,m) for i in PW for m in ms if usable(strict_S,i,m)]
    res["lever_recomputed"][f"lineage_{L}"]={"base":round(float(np.mean(b)),3),"strict":round(float(np.mean(s)),3)}
    print(f"lineage {L:10}: base {np.mean(b):.3f} -> strict {np.mean(s):.3f}")
json.dump(res,open(os.path.join(HERE,'d4_null_analysis.json'),'w'),indent=2)
print("\nWrote d4_null_analysis.json")
