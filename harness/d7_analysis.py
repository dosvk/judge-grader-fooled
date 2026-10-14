"""D7 analysis (checked in per Reviewer-B re-review): strict-rubric false-rejection cost on correct
answers, WITH the selection-bias diagnostics (survivor vs censored baseline, refusal-as-reject bound).
Regenerates d7_false_rejection.json + the sensitivity table. Offline from caches."""
import os, json, sqlite3
import numpy as np
from collections import defaultdict

HERE=os.path.dirname(os.path.abspath(__file__))
JUDGES=[('us.anthropic.claude-opus-4-5-20251101-v1:0','Opus-4.5'),('us.anthropic.claude-sonnet-4-5-20250929-v1:0','Sonnet-4.5'),('us.anthropic.claude-haiku-4-5-20251001-v1:0','Haiku-4.5'),('amazon.nova-pro-v1:0','Nova-Pro'),('amazon.nova-lite-v1:0','Nova-Lite'),('qwen.qwen3-32b-v1:0','Qwen3-32B'),('us.meta.llama3-3-70b-instruct-v1:0','Llama-3.3-70B'),('us.meta.llama4-maverick-17b-instruct-v1:0','Llama-4-Maverick')]
items={json.loads(l)['item_id']:json.loads(l) for l in open(os.path.join(HERE,'items_pool.jsonl'))}

def load(db):
    S=defaultdict(lambda:defaultdict(list))
    for m,i,rep,rj in sqlite3.connect(os.path.join(HERE,db)).execute('SELECT model_id,item_id,rep,response_json FROM calls'):
        try: sc=json.loads(rj).get('score')
        except Exception: sc=None
        if sc is not None: S[i][m].append(float(sc))
    return S

strict=load('raw_capture_rubric.sqlite'); base=load('raw_capture_full.sqlite')
corr=[i for i in strict if i in items and items[i]['key']==1]
usable=lambda S,i,m: len(S[i].get(m,[]))>=2
accept=lambda S,i,m: 1.0 if np.mean(S[i][m])>=0.5 else 0.0

out={}
print(f"{'judge':18} | {'base(all)':>9} | {'strict(surv)':>12} | {'base(surv)':>10} | {'base(cens)':>10} | {'n_cens':>6} | refusal-as-reject strict")
for mid,short in JUDGES:
    surv=[i for i in corr if usable(strict,i,mid)]
    cens=[i for i in corr if not usable(strict,i,mid)]
    b_all=np.mean([accept(base,i,mid) for i in corr if usable(base,i,mid)])
    s_surv=np.mean([accept(strict,i,mid) for i in surv]) if surv else float('nan')
    b_surv=np.mean([accept(base,i,mid) for i in surv if usable(base,i,mid)]) if surv else float('nan')
    b_cens=np.mean([accept(base,i,mid) for i in cens if usable(base,i,mid)]) if cens else float('nan')
    # refusal-as-reject: censored strict cells counted as reject (accept=0)
    s_rar=np.mean([accept(strict,i,mid) if usable(strict,i,mid) else 0.0 for i in corr])
    out[short]={'base_all':round(float(b_all),3),'strict_survivors':round(float(s_surv),3),
        'base_survivors':round(float(b_surv),3),'base_censored':(round(float(b_cens),3) if cens else None),
        'n_censored':len(cens),'strict_refusal_as_reject':round(float(s_rar),3),
        'cost_dropped':round(float(b_all-s_surv),3),'cost_refusal_as_reject':round(float(b_all-s_rar),3)}
    print(f"{short:18} | {b_all:>9.3f} | {s_surv:>12.3f} | {b_surv:>10.3f} | {b_cens if cens else float('nan'):>10.3f} | {len(cens):>6} | {s_rar:.3f}")
json.dump(out,open(os.path.join(HERE,'d7_false_rejection.json'),'w'),indent=2)
print('\nWrote d7_false_rejection.json (with sensitivity bounds)')
