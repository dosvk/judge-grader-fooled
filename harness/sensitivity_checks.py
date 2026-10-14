"""Sensitivity checks added in the JUDGe revision (offline, from the raw caches; no network).
Regenerates: (1) strict-arm worst-case censoring bounds on PW; (2) accept-threshold sweep 0.3/0.5/0.7;
(3) correct-answer acceptance span (ceiling check); (4) capability-control permutation p;
(5) Wilson 95% CI half-width range for per-judge PW at n=244. Matches the paper's Sections 4.1-4.4."""
import json, sqlite3, math, random
from collections import defaultdict
import numpy as np
from canonical import JUDGES
import os
HERE = os.path.dirname(os.path.abspath(__file__))
LIN = {j[0]: j[1] for j in JUDGES}; SHORT = {j[0]: j[2] for j in JUDGES}; M = list(LIN)
pool = {json.loads(l)["item_id"]: json.loads(l) for l in open(os.path.join(HERE, "items_pool.jsonl"))}

def load(db, strict=False):
    S = defaultdict(lambda: defaultdict(list))
    for m, i, rq, rj in sqlite3.connect(db).execute(
            "SELECT model_id,item_id,request_json,response_json FROM calls"):
        if i in pool and m in M:
            if strict and '"mode": "strict"' not in rq and '"mode":"strict"' not in rq: continue
            sc = json.loads(rj).get("score")
            if sc is not None: S[m][i].append(float(sc))
    return S

B = load(os.path.join(HERE, "raw_capture_full.sqlite"))
ST = load(os.path.join(HERE, "raw_capture_rubric.sqlite"), strict=True)
ids = [i for i in pool if all(B[m].get(i) for m in M)]
PW = [i for i in ids if pool[i]["key"] == 0 and pool[i].get("dissociation")]
COR = [i for i in ids if pool[i]["key"] == 1]
out = {}

# (1) strict-arm worst-case censoring bounds
out["strict_censoring_bounds"] = {}
for m in M:
    parsed = [i for i in PW if ST[m].get(i)]
    acc = sum(np.mean(ST[m][i]) >= 0.5 for i in parsed)
    miss = len(PW) - len(parsed)
    p23 = [i for i in PW if len(ST[m].get(i, [])) >= 2]
    out["strict_censoring_bounds"][SHORT[m]] = {
        "parsed": len(parsed), "bound_lo": round(acc / len(PW), 4),
        "bound_hi": round((acc + miss) / len(PW), 4),
        "min2of3": round(float(np.mean([np.mean(ST[m][i]) >= 0.5 for i in p23])), 4) if p23 else None}

# (2) threshold sweep
out["threshold_sweep"] = {}
for th in (0.3, 0.5, 0.7):
    fam = defaultdict(list)
    for m in M: fam[LIN[m]].append(np.mean([np.mean(B[m][i]) >= th for i in PW]))
    out["threshold_sweep"][str(th)] = {f: round(float(np.mean(v)), 4) for f, v in fam.items()}

# (3) ceiling check
ca = [float(np.mean([np.mean(B[m][i]) >= 0.5 for i in COR])) for m in M]
out["correct_acceptance_span"] = [round(min(ca), 3), round(max(ca), 3)]

# (4) capability-control permutation p
acc_c = np.array(ca); pw = np.array([np.mean([np.mean(B[m][i]) >= 0.5 for i in PW]) for m in M])
b, a = np.polyfit(acc_c, pw, 1); res = pw - (a + b * acc_c)
fams = [LIN[m] for m in M]
def share(labels):
    g = defaultdict(list)
    for l, r in zip(labels, res): g[l].append(r)
    return np.var([np.mean(v) for v in g.values()], ddof=1) / np.var(res, ddof=1)
obs = share(fams); rng = random.Random(0); N = 20000
cnt = sum(share(rng.sample(fams, len(fams))) >= obs for _ in range(N))
out["capability_control"] = {"family_share": round(float(obs), 4), "permutation_p": round(cnt / N, 4), "N": N}

# (5) Wilson CI half-width range at n=244
def wilson_hw(p, n=len(PW), z=1.96):
    d = 1 + z * z / n
    return z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
hw = [wilson_hw(float(np.mean([np.mean(B[m][i]) >= 0.5 for i in PW]))) for m in M]
out["wilson_ci_halfwidth_pp"] = [round(min(hw) * 100, 2), round(max(hw) * 100, 2)]

print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "sensitivity_checks.json"), "w"), indent=1)
