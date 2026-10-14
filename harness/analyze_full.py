"""
Analyze the full run from raw_capture_full.sqlite (offline, no network).
Produces: judge accuracy vs keys, per-lineage means, common-mode decomposition + phi interval
(conditional over the 3 sampled lineages; upper bound per Gate-1b), per-domain breakdown,
dissociation delta probe (R1), noise floor. Writes results JSON for the paper's figures.
"""
import os, json, sqlite3
import numpy as np
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "raw_capture_full.sqlite")
POOL = os.path.join(HERE, "items_pool.jsonl")
OUT = os.path.join(HERE, "full_analysis.json")

LINEAGE = {  # model_id -> lineage
    "us.anthropic.claude-opus-4-5-20251101-v1:0": "anthropic",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": "anthropic",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": "anthropic",
    "amazon.nova-pro-v1:0": "amazon", "amazon.nova-lite-v1:0": "amazon",
    "qwen.qwen3-32b-v1:0": "qwen", "us.meta.llama3-3-70b-instruct-v1:0": "meta", "us.meta.llama4-maverick-17b-instruct-v1:0": "meta",
}

def load_items():
    d = {}
    for l in open(POOL):
        it = json.loads(l); d[it["item_id"]] = it
    return d

def load_scores():
    db = sqlite3.connect(DB)
    rows = db.execute("SELECT model_id,item_id,rep,response_json FROM calls").fetchall()
    S = defaultdict(lambda: defaultdict(dict))   # item -> model -> rep -> score
    for model_id, item_id, rep, rj in rows:
        try:
            sc = json.loads(rj).get("score")
        except Exception:
            sc = None
        S[item_id][model_id][rep] = sc
    return S

def main():
    items = load_items()
    S = load_scores()
    models = list(LINEAGE.keys())

    # item x model mean-score matrix (mean over reps), plus binarized verdict (score>=0.5)
    item_ids = [i for i in items if i in S]
    M = np.full((len(item_ids), len(models)), np.nan)
    for r, iid in enumerate(item_ids):
        for c, m in enumerate(models):
            reps = [v for v in S[iid][m].values() if v is not None]
            if reps:
                M[r, c] = np.mean(reps)
    keys = np.array([items[i]["key"] for i in item_ids])          # 1 correct, 0 incorrect
    domains = np.array([items[i]["domain"] for i in item_ids])
    disso = np.array([items[i]["dissociation"] for i in item_ids])

    # judge accuracy vs key (verdict = score>=0.5 means "correct")
    acc = {}
    for c, m in enumerate(models):
        pred = (M[:, c] >= 0.5).astype(float)
        ok = ~np.isnan(M[:, c])
        acc[m] = float((pred[ok] == keys[ok]).mean())

    # ERROR relative to key (the anchor): e = score - key. Decompose common-mode vs orthogonal.
    E = M - keys[:, None]                                          # item x model error
    valid = ~np.isnan(E).any(axis=1)
    Ev = E[valid]
    hatVar_o = np.nanmean(np.nanvar(Ev, axis=1, ddof=1))          # within-item across-judge (orthogonal)
    item_mean_err = Ev.mean(axis=1)
    raw_shared = max(item_mean_err.var(ddof=1) - hatVar_o / len(models), 0.0)
    # interval (eps~0 for keyed anchor; rho band [0,0.7] as conservative default; delta measured below)
    def phi_at(r, eps=0.0):
        vc = max((raw_shared - eps**2) / max((1-r)**2, 1e-6), 0.0)
        return vc/(vc+hatVar_o) if vc+hatVar_o>0 else float("nan")
    phi_lo, phi_hi = phi_at(0.0), phi_at(0.7)

    # between-lineage means (df check)
    lin_means = {}
    for L in set(LINEAGE.values()):
        cols = [i for i,m in enumerate(models) if LINEAGE[m]==L]
        lin_means[L] = float(np.nanmean(M[:, cols]))

    # R1 dissociation delta probe: on dissociation items, correlation of anchor-error direction w/ judges
    # (proxy: mean judge error on dissociation-wrong items — if judges also 'accept' surface-plausible
    #  wrong answers, that's shared blind spot => delta>0)
    disso_wrong = valid & disso & (keys==0)
    delta_proxy = float(np.nanmean(M[disso_wrong].mean(axis=1))) if disso_wrong.any() else None

    # per-domain phi
    per_domain = {}
    for d in set(domains):
        mask = valid & (domains==d)
        Ed = (M[mask]-keys[mask,None])
        if mask.sum()>5 and not np.isnan(Ed).any():
            vo = np.nanmean(np.nanvar(Ed,axis=1,ddof=1))
            rs = max(Ed.mean(axis=1).var(ddof=1)-vo/len(models),0.0)
            per_domain[d] = {"n":int(mask.sum()),"phi_lo":rs/(rs+vo) if rs+vo>0 else None}

    res = {
        "n_items_used": int(valid.sum()), "n_models": len(models),
        "judge_accuracy_vs_key": {m: round(a,3) for m,a in acc.items()},
        "mean_accuracy": round(float(np.mean(list(acc.values()))),3),
        "hatVar_o": round(hatVar_o,5), "raw_shared_common_mode": round(raw_shared,5),
        "phi_interval_conditional_upper_bound": [round(phi_lo,3), round(phi_hi,3)],
        "between_lineage_means": {k:round(v,3) for k,v in lin_means.items()},
        "df_between_lineage": len(lin_means)-1,
        "dissociation_delta_proxy_meanscore_on_plausible_wrong": (round(delta_proxy,3) if delta_proxy is not None else None),
        "per_domain_phi_lo": {d:(round(v["phi_lo"],3) if v["phi_lo"] else None) for d,v in per_domain.items()},
        "per_domain_n": {d:v["n"] for d,v in per_domain.items()},
    }
    json.dump(res, open(OUT,"w"), indent=2)
    print(json.dumps(res, indent=2))
    print("\nWrote", OUT)

if __name__ == "__main__":
    main()
