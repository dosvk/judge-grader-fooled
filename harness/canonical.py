"""
CANONICAL analysis — single source of truth for every number in the paper.
Fixes the Gate-2/Fig discrepancy: ONE definition of 'accept', ONE item set, ONE plausible-wrong set.
All paper numbers must come from here. Writes canonical_results.json. Deterministic, offline.

Definitions (frozen):
- item set: items graded by ALL 8 judges (complete cases).
- per-(judge,item) score: mean over K reps.
- ACCEPT(judge,item) := mean_score >= 0.5   (binary verdict "answer is correct")
- accuracy(judge) := mean over items of [ ACCEPT == (key==1) ]
- plausible-wrong set PW := items with key==0 AND dissociation==True
- plausible-wrong ACCEPTANCE(judge) := mean over PW of ACCEPT   (fraction of plausible-wrong wrongly accepted)
- lineage value := mean over that lineage's judges of the judge-level statistic (equal weight per judge)
"""
import os, json, sqlite3
import numpy as np
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
POOL = os.path.join(HERE, "items_pool.jsonl")
DB   = os.path.join(HERE, "raw_capture_full.sqlite")
DB_R = os.path.join(HERE, "raw_capture_rubric.sqlite")
OUT  = os.path.join(HERE, "canonical_results.json")

JUDGES = [  # (model_id, lineage, short)
 ("us.anthropic.claude-opus-4-5-20251101-v1:0","anthropic","Opus-4.5"),
 ("us.anthropic.claude-sonnet-4-5-20250929-v1:0","anthropic","Sonnet-4.5"),
 ("us.anthropic.claude-haiku-4-5-20251001-v1:0","anthropic","Haiku-4.5"),
 ("amazon.nova-pro-v1:0","amazon","Nova-Pro"),
 ("amazon.nova-lite-v1:0","amazon","Nova-Lite"),
 ("qwen.qwen3-32b-v1:0","qwen","Qwen3-32B"),
 ("us.meta.llama3-3-70b-instruct-v1:0","meta","Llama-3.3-70B"),
 ("us.meta.llama4-maverick-17b-instruct-v1:0","meta","Llama-4-Maverick"),
 ("openai.gpt-oss-120b-1:0","openai","GPT-OSS-120B"),
 ("openai.gpt-oss-20b-1:0","openai","GPT-OSS-20B"),
 ("google.gemma-3-27b-it","google","Gemma-3-27B"),
 ("google.gemma-3-12b-it","google","Gemma-3-12B"),
]
# AUX (pre-reg Amendment 2): us.openai.gpt-5.6-sol runs the same items but its endpoint rejects the
# temperature parameter (provider-fixed sampling); reported separately, EXCLUDED from the uniform panel.
AUX_JUDGE = ("us.openai.gpt-5.6-sol","openai-aux","GPT-5.6-sol")
MODELS=[j[0] for j in JUDGES]; LINEAGE={j[0]:j[1] for j in JUDGES}; SHORT={j[0]:j[2] for j in JUDGES}
ACCEPT_THRESH=0.5

def load_scores(db):
    S=defaultdict(lambda:defaultdict(list))
    for m,i,rep,rj in sqlite3.connect(db).execute("SELECT model_id,item_id,rep,response_json FROM calls"):
        try: sc=json.loads(rj).get("score")
        except Exception: sc=None
        if sc is not None: S[i][m].append(float(sc))
    return S

def mean_score_matrix(S, ids):
    return {i:{m:(np.mean(S[i][m]) if S[i].get(m) else np.nan) for m in MODELS} for i in ids}

def main():
    items={json.loads(l)["item_id"]:json.loads(l) for l in open(POOL)}
    S=load_scores(DB)
    # complete cases: all 8 judges present (>=1 rep each)
    ids=[i for i in items if all(S[i].get(m) for m in MODELS)]
    MS=mean_score_matrix(S, ids)
    key={i:items[i]["key"] for i in ids}
    dom={i:items[i]["domain"] for i in ids}
    disso={i:items[i]["dissociation"] for i in ids}
    def accept(i,m): return 1.0 if MS[i][m]>=ACCEPT_THRESH else 0.0

    # accuracy per judge
    acc={SHORT[m]: round(float(np.mean([1.0 if accept(i,m)==key[i] else 0.0 for i in ids])),3) for m in MODELS}
    # plausible-wrong acceptance per judge
    PW=[i for i in ids if key[i]==0 and disso[i]]
    pw_acc={SHORT[m]: round(float(np.mean([accept(i,m) for i in PW])),3) for m in MODELS}
    # obvious-wrong acceptance (key==0, not dissociation) for contrast
    OW=[i for i in ids if key[i]==0 and not disso[i]]
    ow_acc={SHORT[m]: (round(float(np.mean([accept(i,m) for i in OW])),3) if OW else None) for m in MODELS}
    # lineage-level (equal weight per judge)
    lineages=sorted(set(LINEAGE.values()))
    def lin_stat(perjudge):
        return {L: round(float(np.mean([perjudge[SHORT[m]] for m in MODELS if LINEAGE[m]==L])),3) for L in lineages}
    lin_pw=lin_stat(pw_acc); lin_leniency=lin_stat({SHORT[m]:round(float(np.mean([MS[i][m] for i in ids])),3) for m in MODELS})
    # per-domain plausible-wrong per lineage
    per_dom={}
    for d in sorted(set(dom.values())):
        PWd=[i for i in PW if dom[i]==d]
        if PWd:
            per_dom[d]={L: round(float(np.mean([accept(i,m) for i in PWd for m in MODELS if LINEAGE[m]==L])),3) for L in lineages}

    # decomposition: error e=mean_score-key ; raw shared vs orthogonal; + item-demeaned between-lineage
    E=np.array([[MS[i][m]-key[i] for m in MODELS] for i in ids])
    vo=float(np.mean(np.var(E,axis=1,ddof=1)))
    raw_shared=float(max(E.mean(axis=1).var(ddof=1)-vo/len(MODELS),0.0))
    linmeans=np.array([[np.mean([MS[i][m]-key[i] for m in MODELS if LINEAGE[m]==L]) for L in lineages] for i in ids])
    between_lineage=float(linmeans.mean(axis=0).var(ddof=1))
    # bootstrap CIs
    rng=np.random.default_rng(20260816); n=len(ids); raws=[];bows=[]
    for _ in range(2000):
        idx=rng.integers(0,n,n)
        Eb=E[idx]; vob=np.mean(np.var(Eb,axis=1,ddof=1))
        raws.append(max(Eb.mean(axis=1).var(ddof=1)-vob/len(MODELS),0.0))
        lb=linmeans[idx]; bows.append(lb.mean(axis=0).var(ddof=1))
    ci=lambda a:[round(float(np.percentile(a,2.5)),5),round(float(np.percentile(a,97.5)),5)]

    # rubric lever (strict) plausible-wrong acceptance, same definitions
    SR=load_scores(DB_R)
    def pw_accept_from(Sx):
        out={}
        for L in lineages:
            ms=[m for m in MODELS if LINEAGE[m]==L]
            vals=[1.0 if (np.mean(Sx[i][m])>=ACCEPT_THRESH) else 0.0 for i in PW for m in ms if Sx[i].get(m)]
            out[L]=round(float(np.mean(vals)),3) if vals else None
        return out
    lever={"baseline_by_lineage":pw_accept_from(S),"strict_by_lineage":pw_accept_from(SR)}

    # domain-matched obvious-wrong CONTROL (separate arm; disjoint item_ids from the pool)
    owm_items={json.loads(l)["item_id"]:json.loads(l) for l in open(os.path.join(HERE,"items_ow_matched.jsonl"))}
    Sow=load_scores(DB)
    owm_acc={}; owm_n={}
    for m in MODELS:
        cells=[np.mean(Sow[i][m]) >= ACCEPT_THRESH for i in owm_items if Sow[i].get(m)]
        owm_acc[SHORT[m]]=round(float(np.mean(cells)),4); owm_n[SHORT[m]]=len(cells)
    res={
     "n_items_complete_cases":len(ids),"n_plausible_wrong":len(PW),"n_mathw":len(OW),
     "accept_threshold":ACCEPT_THRESH,"K":3,"n_judges":len(MODELS),"n_lineages":len(lineages),
     "accuracy_by_judge":acc,"accuracy_mean":round(float(np.mean(list(acc.values()))),3),
     "plausible_wrong_acceptance_by_judge":pw_acc,
     "mathw_acceptance_by_judge":ow_acc,
     "plausible_wrong_by_lineage":lin_pw,"leniency_by_lineage":lin_leniency,
     "plausible_wrong_by_domain_lineage":per_dom,
     "decomposition":{"orthogonal_var":round(vo,5),"raw_shared":round(raw_shared,5),
        "raw_shared_CI95":ci(raws),"between_lineage_after_demean":round(between_lineage,5),
        "between_lineage_CI95":ci(bows)},
     "matched_obvious_wrong_control_by_judge":owm_acc,
     "matched_obvious_wrong_n":owm_n,
     "rubric_lever":lever,
    }
    json.dump(res,open(OUT,"w"),indent=2)
    print(json.dumps(res,indent=2))
    print("\nWrote",OUT)

if __name__=="__main__":
    main()
