"""Regenerate all figures from canonical_results.json (single source of truth). Offline."""
import os, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE=os.path.dirname(os.path.abspath(__file__)); FIG=os.path.join(HERE,"figs")
R=json.load(open(os.path.join(HERE,"canonical_results.json")))
order=["Opus-4.5","Sonnet-4.5","Haiku-4.5","Qwen3-32B","Nova-Pro","Nova-Lite","Llama-3.3-70B","Llama-4-Maverick"]

# Fig1: accuracy + per-judge plausible-wrong acceptance
acc=[R["accuracy_by_judge"][k] for k in order]
pw=[R["plausible_wrong_acceptance_by_judge"][k] for k in order]
fig,ax=plt.subplots(1,2,figsize=(12,4))
ax[0].bar(order,acc,color="steelblue"); ax[0].set_title("(a) Accuracy vs objective key"); ax[0].set_ylim(0,1)
ax[0].axhline(R["accuracy_mean"],ls="--",c="gray",label=f"mean {R['accuracy_mean']}"); ax[0].legend(); ax[0].tick_params(axis='x',rotation=40)
ax[1].bar(order,pw,color="darkorange"); ax[1].set_title("(b) Acceptance of surface-plausible WRONG answers"); ax[1].set_ylim(0,0.5); ax[1].tick_params(axis='x',rotation=40)
plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig1_accuracy_leniency.png"),dpi=130); plt.close()

# Fig2: plausible vs obvious wrong
ow=[R["obvious_wrong_acceptance_by_judge"][k] for k in order]
x=np.arange(len(order)); w=0.38
fig,ax=plt.subplots(figsize=(10,4))
ax.bar(x-w/2,pw,w,label="surface-plausible wrong",color="darkorange")
ax.bar(x+w/2,ow,w,label="obvious wrong",color="lightgray")
ax.set_xticks(x); ax.set_xticklabels(order,rotation=40); ax.set_ylabel("acceptance rate"); ax.legend()
ax.set_title("Erroneous acceptance of incorrect answers (lower is better)")
plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig2_plausible_wrong.png"),dpi=130); plt.close()

# Fig3: decomposition collapse + rubric lever
d=R["decomposition"]; lev=R["rubric_lever"]
fig,ax=plt.subplots(1,2,figsize=(12,4))
ax[0].bar(["raw shared","between-lineage\n(item-demeaned)"],[d["raw_shared"],d["between_lineage_after_demean"]],
          yerr=[[d["raw_shared"]-d["raw_shared_CI95"][0],d["between_lineage_after_demean"]-d["between_lineage_CI95"][0]],
                [d["raw_shared_CI95"][1]-d["raw_shared"],d["between_lineage_CI95"][1]-d["between_lineage_after_demean"]]],
          color=["gray","crimson"],capsize=5)
ax[0].set_title("(a) Shared 'error' collapses when\nitem difficulty is removed")
lins=["anthropic","qwen","amazon","meta"]
b=[lev["baseline_by_lineage"][L] for L in lins]; s=[lev["strict_by_lineage"][L] for L in lins]
xx=np.arange(len(lins))
ax[1].bar(xx-0.2,b,0.4,label="baseline rubric",color="darkorange")
ax[1].bar(xx+0.2,s,0.4,label="strict verify-first",color="teal")
ax[1].set_xticks(xx); ax[1].set_xticklabels(lins); ax[1].set_ylabel("plausible-wrong acceptance")
ax[1].set_title("(b) Strict rubric helps unevenly"); ax[1].legend()
plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig3_controlled.png"),dpi=130); plt.close()
print("regenerated fig1/fig2/fig3 from canonical_results.json")
