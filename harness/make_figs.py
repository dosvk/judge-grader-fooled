"""Generate paper figures from cached results (offline, matplotlib). Saves PNGs + the data behind them."""
import os, json, sqlite3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

HERE=os.path.dirname(os.path.abspath(__file__)); FIG=os.path.join(HERE,"figs"); os.makedirs(FIG,exist_ok=True)
DB=os.path.join(HERE,"raw_capture_full.sqlite"); POOL=os.path.join(HERE,"items_pool.jsonl")
LIN={"us.anthropic.claude-opus-4-5-20251101-v1:0":("anthropic","Opus-4.5"),
 "us.anthropic.claude-sonnet-4-5-20250929-v1:0":("anthropic","Sonnet-4.5"),
 "us.anthropic.claude-haiku-4-5-20251001-v1:0":("anthropic","Haiku-4.5"),
 "amazon.nova-pro-v1:0":("amazon","Nova-Pro"),"amazon.nova-lite-v1:0":("amazon","Nova-Lite"),
 "qwen.qwen3-32b-v1:0":("qwen","Qwen3-32B"),"us.meta.llama3-3-70b-instruct-v1:0":("meta","Llama3.3-70B"),"us.meta.llama4-maverick-17b-instruct-v1:0":("meta","Llama4-Mav"),
 "openai.gpt-oss-120b-1:0":("openai","GPT-OSS-120B"),"openai.gpt-oss-20b-1:0":("openai","GPT-OSS-20B"),
 "google.gemma-3-27b-it":("google","Gemma-3-27B"),"google.gemma-3-12b-it":("google","Gemma-3-12B")}
items={json.loads(l)["item_id"]:json.loads(l) for l in open(POOL)}
db=sqlite3.connect(DB); S=defaultdict(lambda:defaultdict(list))
for m,i,rep,rj in db.execute("SELECT model_id,item_id,rep,response_json FROM calls"):
    try: sc=json.loads(rj).get("score")
    except: sc=None
    if sc is not None: S[i][m].append(sc)
models=list(LIN.keys()); ids=[i for i in items if i in S and all(m in S[i] for m in models)]
M=np.array([[np.mean(S[i][m]) for m in models] for i in ids]); keys=np.array([items[i]["key"] for i in ids])
labels=[LIN[m][1] for m in models]

# Fig1: accuracy vs key + leniency
acc=[((M[:,k]>=0.5).astype(float)==keys).mean() for k in range(len(models))]
leniency=[M[:,k].mean() for k in range(len(models))]
fig,ax=plt.subplots(1,2,figsize=(11,4))
ax[0].bar(labels,acc,color="steelblue"); ax[0].set_title("Judge accuracy vs objective key"); ax[0].set_ylim(0,1)
ax[0].axhline(np.mean(acc),ls="--",c="gray",label=f"mean {np.mean(acc):.2f}"); ax[0].legend(); ax[0].tick_params(axis='x',rotation=30)
ax[1].bar(labels,leniency,color="indianred"); ax[1].set_title("Leniency (mean score; higher=more accepting)"); ax[1].set_ylim(0,1); ax[1].tick_params(axis='x',rotation=30)
plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig1_accuracy_leniency.png"),dpi=130); plt.close()

# Fig2: ACCEPT-RATE (mean-of-reps>=0.5) on plausible-wrong vs the DOMAIN-MATCHED obvious-wrong control
disso=np.array([items[i]["dissociation"] for i in ids])
plausible_wrong=(keys==0)&disso
acc_pw=[(M[plausible_wrong,k]>=0.5).mean() for k in range(len(models))]
owm={json.loads(l)["item_id"]:json.loads(l) for l in open(os.path.join(HERE,"items_ow_matched.jsonl"))}
acc_ow=[]
for m,_ in [(mm,None) for mm in models]:
    cells=[np.mean(S[i][m])>=0.5 for i in owm if S[i].get(m)]
    acc_ow.append(np.mean(cells))
x=np.arange(len(models)); w=0.38
fig,ax=plt.subplots(figsize=(9,4))
ax.bar(x-w/2,acc_pw,w,label="surface-plausible wrong",color="darkorange")
ax.bar(x+w/2,acc_ow,w,label="domain-matched obvious wrong",color="lightgray")
ax.set_xticks(x); ax.set_xticklabels(labels,rotation=30); ax.set_ylabel("acceptance rate (mean of reps >= 0.5)")
ax.set_title("Erroneous acceptance of INCORRECT answers"); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig2_plausible_wrong.png"),dpi=130); plt.close()

# Fig3: difficulty-controlled between-lineage leniency + pairwise corr (from controlled json)
ctrl=json.load(open(os.path.join(HERE,"decomposition_controlled.json")))
fig,ax=plt.subplots(1,2,figsize=(11,4))
ll=ctrl["lineage_leniency_meanscore"]; ax[0].bar(list(ll.keys()),list(ll.values()),color="teal")
ax[0].set_title("Lineage leniency (mean score)"); ax[0].set_ylim(0,0.7)
raw=ctrl["raw_common_mode"]; surv=ctrl["AFTER_item_demeaning"]["between_lineage_var_(leniency_spread)"]
ax[1].bar(["raw shared","after item-demean"],[raw,surv],color=["gray","crimson"])
ax[1].set_title("Shared component collapses when\nitem difficulty removed")
plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig3_controlled.png"),dpi=130); plt.close()

json.dump({"labels":labels,"accuracy":acc,"leniency":leniency,"acc_plausible_wrong":acc_pw},
          open(os.path.join(FIG,"figure_data.json"),"w"),indent=2,default=float)
print("wrote fig1_accuracy_leniency.png, fig2_plausible_wrong.png, fig3_controlled.png + figure_data.json")
print("accuracy:",{l:round(a,3) for l,a in zip(labels,acc)})
print("plausible-wrong MEAN SCORE (not thresholded accept rate; see canonical.py for rates):",{l:round(a,3) for l,a in zip(labels,acc_pw)})
