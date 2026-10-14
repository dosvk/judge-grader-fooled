"""D7: measure the strict rubric's FALSE-REJECTION cost on CORRECT answers (the lever was one-sided).
Sample 250 correct items (stratified by domain), grade with all 8 judges under the STRICT rubric,
compare accept-rate vs baseline. Appends to raw_capture_rubric.sqlite (mode=strict keys distinct)."""
import os, json, time
from concurrent.futures import ThreadPoolExecutor
from judge_harness import RawCapture, Judge
from rubric_lever import StrictBackend
from bedrock_backend import PANEL

HERE=os.path.dirname(os.path.abspath(__file__)); DB=os.path.join(HERE,"raw_capture_rubric.sqlite"); K=3
META=[Judge("meta","us.meta.llama3-3-70b-instruct-v1:0","us-east-1"),
      Judge("meta","us.meta.llama4-maverick-17b-instruct-v1:0","us-east-1")]
ALL8=PANEL+META

def correct_sample(n=250):
    import random; rng=random.Random(20260818)
    items=[json.loads(l) for l in open(os.path.join(HERE,"items_pool.jsonl"))]
    corr=[it for it in items if it["key"]==1]
    by={}
    for it in corr: by.setdefault(it["domain"],[]).append(it)
    out=[]
    per=n//len(by)
    for d,v in by.items():
        rng.shuffle(v); out+=v[:per]
    return out

def main():
    items=correct_sample()
    cap=RawCapture(path=DB); be=StrictBackend()
    cached=set(r[0] for r in cap.db.execute("SELECT call_hash FROM calls").fetchall())
    work=[]
    for it in items:
        for j in ALL8:
            for rep in range(K):
                req=json.dumps({"item":it["item_id"],"prompt":it["prompt"],"rep":rep,"mode":"strict"})
                if cap.key(j.model_id,it["item_id"],rep,req) not in cached: work.append((it,j,rep,req))
    print(f"[start] correct-sample={len(items)} todo={len(work)}",flush=True)
    if not work: print("ALL DONE"); return
    t0=time.time(); made=0
    def task(w):
        it,j,rep,req=w; return (j,it,rep,req,be.call_model(j,it["item_id"],it["prompt"],rep))
    for s in range(0,len(work),160):
        if time.time()-t0>220: break
        with ThreadPoolExecutor(max_workers=8) as ex:
            for j,it,rep,req,resp in ex.map(task,work[s:s+160]):
                cap.put(j.lineage,j.model_id,j.region,it["item_id"],rep,req,resp); made+=1
    print(f"[run] made={made} {'ALL DONE' if made==len(work) else 'PARTIAL — re-run'}",flush=True)
if __name__=="__main__": main()
