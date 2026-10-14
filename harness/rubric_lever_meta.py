"""Finish strict-rubric grading for the Meta (L4) judges on plausible-wrong items (the missing cells).
Appends into raw_capture_rubric.sqlite. Resumable."""
import os, json, time
from concurrent.futures import ThreadPoolExecutor
from judge_harness import RawCapture, Judge
from rubric_lever import StrictBackend, target_items

HERE=os.path.dirname(os.path.abspath(__file__)); DB=os.path.join(HERE,"raw_capture_rubric.sqlite"); K=3
L4=[Judge("meta","us.meta.llama3-3-70b-instruct-v1:0","us-east-1"),
    Judge("meta","us.meta.llama4-maverick-17b-instruct-v1:0","us-east-1")]
def main():
    items=target_items(); cap=RawCapture(path=DB); be=StrictBackend()
    cached=set(r[0] for r in cap.db.execute("SELECT call_hash FROM calls").fetchall())
    work=[]
    for it in items:
        for j in L4:
            for rep in range(K):
                req=json.dumps({"item":it["item_id"],"prompt":it["prompt"],"rep":rep,"mode":"strict"})
                if cap.key(j.model_id,it["item_id"],rep,req) not in cached: work.append((it,j,rep,req))
    print(f"[start] todo={len(work)}",flush=True)
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
