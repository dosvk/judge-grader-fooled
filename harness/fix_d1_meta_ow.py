"""D1 fix: grade the domain-matched obvious-wrong set with the TWO META judges (the build script used
the original 6-judge PANEL and never ran Meta). Appends to raw_capture_full.sqlite. Resumable."""
import os, json, time
from concurrent.futures import ThreadPoolExecutor
from judge_harness import RawCapture, Judge
from bedrock_backend import BedrockBackend

HERE=os.path.dirname(os.path.abspath(__file__)); DB=os.path.join(HERE,"raw_capture_full.sqlite"); K=3
OWM=os.path.join(HERE,"items_ow_matched.jsonl")
META=[Judge("meta","us.meta.llama3-3-70b-instruct-v1:0","us-east-1"),
      Judge("meta","us.meta.llama4-maverick-17b-instruct-v1:0","us-east-1")]
def main():
    items=[json.loads(l) for l in open(OWM)]
    cap=RawCapture(path=DB); be=BedrockBackend()
    cached=set(r[0] for r in cap.db.execute("SELECT call_hash FROM calls").fetchall())
    work=[]
    for it in items:
        for j in META:
            for rep in range(K):
                req=json.dumps({"item":it["item_id"],"prompt":it["prompt"],"rep":rep})
                if cap.key(j.model_id,it["item_id"],rep,req) not in cached: work.append((it,j,rep,req))
    print(f"[start] items={len(items)} todo={len(work)} (expect 260x2x3=1560)",flush=True)
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
