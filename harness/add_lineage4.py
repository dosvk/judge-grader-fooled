"""Add L4 = Meta Llama lineage to the full baseline run. Grades all 776 items with 2 new judges,
K=3, appending into raw_capture_full.sqlite. Resumable. Same JUDGE_INSTRUCTION as the baseline."""
import os, json, time
from concurrent.futures import ThreadPoolExecutor
from judge_harness import RawCapture, Judge
from bedrock_backend import BedrockBackend

HERE=os.path.dirname(os.path.abspath(__file__)); POOL=os.path.join(HERE,"items_pool.jsonl")
DB=os.path.join(HERE,"raw_capture_full.sqlite"); K=3; WORKERS=8

L4 = [Judge("meta","us.meta.llama3-3-70b-instruct-v1:0","us-east-1"),
      Judge("meta","us.meta.llama4-maverick-17b-instruct-v1:0","us-east-1")]

def main():
    items=[json.loads(l) for l in open(POOL)]
    cap=RawCapture(path=DB); be=BedrockBackend()
    cached=set(r[0] for r in cap.db.execute("SELECT call_hash FROM calls").fetchall())
    work=[]
    for it in items:
        for j in L4:
            for rep in range(K):
                req=json.dumps({"item":it["item_id"],"prompt":it["prompt"],"rep":rep})
                if cap.key(j.model_id,it["item_id"],rep,req) not in cached:
                    work.append((it,j,rep,req))
    target=len(items)*len(L4)*K
    done=target-len(work)
    print(f"[start] L4 target={target} cached={done} todo={len(work)}",flush=True)
    if not work: print("ALL DONE"); return
    t0=time.time(); made=0
    def task(w):
        it,j,rep,req=w; return (j,it,rep,req,be.call_model(j,it["item_id"],it["prompt"],rep))
    for s in range(0,len(work),WORKERS*20):
        if time.time()-t0>220: break
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for j,it,rep,req,resp in ex.map(task,work[s:s+WORKERS*20]):
                cap.put(j.lineage,j.model_id,j.region,it["item_id"],rep,req,resp); made+=1
    print(f"[run] made={made} rows_new_lineage_done={done+made}/{target} "
          f"{'ALL DONE' if done+made>=target else 'PARTIAL — re-run'}",flush=True)

if __name__=="__main__":
    main()
