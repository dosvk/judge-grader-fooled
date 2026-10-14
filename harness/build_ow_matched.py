"""Build DOMAIN-MATCHED obvious-wrong items for factual + code (the OW set was all-math, a confound).
Obvious-wrong = blatantly, unmistakably incorrect answer in the SAME domains as plausible-wrong.
Grade them with all 8 judges (baseline rubric), append to raw_capture_full.sqlite. Resumable."""
import os, json, time
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
from judge_harness import RawCapture
from bedrock_backend import BedrockBackend, PANEL

HERE=os.path.dirname(os.path.abspath(__file__)); DB=os.path.join(HERE,"raw_capture_full.sqlite"); K=3
OUT=os.path.join(HERE,"items_ow_matched.jsonl")

def build():
    import random; rng=random.Random(20260817)
    items=[]
    # factual obvious-wrong: answer with a nonsense/blatant token
    s=load_dataset("allenai/sciq",split="test")
    for i in range(135):
        q=s[i]["question"].strip()
        items.append({"item_id":f"owf{i:04d}","domain":"factual",
                      "prompt":f"QUESTION: {q}\nANSWER: purple","key":0,"dissociation":False,"ow_matched":True})
    # code obvious-wrong: candidate is obviously unrelated/garbage
    m=load_dataset("google-research-datasets/mbpp","sanitized",split="test")
    for i in range(125):
        task=m[i]["prompt"].strip(); test=m[i]["test_list"][0] if m[i]["test_list"] else ""
        items.append({"item_id":f"owc{i:04d}","domain":"code",
                      "prompt":f"TASK: {task}\nCANDIDATE:\ndef f():\n    return 0  # unrelated stub\nTEST: {test}",
                      "key":0,"dissociation":False,"ow_matched":True})
    open(OUT,"w").write("\n".join(json.dumps(it) for it in items))
    return items

def main():
    items=build() if not os.path.exists(OUT) else [json.loads(l) for l in open(OUT)]
    cap=RawCapture(path=DB); be=BedrockBackend()
    cached=set(r[0] for r in cap.db.execute("SELECT call_hash FROM calls").fetchall())
    work=[]
    for it in items:
        for j in PANEL:
            for rep in range(K):
                req=json.dumps({"item":it["item_id"],"prompt":it["prompt"],"rep":rep})
                if cap.key(j.model_id,it["item_id"],rep,req) not in cached: work.append((it,j,rep,req))
    print(f"[start] matched-OW items={len(items)} todo={len(work)}",flush=True)
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
