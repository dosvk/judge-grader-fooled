"""
Fast resumable driver: preload cached keys (one query), parallelize independent judge calls with a
thread pool, self-stop at ~230s. Re-invoke until ALL DONE.
"""
import os, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from judge_harness import RawCapture
from bedrock_backend import BedrockBackend, PANEL

HERE = os.path.dirname(os.path.abspath(__file__))
POOL = os.path.join(HERE, "items_pool.jsonl")
DB = os.path.join(HERE, "raw_capture_full.sqlite")
K = 3
WORKERS = 8

def main():
    items = [json.loads(l) for l in open(POOL)]
    target = len(items) * len(PANEL) * K
    cap = RawCapture(path=DB)
    # preload all cached keys once
    cached = set(r[0] for r in cap.db.execute("SELECT call_hash FROM calls").fetchall())
    be = BedrockBackend()

    # build worklist of only-missing (item, judge, rep)
    work = []
    for it in items:
        for j in PANEL:
            for rep in range(K):
                req = json.dumps({"item": it["item_id"], "prompt": it["prompt"], "rep": rep})
                if cap.key(j.model_id, it["item_id"], rep, req) not in cached:
                    work.append((it, j, rep, req))
    print(f"[start] cached={len(cached)}/{target} todo={len(work)}", flush=True)
    if not work:
        print("ALL DONE"); return

    t0 = time.time()
    made = 0
    def task(w):
        it, j, rep, req = w
        resp = be.call_model(j, it["item_id"], it["prompt"], rep)
        return (j, it, rep, req, resp)

    # bounded batches so we can exit cleanly before the shell timeout (no dangling executor)
    BATCH = WORKERS * 20
    for start in range(0, len(work), BATCH):
        if time.time() - t0 > 220:
            break
        batch = work[start:start + BATCH]
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for j, it, rep, req, resp in ex.map(task, batch):
                cap.put(j.lineage, j.model_id, j.region, it["item_id"], rep, req, resp)
                made += 1
    rows = cap.count()
    print(f"[run] made={made} cached_rows={rows}/{target} elapsed={time.time()-t0:.0f}s "
          f"{'ALL DONE' if rows>=target else 'PARTIAL — re-run'}", flush=True)

if __name__ == "__main__":
    main()
