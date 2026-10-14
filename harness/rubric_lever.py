"""
Rubric-lever experiment (Gate-4 fix #2): does a STRICT verify-first rubric close the plausible-wrong
acceptance gap, esp. for Nova? Re-grade ONLY the surface-plausible-wrong items (keyed incorrect,
dissociation=True) under a stricter instruction, compare to baseline acceptance. Resumable, cached
separately (raw_capture_rubric.sqlite). Offline analysis after.
"""
import os, re, json, time, sqlite3
from concurrent.futures import ThreadPoolExecutor
import boto3
from judge_harness import RawCapture, Judge
from bedrock_backend import PANEL, _extract

HERE = os.path.dirname(os.path.abspath(__file__))
POOL = os.path.join(HERE, "items_pool.jsonl")
DB = os.path.join(HERE, "raw_capture_rubric.sqlite")
K = 3
WORKERS = 8

# STRICT rubric: force explicit verification before a verdict, penalize accepting unverifiable answers.
STRICT_INSTRUCTION = (
    "You are a STRICT grader. Do NOT be swayed by fluent, well-formatted, or plausible-looking answers. "
    "First, independently work out the correct answer yourself. Then compare. An answer is 'correct' "
    "ONLY if it actually matches the truth you derived; surface plausibility is NOT sufficient. "
    "When in doubt, mark incorrect. Reply with ONLY compact JSON: "
    '{"correct": true|false, "score": <0..1 float>}. No other text.'
)


class StrictBackend:
    def __init__(self, region="us-east-1"):
        self.rt = boto3.client("bedrock-runtime", region_name=region)
    def call_model(self, judge, item_id, prompt, rep):
        body = {"messages": [{"role": "user", "content": [{"text": STRICT_INSTRUCTION + "\n\n" + prompt}]}],
                "inferenceConfig": {"maxTokens": 400, "temperature": 0}}  # more tokens for the verify step
        for attempt in range(5):
            try:
                r = self.rt.converse(modelId=judge.model_id, **body)
                txt = r["output"]["message"]["content"][0]["text"]
                sc, obj = _extract(txt)
                return {"score": sc, "raw_text": txt, "parsed": obj, "model_id": judge.model_id, "rep": rep}
            except Exception as e:
                if attempt == 4:
                    return {"score": None, "error": str(e)[:200], "model_id": judge.model_id, "rep": rep}
                time.sleep(1.5 * (attempt + 1))


def target_items():
    items = [json.loads(l) for l in open(POOL)]
    return [it for it in items if it["key"] == 0 and it["dissociation"]]  # surface-plausible wrong only


def main():
    items = target_items()
    cap = RawCapture(path=DB)
    be = StrictBackend()
    cached = set(r[0] for r in cap.db.execute("SELECT call_hash FROM calls").fetchall())
    work = []
    for it in items:
        for j in PANEL:
            for rep in range(K):
                req = json.dumps({"item": it["item_id"], "prompt": it["prompt"], "rep": rep, "mode": "strict"})
                if cap.key(j.model_id, it["item_id"], rep, req) not in cached:
                    work.append((it, j, rep, req))
    target = len(items) * len(PANEL) * K
    print(f"[start] plausible-wrong items={len(items)} cached={len(cached)}/{target} todo={len(work)}", flush=True)
    if not work:
        print("ALL DONE"); return
    t0 = time.time()
    def task(w):
        it, j, rep, req = w
        return (j, it, rep, req, be.call_model(j, it["item_id"], it["prompt"], rep))
    made = 0
    BATCH = WORKERS * 20
    for s in range(0, len(work), BATCH):
        if time.time() - t0 > 220:
            break
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for j, it, rep, req, resp in ex.map(task, work[s:s+BATCH]):
                cap.put(j.lineage, j.model_id, j.region, it["item_id"], rep, req, resp)
                made += 1
    print(f"[run] made={made} rows={cap.count()}/{target} {'ALL DONE' if cap.count()>=target else 'PARTIAL — re-run'}", flush=True)


if __name__ == "__main__":
    main()
