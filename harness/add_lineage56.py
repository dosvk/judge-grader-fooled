"""
FULL collection for panel expansion (PREREG Amendment 2): L5 OpenAI (gpt-oss-120b/20b), L6 Google
(Gemma-3-27b/12b), + gpt-5.6-sol auxiliary arm. Replicates the exact arm construction and request
formats of the existing panel so cache keys and canonical.py analysis remain compatible:
  A base pool        -> raw_capture_full.sqlite   req {item,prompt,rep}          judge instruction
  B ow-matched ctrl  -> raw_capture_full.sqlite   req {item,prompt,rep}          judge instruction
  C strict PW regrade-> raw_capture_rubric.sqlite req {item,prompt,rep,mode}     strict instruction
  D strict correct   -> raw_capture_rubric.sqlite req {item,prompt,rep,mode}     strict instruction
Resumable (append-only immutable cache). Run under nohup until 'ALL ARMS DONE'.
"""
import os, json, time
from concurrent.futures import ThreadPoolExecutor
from judge_harness import RawCapture, Judge
from bedrock_backend import JUDGE_INSTRUCTION, _extract
from rubric_lever import STRICT_INSTRUCTION, target_items
from d7_correct_strict import correct_sample
import boto3

HERE = os.path.dirname(os.path.abspath(__file__))
K = 3; WORKERS = 10

NEW_PANEL = [
    Judge("openai", "openai.gpt-oss-120b-1:0", "us-east-1"),
    Judge("openai", "openai.gpt-oss-20b-1:0", "us-east-1"),
    Judge("google", "google.gemma-3-27b-it", "us-east-1"),
    Judge("google", "google.gemma-3-12b-it", "us-east-1"),
    Judge("openai-aux", "us.openai.gpt-5.6-sol", "us-east-1"),
]

class XBackend:
    """Multi-block-safe, no-temp-safe backend, parametrized by instruction."""
    def __init__(self, instruction, region="us-east-1"):
        self.rt = boto3.client("bedrock-runtime", region_name=region)
        self.instruction = instruction
    def call_model(self, judge, item_id, prompt, rep):
        needs_big = ("gpt-oss" in judge.model_id) or ("gpt-5.6" in judge.model_id)
        cfg = {"maxTokens": 512 if needs_big else 400}
        if "gpt-5.6" not in judge.model_id:
            cfg["temperature"] = 0
        body = {"messages": [{"role": "user", "content": [{"text": self.instruction + "\n\n" + prompt}]}],
                "inferenceConfig": cfg}
        for attempt in range(6):
            try:
                r = self.rt.converse(modelId=judge.model_id, **body)
                blocks = r["output"]["message"]["content"]
                txt = next((b["text"] for b in reversed(blocks) if "text" in b), "")
                sc, obj = _extract(txt)
                return {"score": sc, "raw_text": txt, "parsed": obj, "n_blocks": len(blocks),
                        "model_id": judge.model_id, "rep": rep, "usage": r.get("usage", {})}
            except Exception as e:
                if attempt == 5:
                    return {"score": None, "error": str(e)[:200], "model_id": judge.model_id, "rep": rep}
                time.sleep(1.5 * (attempt + 1))

def load_jsonl(p):
    return [json.loads(l) for l in open(p)]

def run_arm(name, items, db_path, backend, mode=None):
    cap = RawCapture(path=db_path)
    cached = set(r[0] for r in cap.db.execute("SELECT call_hash FROM calls").fetchall())
    work = []
    for it in items:
        for j in NEW_PANEL:
            for rep in range(K):
                d = {"item": it["item_id"], "prompt": it["prompt"], "rep": rep}
                if mode: d["mode"] = mode
                req = json.dumps(d)
                if cap.key(j.model_id, it["item_id"], rep, req) not in cached:
                    work.append((it, j, rep, req))
    target = len(items) * len(NEW_PANEL) * K
    print(f"[{name}] target={target} todo={len(work)}", flush=True)
    made = 0; t0 = time.time()
    def task(w):
        it, j, rep, req = w
        return (j, it, rep, req, backend.call_model(j, it["item_id"], it["prompt"], rep))
    for s in range(0, len(work), 200):
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for j, it, rep, req, resp in ex.map(task, work[s:s + 200]):
                cap.put(j.lineage, j.model_id, j.region, it["item_id"], rep, req, resp); made += 1
        rate = made / max(time.time() - t0, 1)
        print(f"[{name}] {made}/{len(work)} ({rate:.1f}/s)", flush=True)
    print(f"[{name}] DONE made={made}", flush=True)

def main():
    full_db = os.path.join(HERE, "raw_capture_full.sqlite")
    rubric_db = os.path.join(HERE, "raw_capture_rubric.sqlite")
    judge_be = XBackend(JUDGE_INSTRUCTION)
    strict_be = XBackend(STRICT_INSTRUCTION)
    run_arm("A-base", load_jsonl(os.path.join(HERE, "items_pool.jsonl")), full_db, judge_be)
    run_arm("B-owctrl", load_jsonl(os.path.join(HERE, "items_ow_matched.jsonl")), full_db, judge_be)
    run_arm("C-strictPW", target_items(), rubric_db, strict_be, mode="strict")
    run_arm("D-strictOK", correct_sample(), rubric_db, strict_be, mode="strict")
    print("ALL ARMS DONE", flush=True)

if __name__ == "__main__":
    main()
