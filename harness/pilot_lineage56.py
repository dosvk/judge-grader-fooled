"""
Pilot for panel expansion (PREREGISTRATION Amendment 2): L5 OpenAI open-weights, L6 Google Gemma-3,
plus gpt-5.6-sol auxiliary arm (provider-fixed sampling; temperature param rejected by endpoint).
30 keyed pilot items x 5 judges x K=3 -> raw_capture_pilot56.sqlite. Resumable; verbatim capture.
"""
import os, json, time
from concurrent.futures import ThreadPoolExecutor
from judge_harness import Judge, RawCapture
from bedrock_backend import BedrockBackend, JUDGE_INSTRUCTION, _extract, pilot_items, noise_floor

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "raw_capture_pilot56.sqlite")
K = 3; WORKERS = 8

NEW_PANEL = [
    Judge("openai", "openai.gpt-oss-120b-1:0", "us-east-1"),
    Judge("openai", "openai.gpt-oss-20b-1:0", "us-east-1"),
    Judge("google", "google.gemma-3-27b-it", "us-east-1"),
    Judge("google", "google.gemma-3-12b-it", "us-east-1"),
    Judge("openai-aux", "us.openai.gpt-5.6-sol", "us-east-1"),   # AUX: no temperature support
]

class ExpandedBackend(BedrockBackend):
    """Same contract as BedrockBackend; handles (a) reasoning-first multi-block content by taking the
    LAST text block, (b) endpoints that reject the temperature parameter, (c) larger token budget for
    reasoning models so the final JSON is not truncated."""
    def call_model(self, judge, item_id, prompt, rep):
        needs_big = ("gpt-oss" in judge.model_id) or ("gpt-5.6" in judge.model_id)
        cfg = {"maxTokens": 512 if needs_big else 120}
        if "gpt-5.6" not in judge.model_id:
            cfg["temperature"] = 0
        body = {"messages": [{"role": "user", "content": [{"text": JUDGE_INSTRUCTION + "\n\n" + prompt}]}],
                "inferenceConfig": cfg}
        for attempt in range(5):
            try:
                r = self.rt.converse(modelId=judge.model_id, **body)
                blocks = r["output"]["message"]["content"]
                txt = next((b["text"] for b in reversed(blocks) if "text" in b), "")
                sc, obj = _extract(txt)
                return {"score": sc, "raw_text": txt, "parsed": obj, "n_blocks": len(blocks),
                        "model_id": judge.model_id, "rep": rep, "usage": r.get("usage", {})}
            except Exception as e:
                if attempt == 4:
                    return {"score": None, "error": str(e)[:200], "model_id": judge.model_id, "rep": rep}
                time.sleep(1.5 * (attempt + 1))

def main():
    items = pilot_items(30)
    cap = RawCapture(path=DB); be = ExpandedBackend()
    cached = set(r[0] for r in cap.db.execute("SELECT call_hash FROM calls").fetchall())
    work = []
    for it in items:
        for j in NEW_PANEL:
            for rep in range(K):
                req = json.dumps({"item": it["item_id"], "prompt": it["prompt"], "rep": rep})
                if cap.key(j.model_id, it["item_id"], rep, req) not in cached:
                    work.append((it, j, rep, req))
    target = len(items) * len(NEW_PANEL) * K
    print(f"[start] pilot56 target={target} cached={target-len(work)} todo={len(work)}", flush=True)
    t0 = time.time(); made = 0
    def task(w):
        it, j, rep, req = w
        return (j, it, rep, req, be.call_model(j, it["item_id"], it["prompt"], rep))
    for s in range(0, len(work), WORKERS * 10):
        if time.time() - t0 > 260: break
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for j, it, rep, req, resp in ex.map(task, work[s:s + WORKERS * 10]):
                cap.put(j.lineage, j.model_id, j.region, it["item_id"], rep, req, resp); made += 1
    rows = cap.count()
    print(f"[run] made={made} rows={rows}/{target} elapsed={time.time()-t0:.0f}s "
          f"{'ALL DONE' if rows >= target else 'PARTIAL - rerun'}", flush=True)
    if rows >= target:
        # per-judge parse rate + accuracy vs key + noise floor
        import collections, statistics
        per = collections.defaultdict(lambda: {"n":0,"ok":0,"acc_n":0,"acc_ok":0})
        key = {it["item_id"]: it["key"] for it in items}
        for lineage, model_id, item_id, rep, resp_json in cap.db.execute(
                "SELECT lineage, model_id, item_id, rep, response_json FROM calls"):
            r = json.loads(resp_json); sc = r.get("score")
            d = per[model_id]; d["n"] += 1; d["ok"] += (sc is not None)
            if sc is not None:
                d["acc_n"] += 1; d["acc_ok"] += ((sc >= 0.5) == (key[item_id] == 1))
        for m, d in sorted(per.items()):
            print(f"  {m}: parse={d['ok']}/{d['n']}  acc={d['acc_ok']}/{d['acc_n']}"
                  f" ({d['acc_ok']/max(d['acc_n'],1):.2f})", flush=True)

if __name__ == "__main__":
    main()
