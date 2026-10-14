"""
Bedrock backend + pinned judge panel + verifiable-item pilot (STOP-1).
Drop-in for MockBackend: same .call_model(judge, item_id, prompt, rep) -> {"score",...}.
temp=0; every raw response captured verbatim by RawCapture. Runs the noise-floor PILOT:
K re-queries per (judge,item) on keyed items => intra-judge jitter floor vs cross-judge signal.
"""
import os, re, json, time
import boto3
from judge_harness import Judge, RawCapture, run_panel

REGION = "us-east-1"

# PINNED PANEL (probed invokable 2026-08-16). 3 lineages.
PANEL = [
    Judge("anthropic", "us.anthropic.claude-opus-4-5-20251101-v1:0", REGION),
    Judge("anthropic", "us.anthropic.claude-sonnet-4-5-20250929-v1:0", REGION),
    Judge("anthropic", "us.anthropic.claude-haiku-4-5-20251001-v1:0", REGION),
    Judge("amazon",    "amazon.nova-pro-v1:0", REGION),
    Judge("amazon",    "amazon.nova-lite-v1:0", REGION),
    Judge("qwen",      "qwen.qwen3-32b-v1:0", REGION),   # Apache-2.0
]

JUDGE_INSTRUCTION = (
    "You are a strict grader. Given a QUESTION and a candidate ANSWER, decide if the answer is correct "
    "and rate quality 0..1. Reply with ONLY a compact JSON object: "
    '{"correct": true|false, "score": <0..1 float>}. No other text.'
)


def _extract(text):
    m = re.search(r'\{[^{}]*"score"[^{}]*\}', text, re.S)
    if not m:
        m = re.search(r'\{.*\}', text, re.S)
    try:
        obj = json.loads(m.group(0)) if m else {}
    except Exception:
        obj = {}
    sc = obj.get("score")
    if sc is None:
        sc = 1.0 if obj.get("correct") is True else (0.0 if obj.get("correct") is False else None)
    try:
        sc = float(sc)
    except Exception:
        sc = None
    return sc, obj


class BedrockBackend:
    def __init__(self, region=REGION):
        self.rt = boto3.client("bedrock-runtime", region_name=region)

    def call_model(self, judge: Judge, item_id: str, prompt: str, rep: int):
        body = {"messages": [{"role": "user", "content": [{"text": JUDGE_INSTRUCTION + "\n\n" + prompt}]}],
                "inferenceConfig": {"maxTokens": 120, "temperature": 0}}
        for attempt in range(5):
            try:
                r = self.rt.converse(modelId=judge.model_id, **body)
                txt = r["output"]["message"]["content"][0]["text"]
                sc, obj = _extract(txt)
                return {"score": sc, "raw_text": txt, "parsed": obj,
                        "model_id": judge.model_id, "rep": rep, "usage": r.get("usage", {})}
            except Exception as e:
                if attempt == 4:
                    return {"score": None, "error": str(e)[:200], "model_id": judge.model_id, "rep": rep}
                time.sleep(1.5 * (attempt + 1))


# Tiny PUBLIC keyed item pool for the pilot (arithmetic + short factual). Deterministic, no external deps.
def pilot_items(n=30):
    items = []
    for i in range(n):
        a, b = 12 + i, 7 + (i % 9)
        correct = (i % 3 != 0)                       # 2/3 correct, 1/3 wrong on purpose
        given = a * b if correct else a * b + 3
        items.append({"item_id": f"math{i:03d}", "key": 1 if correct else 0,
                      "prompt": f"QUESTION: What is {a} * {b}?\nANSWER: {given}"})
    return items


def noise_floor(scores):
    import numpy as np
    intra, cross = [], []
    for it, per in scores.items():
        reps_var = [np.var(v, ddof=1) for v in per.values() if len(v) > 1 and None not in v]
        if reps_var:
            intra.append(np.mean(reps_var))                       # within-judge jitter (K reps)
        means = [np.mean(v) for v in per.values() if None not in v]
        if len(means) > 1:
            cross.append(np.var(means, ddof=1))                   # across-judge spread
    return (float(np.mean(intra)) if intra else float("nan"),
            float(np.mean(cross)) if cross else float("nan"))


if __name__ == "__main__":
    items = pilot_items(30)
    cap = RawCapture(path=os.path.join(os.path.dirname(__file__), "raw_capture_pilot.sqlite"))
    be = BedrockBackend()
    t0 = time.time()
    scores = run_panel(items, PANEL, be, cap, K=3)
    dt = time.time() - t0
    intra, cross = noise_floor(scores)
    # coverage: fraction of calls that parsed a score
    total = ok = 0
    for it, per in scores.items():
        for m, reps in per.items():
            for v in reps:
                total += 1; ok += (v is not None)
    print(json.dumps({
        "calls": total, "parsed_ok": ok, "parse_rate": round(ok / total, 3),
        "captured_rows": cap.count(), "wall_sec": round(dt, 1),
        "intra_judge_var_floor": round(intra, 5), "cross_judge_var_signal": round(cross, 5),
        "signal_over_floor": round((cross / intra), 2) if intra and intra == intra and intra > 0 else None,
    }, indent=2))
    print("PILOT DONE — see raw_capture_pilot.sqlite for verbatim responses.")
