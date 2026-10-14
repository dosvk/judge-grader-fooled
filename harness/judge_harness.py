"""
Judge harness + raw_capture layer (Gate-1 REPRO #2 — MUST exist before any spend).
Offline-testable via a deterministic MOCK backend so the whole pipeline is validated with ZERO
Bedrock cost. Swapping in the real Bedrock backend later changes only `call_model`.

Design invariants (from Gate-0/1 reviews):
  - Every request payload + raw response + model ID + region + timestamp stored VERBATIM, immutably.
    (Live API responses are unrecoverable; the cache IS the artifact of record.)
  - Deterministic offline via seeds; real API is nondeterministic => the K-re-query noise floor is the
    reproducibility substitute (documented).
  - No employer systems/data. Public/synthetic items only.
"""
import os, json, time, hashlib, sqlite3
from dataclasses import dataclass, asdict
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DB = os.path.join(HERE, "raw_capture.sqlite")


# ------------------------- raw capture (immutable append-only) -------------------------
class RawCapture:
    """Append-only store of every model interaction. Never overwrites; keyed by content hash."""
    def __init__(self, path=CACHE_DB):
        self.db = sqlite3.connect(path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS calls(
            call_hash TEXT PRIMARY KEY, ts REAL, lineage TEXT, model_id TEXT, region TEXT,
            item_id TEXT, rep INT, request_json TEXT, response_json TEXT)""")
        self.db.commit()

    def key(self, model_id, item_id, rep, request_json):
        return hashlib.sha256(f"{model_id}|{item_id}|{rep}|{request_json}".encode()).hexdigest()[:16]

    def get(self, model_id, item_id, rep, request_json):
        k = self.key(model_id, item_id, rep, request_json)
        r = self.db.execute("SELECT response_json FROM calls WHERE call_hash=?", (k,)).fetchone()
        return json.loads(r[0]) if r else None

    def put(self, lineage, model_id, region, item_id, rep, request_json, response_json):
        k = self.key(model_id, item_id, rep, request_json)
        self.db.execute("INSERT OR IGNORE INTO calls VALUES (?,?,?,?,?,?,?,?,?)",
                        (k, time.time(), lineage, model_id, region, item_id, rep,
                         request_json, json.dumps(response_json)))
        self.db.commit()

    def count(self):
        return self.db.execute("SELECT COUNT(*) FROM calls").fetchone()[0]


# ------------------------- backends -------------------------
@dataclass
class Judge:
    lineage: str          # "anthropic" | "amazon" | "openweights"
    model_id: str
    region: str = "us-east-1"


class MockBackend:
    """Deterministic offline judge. Simulates a keyed-item verdict with a planted common-mode +
    orthogonal + a lineage-shared style bias, so the FULL analysis pipeline can be exercised with
    no cost and known ground truth. Replaced by BedrockBackend at spend time (same interface)."""
    def __init__(self, seed=7):
        import numpy as np
        self.np = np
        self.rng = np.random.default_rng(seed)
        # planted, per-item latent structure created lazily & cached
        self._c = {}

    def _common(self, item_id):
        if item_id not in self._c:
            h = int(hashlib.sha256(item_id.encode()).hexdigest(), 16) % (2**32)
            self._c[item_id] = self.np.random.default_rng(h).normal(0, 0.7)
        return self._c[item_id]

    def call_model(self, judge: Judge, item_id: str, prompt: str, rep: int):
        # verdict score in [0,1]; shared common-mode + lineage style + orthogonal noise
        c = self._common(item_id)
        lineage_bias = {"anthropic": 0.05, "amazon": 0.04, "openweights": 0.06}[judge.lineage]
        rng = self.np.random.default_rng(int(hashlib.sha256(f"{judge.model_id}{item_id}{rep}".encode()).hexdigest(), 16) % (2**32))
        raw = 0.5 + 0.3 * c + lineage_bias + rng.normal(0, 0.15)
        score = float(min(max(raw, 0), 1))
        return {"score": score, "rationale": "mock", "model_id": judge.model_id, "rep": rep}


# BedrockBackend stub — implemented at spend time; identical .call_model signature.
class BedrockBackend:  # pragma: no cover (requires real spend)
    def __init__(self, client=None):
        self.client = client
    def call_model(self, judge, item_id, prompt, rep):
        raise NotImplementedError("Enable only after Gate-1 STOP-1 spend approval.")


# ------------------------- harness -------------------------
def run_panel(items, judges, backend, capture: RawCapture, K=3):
    """items: list of dicts {item_id, prompt, key}. Returns scores[item_id][model_id] = [reps]."""
    scores = {}
    for it in items:
        scores[it["item_id"]] = {}
        for j in judges:
            reps = []
            for rep in range(K):
                req = json.dumps({"item": it["item_id"], "prompt": it["prompt"], "rep": rep})
                cached = capture.get(j.model_id, it["item_id"], rep, req)
                if cached is None:
                    resp = backend.call_model(j, it["item_id"], it["prompt"], rep)
                    capture.put(j.lineage, j.model_id, j.region, it["item_id"], rep, req, resp)
                    cached = resp
                reps.append(cached["score"])
            scores[it["item_id"]][j.model_id] = reps
    return scores


if __name__ == "__main__":
    # OFFLINE smoke test: 20 synthetic items x 5 judges x K=3, mock backend, prove capture+rerun determinism.
    items = [{"item_id": f"itm{i:03d}", "prompt": f"q{i}", "key": i % 2} for i in range(20)]
    judges = [Judge("anthropic", "mock-claude-opus"), Judge("anthropic", "mock-claude-sonnet"),
              Judge("amazon", "mock-nova-pro"), Judge("amazon", "mock-nova-lite"),
              Judge("openweights", "mock-llama")]
    cap = RawCapture(path=os.path.join(HERE, "raw_capture_smoke.sqlite"))
    be = MockBackend()
    s1 = run_panel(items, judges, be, cap, K=3)
    n1 = cap.count()
    # rerun: must hit cache (no new rows) and identical scores
    s2 = run_panel(items, judges, be, cap, K=3)
    n2 = cap.count()
    identical = s1 == s2
    print(f"captured calls after run1={n1}, after run2={n2} (cache hit => equal: {n1==n2})")
    print(f"rerun identical scores: {identical}")
    print("SMOKE OK" if (n1 == n2 and identical and n1 == 20*5*3) else "SMOKE FAILED")
    os.remove(os.path.join(HERE, "raw_capture_smoke.sqlite"))
