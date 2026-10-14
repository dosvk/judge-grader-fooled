"""
Build the verifiable keyed item pool from PUBLIC datasets (licenses recorded).
Each item = {item_id, domain, prompt(QUESTION+ANSWER to grade), key(1=correct,0=incorrect), source}.
Half the candidate answers are correct (key=1), half are corrupted to a plausible wrong answer (key=0),
deterministically, so judges have a real correct/incorrect discrimination to make.

Sources + licenses (for the paper's data statement):
  - GSM8K (openai/gsm8k, MIT)                         -> math word problems, numeric key
  - MBPP sanitized (google-research-datasets, CC-BY-4.0) -> code; key = does candidate match reference intent
  - SciQ (allenai/sciq, CC-BY-NC-3.0)                 -> factual QA; correct_answer + distractors as wrong

Dissociation subset (Gate-1 R1): a tagged subset where the WRONG answer is surface-plausible
(well-formatted / idiomatic) so we can later measure Anchor-1 residual delta.
"""
import json, os, random
from datasets import load_dataset

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "items_pool.jsonl")
RNG = random.Random(20260816)


def gsm8k_items(n):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    out = []
    for i in range(min(n, len(ds))):
        q = ds[i]["question"].strip()
        gold = ds[i]["answer"].split("####")[-1].strip().replace(",", "")
        correct = (i % 2 == 0)
        try:
            wrong = str(int(float(gold)) + RNG.choice([-3, -2, 2, 3, 10]))
        except Exception:
            wrong = gold + "0"
        given = gold if correct else wrong
        out.append({"item_id": f"gsm{i:04d}", "domain": "math",
                    "prompt": f"QUESTION: {q}\nANSWER: {given}",
                    "key": 1 if correct else 0, "source": "gsm8k(MIT)",
                    "dissociation": False})
    return out


def sciq_items(n):
    ds = load_dataset("allenai/sciq", split="test")
    out = []
    for i in range(min(n, len(ds))):
        q = ds[i]["question"].strip()
        gold = ds[i]["correct_answer"].strip()
        dist = ds[i]["distractor1"].strip()
        correct = (i % 2 == 0)
        given = gold if correct else dist
        # a well-formed distractor is a surface-plausible wrong answer -> dissociation candidate
        out.append({"item_id": f"sciq{i:04d}", "domain": "factual",
                    "prompt": f"QUESTION: {q}\nANSWER: {given}",
                    "key": 1 if correct else 0, "source": "sciq(CC-BY-NC-3.0)",
                    "dissociation": (not correct)})
    return out


def mbpp_items(n):
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    out = []
    for i in range(min(n, len(ds))):
        task = ds[i]["prompt"].strip()
        ref = ds[i]["code"].strip()
        test = ds[i]["test_list"][0] if ds[i]["test_list"] else ""
        correct = (i % 2 == 0)
        if correct:
            cand = ref
        else:
            # corrupt: return None / off-by-one style break, surface-plausible (dissociation)
            cand = ref.replace("return", "return None #", 1) if "return" in ref else ref + "\n    pass"
        out.append({"item_id": f"mbpp{i:04d}", "domain": "code",
                    "prompt": f"TASK: {task}\nCANDIDATE:\n{cand}\nTEST: {test}",
                    "key": 1 if correct else 0, "source": "mbpp(CC-BY-4.0)",
                    "dissociation": (not correct)})
    return out


def build(n_each=270):
    items = gsm8k_items(n_each) + sciq_items(n_each) + mbpp_items(min(n_each, 250))
    RNG.shuffle(items)
    with open(OUT, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    # summary
    from collections import Counter
    dom = Counter(x["domain"] for x in items)
    keys = Counter(x["key"] for x in items)
    print(f"wrote {len(items)} items -> {OUT}")
    print("domains:", dict(dom), "| keys(1=correct):", dict(keys),
          "| dissociation:", sum(x["dissociation"] for x in items))
    return items


if __name__ == "__main__":
    build()
