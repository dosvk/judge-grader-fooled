# Supplementary Material — "When the Grader Is Fooled" (JUDGe 2026 camera-ready)

## Contents
- `harness/raw_capture_{full,rubric,pilot}.sqlite` — ALL 60,801 raw verdicts (40,950 + 19,851, incl. the six-family expansion + auxiliary judge;
  request+response JSON per call; append-only cache)
- `harness/items_pool.jsonl`, `items_ow_matched.jsonl` — the keyed item pools
- `harness/canonical.py` — **single source of truth: every number in the paper** -> canonical_results.json
- `harness/decompose_controlled.py` — item-demeaned decomposition (the 0.071 -> 0.0076 result)
- `harness/make_figs.py` — Figures 1-3 (requires the two scripts above to have run)
- `harness/judge_harness.py`, `analysis.py` — collection harness + estimator-pipeline validation (mock-backed)
- `harness/add_lineage56.py`, `pilot_lineage56.py` — six-family expansion collection (pre-reg Amendment 2)
- `harness/sensitivity_checks.py` — revision sensitivity analyses (censoring bounds, threshold sweep, ceiling, permutation p, Wilson CIs)
- `oracle/` — synthetic-oracle validation of the estimator (pinned venv, tolerance goldens)

## Reproduce (4 commands, no network beyond pip)
    python3 -m venv .venv && .venv/bin/pip install numpy pandas matplotlib
    .venv/bin/python harness/canonical.py            # -> harness/canonical_results.json (all paper numbers)
    .venv/bin/python harness/decompose_controlled.py && .venv/bin/python harness/make_figs.py   # -> figs 1-3
    (cd oracle && bash reproduce.sh)                 # -> "ALL ORACLE CHECKS PASSED" (own pinned venv)

Expected: canonical_results.json matches the paper's abstract/tables (raw shared 0.058 [0.051,0.066];
item-demeaned 0.0062 [0.0050,0.0079]; PW acceptance openai .014 / anthropic .016 / qwen .033 /
amazon .197 / google .252 / meta .346; strict-rubric meta .059; accuracy 0.741-0.968, mean 0.845).

No credentials or vendor APIs required; the caches are the primary data.
Released alongside the camera-ready version.
