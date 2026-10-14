# Paper-2 Oracle — reproducibility package

**Paper:** "How Much LLM-Judge Error Is Common-Mode? Identifying an Internally-Non-Identifiable
Quantity via External Anchoring, Across Model Families."
**What this dir is:** the *offline, deterministic* proof-of-concept + go/no-go gate for the paper's
estimator. It contains **no cloud calls and no human data** — it validates the estimator's math and
maps its failure modes so the real experiment (Phase 1+) is designed correctly.

## One-command reproduction
```bash
./reproduce.sh
```
Creates a pinned venv (numpy==2.5.2), runs oracle v2 + v3, and checks goldens with numeric
tolerance. Exits non-zero on any mismatch. Expected runtime: < 30 s on a laptop.
`OMP_NUM_THREADS=1` is set to stabilize float-reduction order across BLAS builds.

## Files
| File | Purpose |
|---|---|
| `common_mode_oracle_v2.py` | Clean-anchor recovery, confound bias-law, interval estimator |
| `common_mode_oracle_v3.py` | Gate-0 changes: misspecification arms (C6), interval-failure at high rho (C3), sigma_eps misspec (C4), **multi-anchor rho-identification (C5)** |
| `check_goldens.py` | Tolerance-based golden comparison (NOT text hashing — see below) |
| `requirements.txt` | Pinned deps |
| `reproduce.sh` | One-command runner |
| `figs/oracle_v3_results.json` / `.npz` | Raw outputs behind every table (regenerate figures offline) |

## Environment (frozen 2026-08-16)
- Python 3.14.6, numpy==2.5.2, seeds [11,22,33,44,55], N=4000, J=6.
- Provenance (python/numpy/git) is embedded in `figs/oracle_v3_results.json`.

## What the oracle shows (and does NOT show)
- **Shows:** the estimator is algebraically unbiased **under its model** (clean-anchor recovery to
  <0.005); the human-anchor confound follows a lawful `(1-rho)^2` shrink; a fixed `[0,0.7]` interval
  **fails** when true rho>0.7 (coverage → 0); wrong sigma_eps degrades coverage; and **two heterogeneous
  anchors identify rho** via differential shrinkage (recovery to ~0.005).
- **Does NOT show:** that real LLM judge errors follow this model. Under misspecification (ordinal,
  heteroscedastic, multiplicative-style), phi bias is +0.06 / -0.115 / +0.06 respectively — so the
  paper must scope claims to low-bias score types or model these effects. **Clean-anchor recovery is a
  code-correctness check, NOT real-world validation** (per Gate-0 change C1).

## Why goldens use tolerance, not hashing
BLAS backend, thread count, numpy version, and float formatting all cause last-digit drift that would
break a text/hash comparison while the science is unchanged. `check_goldens.py` compares numerically
(`np.allclose`, atol=0.02) to catch real regressions without false alarms.

## Provenance / integrity
This is clean-provenance work: conceived for this paper, run on a personal machine, public/synthetic
data only, no employer systems or data involved.
