"""
Oracle v4 — Gate-1 R2 blocking test. Deterministic, offline.
Question: the multi-anchor rho-identification recovers Anchor-2's rho RELATIVE to Anchor-1.
If Anchor-1 is not perfectly clean (residual delta = rho(Anchor-1) != 0), how much does rho_hat(A2)
break? Cross that with the misspecification modes (ordinal/hetero/multiplicative) that already bias phi.

Pre-committed PASS criterion (frozen before running):
  For delta up to 0.05 (a plausible residual for a keyed anchor), rho_hat(A2) bias must stay < 0.10
  for the gaussian and ordinal arms. If heteroscedastic/multiplicative blow past that, we SCOPE the
  identified-rho claim to the well-behaved score types and treat others as descriptive-only.
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
SEEDS = [11, 22, 33, 44, 55]
N, J = 4000, 6
TRUE_PHI = 0.5
TRUE_RHO2 = 0.6          # the Anchor-2 rho we are trying to recover
EPS = 0.2

def make(seed, arm):
    rng = np.random.default_rng(seed)
    sc, so = np.sqrt(TRUE_PHI), np.sqrt(1 - TRUE_PHI)
    t = rng.normal(0, 1, N)
    if arm == "multiplicative":
        style = rng.normal(0, 1, N); c = rng.normal(0, sc, N) + 0.4 * style
    else:
        c = rng.normal(0, sc, N)
    if arm == "heteroscedastic":
        diff = np.abs(rng.normal(0, 1, N)); diff /= diff.mean()
        o = rng.normal(0, 1, (N, J)) * (so * diff[:, None])
    else:
        o = rng.normal(0, so, (N, J))
    s = t[:, None] + c[:, None] + o
    if arm == "ordinal":
        def ordz(x):
            z = (x - x.mean()) / (x.std() + 1e-9)
            return np.clip(np.round(z * 1.25 + 3), 1, 5)
        s = ordz(s)
    return t, c, s, rng

def recover_rho2(seed, arm, delta):
    t, c, s, rng = make(seed, arm)
    a1 = t + delta * c + rng.normal(0, EPS, N)      # Anchor-1: residual rho = delta (NOT clean)
    a2 = t + TRUE_RHO2 * c + rng.normal(0, EPS, N)   # Anchor-2: true rho = 0.6
    hVo = s.var(axis=1, ddof=1).mean()
    rs1 = max((s.mean(axis=1) - a1).var(ddof=1) - hVo / J, 0.0)   # ~ (1-delta)^2 sigc2 + eps^2
    rs2 = max((s.mean(axis=1) - a2).var(ddof=1) - hVo / J, 0.0)   # ~ (1-rho2)^2 sigc2 + eps^2
    # NAIVE (assume Anchor-1 clean): sigc2_hat = rs1 - eps^2  -> biased if delta!=0
    sigc2 = max(rs1 - EPS ** 2, 1e-9)
    ratio = max(rs2 - EPS ** 2, 0.0) / sigc2
    return 1 - np.sqrt(max(ratio, 0.0))

def main():
    print("=" * 84)
    print("ORACLE v4 — R2: rho_hat(Anchor-2) robustness to Anchor-1 residual delta (true rho2=0.60)")
    print("=" * 84)
    results = {}
    passed = True
    for arm in ["gaussian", "ordinal", "heteroscedastic", "multiplicative"]:
        row = []
        print(f"\n  arm = {arm}")
        print(f"    {'delta(A1)':>9} | {'rho2_hat':>8} | {'abs err':>7}")
        for delta in [0.0, 0.02, 0.05, 0.10]:
            rh = float(np.mean([recover_rho2(s, arm, delta) for s in SEEDS]))
            err = abs(rh - TRUE_RHO2)
            row.append((delta, rh, err))
            print(f"    {delta:>9.2f} | {rh:>8.3f} | {err:>7.3f}")
        results[arm] = row
        # pass check at delta<=0.05 for gaussian+ordinal
        if arm in ("gaussian", "ordinal"):
            worst = max(e for d, _, e in row if d <= 0.05)
            ok = worst < 0.10
            passed &= ok
            print(f"    -> delta<=0.05 worst err = {worst:.3f}  [{'PASS' if ok else 'FAIL'}]")

    json.dump(results, open(os.path.join(FIG, "oracle_v4_results.json"), "w"), indent=2, default=str)
    print("\n" + "=" * 84)
    print("R2 PRE-COMMITTED CRITERION (gaussian+ordinal, delta<=0.05, err<0.10):",
          "PASS" if passed else "FAIL")
    print("Interpretation: identified-rho claim is valid where residual Anchor-1 delta is small AND")
    print("scores are well-behaved; heteroscedastic/multiplicative arms show how far it degrades =>")
    print("scope the identified-rho deliverable to those regimes; else report descriptive-only.")
    print("Wrote figs/oracle_v4_results.json")
    return passed

if __name__ == "__main__":
    main()
