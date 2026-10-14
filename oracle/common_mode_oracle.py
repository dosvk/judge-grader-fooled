"""
Synthetic Oracle for Paper-2 Candidate A: common-mode vs orthogonal judge-error decomposition.

GO/NO-GO GATE. Pure, deterministic, offline. No Bedrock, no humans.

Question this answers
---------------------
If we PLANT a known decomposition of judge error into
  - common-mode  (shared across all judges; the part cross-judge disagreement is BLIND to), and
  - orthogonal   (idiosyncratic per judge; what disagreement CAN see),
can an estimator that only sees judge scores + an EXTERNAL anchor (human labels) recover the
planted common-mode fraction? And critically: how much does HUMAN-LABEL NOISE that is itself
common-mode (and possibly correlated with the judges' shared bias) corrupt the estimate?

If the estimator recovers planted values when the human anchor is clean, AND we can characterize
the bias as a function of anchor-noise / anchor-judge correlation, then Candidate A is viable
(we report a bounded interval, not a point). If recovery is hopeless even with a clean anchor,
Candidate A is dead -> fall back to Candidate B.

Model
-----
For item i (i=1..N), latent truth t_i ~ N(0,1).
Judge j (j=1..J) score:  s_ij = t_i + c_i + o_ij
  c_i  ~ N(0, sigma_c^2)        common-mode error: SAME for all judges on item i (shared blind spot)
  o_ij ~ N(0, sigma_o^2)        orthogonal error: independent per (i,j)
Planted common-mode fraction of judge-error variance:
  phi = sigma_c^2 / (sigma_c^2 + sigma_o^2)

External human anchor:  a_i = t_i + h_i
  h_i = rho * c_i + eps_i       human error partly SHARES the judges' common-mode blind spot
                                 rho in [0,1] controls anchor-judge correlation
  eps_i ~ N(0, sigma_eps^2)     independent human noise

Estimators (using only {s_ij} and {a_i}; t_i is hidden)
-------------------------------------------------------
Judge error relative to anchor:  e_ij = s_ij - a_i
  Var over j (within item) isolates ORTHOGONAL var (anchor + common-mode cancel out across judges):
      hatVar_o = mean_i [ sample_var_j(s_ij) ]                      # anchor-free, unbiased for sigma_o^2
  Total dispersion of judge mean around anchor carries common-mode + anchor error:
      m_i = mean_j s_ij = t_i + c_i + obar_i
      d_i = m_i - a_i = c_i - h_i + obar_i = (1-rho)c_i - eps_i + obar_i
      Var_i(d_i) = (1-rho)^2 sigma_c^2 + sigma_eps^2 + sigma_o^2/J
  So estimated common-mode (naive, ignoring anchor error):
      hatVar_c_naive = Var(d) - hatVar_o / J
                     = (1-rho)^2 sigma_c^2 + sigma_eps^2          # BIASED: (1-rho)^2 shrink + eps inflation
  Estimated common-mode fraction:
      hatPhi = hatVar_c / (hatVar_c + hatVar_o)

The oracle sweeps (phi, rho, sigma_eps) and reports recovery error, so we learn the bias law and
whether a CLEAN anchor (rho=0, eps small) recovers phi. That bias law IS the paper's contribution.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class Planted:
    phi: float          # planted common-mode fraction
    rho: float          # anchor-judge common-mode correlation (0 = clean anchor)
    sigma_eps: float    # independent human-label noise sd
    J: int = 6          # judges
    N: int = 4000       # items
    err_scale: float = 1.0  # total judge-error sd (sigma_c^2 + sigma_o^2 = err_scale^2)


def simulate(p: Planted, seed: int):
    rng = np.random.default_rng(seed)
    sigma_c = np.sqrt(p.phi) * p.err_scale
    sigma_o = np.sqrt(1.0 - p.phi) * p.err_scale
    t = rng.normal(0, 1, p.N)
    c = rng.normal(0, sigma_c, p.N)                     # common-mode, per item
    o = rng.normal(0, sigma_o, (p.N, p.J))              # orthogonal, per item x judge
    s = t[:, None] + c[:, None] + o                     # judge scores
    eps = rng.normal(0, p.sigma_eps, p.N)
    h = p.rho * c + eps                                  # human error shares rho of common-mode
    a = t + h                                            # human anchor
    return s, a, sigma_c, sigma_o


def estimate(s, a):
    N, J = s.shape
    # orthogonal variance: within-item sample variance across judges (anchor-free)
    within = s.var(axis=1, ddof=1)              # per item
    hatVar_o = within.mean()
    # common-mode: dispersion of judge-mean around anchor, minus mean-of-orthogonal shrink
    m = s.mean(axis=1)
    d = m - a
    hatVar_c = d.var(ddof=1) - hatVar_o / J
    hatVar_c = max(hatVar_c, 0.0)               # clamp (variance >= 0)
    hatPhi = hatVar_c / (hatVar_c + hatVar_o) if (hatVar_c + hatVar_o) > 0 else float("nan")
    return hatPhi, hatVar_c, hatVar_o


def run_grid():
    print("=" * 92)
    print("SYNTHETIC ORACLE — common-mode/orthogonal recovery (mean over 5 seeds; N=4000, J=6)")
    print("=" * 92)
    print(f"{'planted phi':>11} | {'rho':>4} | {'sig_eps':>7} | {'est phi':>8} | {'abs err':>8} | verdict")
    print("-" * 92)
    seeds = [11, 22, 33, 44, 55]
    rows = []
    for phi in [0.1, 0.3, 0.5, 0.7]:
        for rho in [0.0, 0.5, 1.0]:
            for sig_eps in [0.0, 0.3]:
                p = Planted(phi=phi, rho=rho, sigma_eps=sig_eps)
                ests = []
                for sd in seeds:
                    s, a, _, _ = simulate(p, sd)
                    hatPhi, _, _ = estimate(s, a)
                    ests.append(hatPhi)
                est = float(np.mean(ests))
                err = abs(est - phi)
                clean = (rho == 0.0 and sig_eps == 0.0)
                verdict = ("RECOVERED" if err < 0.03 else "biased") if clean else \
                          ("(anchor-confounded)" if err >= 0.03 else "ok-despite-noise")
                rows.append((phi, rho, sig_eps, est, err, clean))
                print(f"{phi:>11.2f} | {rho:>4.1f} | {sig_eps:>7.2f} | {est:>8.3f} | {err:>8.3f} | {verdict}")
    print("-" * 92)

    # GO/NO-GO logic
    clean_rows = [r for r in rows if r[5]]
    clean_ok = all(r[4] < 0.03 for r in clean_rows)
    # is the confound direction as theory predicts? rho up => estimate down (shrink by (1-rho)^2)
    monotone = True
    for phi in [0.1, 0.3, 0.5, 0.7]:
        seq = [r[3] for r in rows if r[0] == phi and r[2] == 0.0]  # sig_eps=0, rho=0,.5,1
        monotone &= (seq[0] >= seq[1] >= seq[2] - 1e-9)
    print("\nGO/NO-GO EVALUATION")
    print(f"  (1) Clean anchor (rho=0, eps=0) recovers planted phi within 0.03:  {'PASS' if clean_ok else 'FAIL'}")
    print(f"  (2) Bias is lawful/characterizable (rho^ => est phi v, monotone):   {'PASS' if monotone else 'FAIL'}")
    verdict = clean_ok and monotone
    print("\n  ==> ORACLE VERDICT:",
          "GO — estimator is sound with a clean anchor; confound is a characterizable bias law\n"
          "      (report common-mode as an interval bounded by anchor-noise). Candidate A viable."
          if verdict else
          "  NO-GO — estimator cannot recover planted values even with a clean anchor. Fall back to B.")
    return verdict


if __name__ == "__main__":
    run_grid()
