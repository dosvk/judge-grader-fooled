"""
Golden check for the Paper-2 oracle (Gate-0 REPRO requirement).
Compares oracle_v3 outputs to frozen expected values using NUMERIC TOLERANCE (np.allclose),
NOT text hashing — because BLAS/float-reduction/format drift causes false failures on hashes.
Exits non-zero on mismatch so reproduce.sh fails loudly.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "figs", "oracle_v3_results.json")

# Frozen expected values (from the reviewed 2026-08-16 run, seeds 11-55, N=4000, J=6, numpy 2.5.2).
GOLDEN = {
    "C6_misspecification": {          # name -> expected est phi (planted 0.5)
        "gaussian(base)": 0.504, "ordinal5": 0.560,
        "heteroscedastic": 0.385, "multiplicative_style": 0.563,
    },
    "C3_interval_failure": {          # true_rho -> (phi_lo, phi_hi, coverage)
        0.5: (0.207, 0.744, 1.0), 0.7: (0.090, 0.524, 1.0),
        0.8: (0.046, 0.349, 0.0), 0.9: (0.017, 0.162, 0.0), 1.0: (0.007, 0.070, 0.0),
    },
    "C4_eps_misspec": {0.5:1.0, 0.75:1.0, 1.0:1.0, 1.5:1.0, 2.0:0.0},
    "C5_multianchor_rho_id": {0.3: 0.299, 0.6: 0.598, 0.9: 0.895},  # true_rho2 -> rho2_hat
}
ATOL = 0.02   # tolerance absorbs cross-BLAS/version last-digit drift while catching real regressions

def fail(msg):
    print("GOLDEN MISMATCH:", msg); sys.exit(1)

def main():
    if not os.path.exists(RES):
        fail(f"missing {RES}; run common_mode_oracle_v3.py first")
    r = json.load(open(RES))

    for name, est, _ in r["C6_misspecification"]:
        exp = GOLDEN["C6_misspecification"][name]
        if not np.isclose(float(est), exp, atol=ATOL):
            fail(f"C6 {name}: got {est}, expected {exp}")

    for row in r["C3_interval_failure"]:
        rho, lo, hi, cov = row
        elo, ehi, ecov = GOLDEN["C3_interval_failure"][float(rho)]
        if not (np.isclose(lo, elo, atol=ATOL) and np.isclose(hi, ehi, atol=ATOL) and np.isclose(cov, ecov, atol=1e-6)):
            fail(f"C3 rho={rho}: got {(lo,hi,cov)}, expected {(elo,ehi,ecov)}")

    for ratio, cov in r["C4_eps_misspec"]:
        if not np.isclose(cov, GOLDEN["C4_eps_misspec"][float(ratio)], atol=1e-6):
            fail(f"C4 ratio={ratio}: coverage {cov} != {GOLDEN['C4_eps_misspec'][float(ratio)]}")

    for rho2, rhat, _ in r["C5_multianchor_rho_id"]:
        if not np.isclose(float(rhat), GOLDEN["C5_multianchor_rho_id"][float(rho2)], atol=ATOL):
            fail(f"C5 rho2={rho2}: rho2_hat {rhat} != {GOLDEN['C5_multianchor_rho_id'][float(rho2)]}")

    print(f"GOLDENS OK (atol={ATOL}) — provenance: {r.get('provenance')}")

if __name__ == "__main__":
    main()
