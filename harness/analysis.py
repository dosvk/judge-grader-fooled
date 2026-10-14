"""
Analysis pipeline: harness scores -> common-mode decomposition across lineages + rho-identified interval.
Offline-validated against the mock harness (known planted structure). Regenerates all numbers from the
raw_capture cache with NO network (REPRO requirement). Estimand labeled per Gate-1b binding change:
"conditional common-mode over the SAMPLED lineages" (upper bound; wide df interval).
"""
import os, json
import numpy as np
from judge_harness import RawCapture, Judge, MockBackend, run_panel

HERE = os.path.dirname(os.path.abspath(__file__))


def decompose(scores, judges, eps_known=0.1, rho_lo=0.0, rho_hi=0.7):
    """scores[item][model] = [reps]. Returns phi decomposition + interval, per Gate-1b framing."""
    models = [j.model_id for j in judges]
    items = list(scores.keys())
    # mean over reps -> item x model matrix
    M = np.array([[np.mean(scores[it][m]) for m in models] for it in items])   # (I, J)
    # orthogonal var = mean within-item variance across judges
    hatVar_o = M.var(axis=1, ddof=1).mean()
    # common-mode proxy: variance of item-mean-judge around its grand behavior
    item_mean = M.mean(axis=1)
    raw_shared = max(item_mean.var(ddof=1) - hatVar_o / len(models), 0.0)
    # interval on phi correcting for assumed rho band + known eps (bounds per oracle law)
    def phi_at(r):
        vc = max((raw_shared - eps_known ** 2) / max((1 - r) ** 2, 1e-6), 0.0)
        return vc / (vc + hatVar_o) if vc + hatVar_o > 0 else float("nan")
    return {"hatVar_o": hatVar_o, "raw_shared": raw_shared,
            "phi_lo": phi_at(rho_lo), "phi_hi": phi_at(rho_hi)}


def between_lineage(scores, judges):
    """df check: between-lineage component needs >=3 lineages for a meaningful common-mode."""
    lineages = sorted(set(j.lineage for j in judges))
    per = {L: [] for L in lineages}
    items = list(scores.keys())
    for it in items:
        for j in judges:
            per[j.lineage].append(np.mean(scores[it][j.model_id]))
    lineage_means = {L: float(np.mean(v)) for L, v in per.items()}
    return {"n_lineages": len(lineages), "df_between": len(lineages) - 1,
            "lineage_means": lineage_means}


if __name__ == "__main__":
    # Validate end-to-end on the mock harness (planted common-mode present -> phi should be clearly >0).
    items = [{"item_id": f"itm{i:03d}", "prompt": f"q{i}", "key": i % 2} for i in range(300)]
    judges = [Judge("anthropic", "mock-claude-opus"), Judge("anthropic", "mock-claude-sonnet"),
              Judge("amazon", "mock-nova-pro"), Judge("amazon", "mock-nova-lite"),
              Judge("openweights", "mock-llama")]
    cap = RawCapture(path=os.path.join(HERE, "raw_capture_analysis.sqlite"))
    s = run_panel(items, judges, MockBackend(), cap, K=3)
    dec = decompose(s, judges)
    bl = between_lineage(s, judges)
    print("BETWEEN-LINEAGE:", json.dumps(bl, indent=2))
    print("DECOMPOSITION  :", json.dumps(dec, indent=2))
    # sanity: mock planted a common-mode (0.3*c shared) => raw_shared>0 and phi interval away from 0
    ok = dec["raw_shared"] > 0 and dec["phi_hi"] > dec["phi_lo"] >= 0 and bl["df_between"] == 2
    print("ANALYSIS PIPELINE OK" if ok else "ANALYSIS PIPELINE FAILED")
    os.remove(os.path.join(HERE, "raw_capture_analysis.sqlite"))
