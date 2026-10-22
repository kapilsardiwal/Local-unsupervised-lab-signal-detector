"""Panel B: planted safety / data-quality patterns in the CDISC pilot ADLB.

1. Null control: the unmodified pilot data (what does the pipeline flag on its own?).
2. Spike-in power curves: 5 subjects per pattern (25 of 254) at effect sizes
   k = 0.25 .. 1.0, repeated over random draws, compared with range-based rules.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import labsignal as ls
from labsignal.spikein import PATTERNS, spike, rule_flags

ROOT = Path(__file__).resolve().parents[1]
REPS = int(os.environ.get("REPS", 8))
KS = [0.25, 0.5, 0.75, 1.0]
PER = 5
VISITS = ["Baseline", "Week 2", "Week 4", "Week 6", "Week 8", "Week 12", "Week 16", "Week 20", "Week 24", "Week 26"]


def load_adlb() -> pd.DataFrame:
    d = pd.concat([pd.read_sas(ROOT / f"data/raw/{f}.xpt", encoding="latin1") for f in ["adlbc", "adlbh"]])
    d = d[(d["SAFFL"] == "Y") & d["AVISIT"].isin(VISITS)]
    return ls.standardize(d, ls.ColumnMap.adlb())


def main():
    long = load_adlb()
    print(f"records={len(long):,} subjects={long.subject.nunique()} params={long.param.nunique()}", flush=True)

    # 1. null control
    F0 = ls.subject_features(long)
    res0 = ls.run(F0, ls.Config(seed=42))
    expl0 = ls.explain(res0)
    natural = set(res0.flagged)
    rules0 = rule_flags(long)
    print(f"null control: {len(natural)} flagged; rules: {rules0.sum().to_dict()}", flush=True)
    print(expl0.to_string(), flush=True)
    res0.scores.join(expl0).join(rules0).to_csv(ROOT / "results/adlb_null_scores.csv")
    np.save(ROOT / "results/adlb_null_umap2d.npy", res0.embedding2d)

    # 2. power curves
    rows, aucs = [], []
    for k in KS:
        for rep in range(REPS):
            rng = np.random.default_rng(1000 + rep)
            sl, truth = spike(long, PER, rng, k=k)
            first = k == 1.0 and rep == 0
            res = ls.run(ls.subject_features(sl), ls.Config(seed=rep, run_umap=first))
            s = res.scores.join(truth).join(rule_flags(sl))
            s["pattern"] = s["pattern"].fillna("none")
            s["spiked"] = s["pattern"] != "none"
            s["natural"] = s.index.isin(natural) & ~s.spiked
            aucs.append({"k": k, "rep": rep, "auroc": roc_auc_score(s.spiked, s.z)})
            for method, col in [("pipeline", "flag"), ("rule_clin_sig", "clin_sig"), ("rule_any_oor", "any_oor")]:
                for pat, v in s.groupby("pattern")[col].mean().items():
                    rows.append({"k": k, "rep": rep, "method": method, "pattern": pat, "rate": v})
                f = s[col]
                rows += [
                    {"k": k, "rep": rep, "method": method, "pattern": "_n_flagged", "rate": int(f.sum())},
                    # flags that are neither planted nor flagged on the unmodified data
                    {"k": k, "rep": rep, "method": method, "pattern": "_fdr_new",
                     "rate": float((f & ~s.spiked & ~s.natural).sum() / max(1, f.sum()))},
                ]
            if first:
                s.join(ls.explain(res)).to_csv(ROOT / "results/adlb_k1_scores.csv")
                np.save(ROOT / "results/adlb_k1_umap2d.npy", res.embedding2d)
            print(f"k={k} rep={rep}: flagged={s.flag.sum()} planted-found={(s.flag & s.spiked).sum()} "
                  f"auroc={aucs[-1]['auroc']:.3f}", flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(ROOT / "results/adlb_power.csv", index=False)
    pd.DataFrame(aucs).to_csv(ROOT / "results/adlb_auroc.csv", index=False)
    summ = R.groupby(["k", "pattern", "method"]).rate.mean().unstack("method")
    print(summ.round(2).to_string())
    json.dump({
        "records": len(long), "subjects": int(long.subject.nunique()), "params": int(long.param.nunique()),
        "features": int(res0.X.shape[1]), "panels": {g: len(v) for g, v in res0.groups.items()},
        "null_flagged": len(natural), "null_rules": {k: int(v) for k, v in rules0.sum().items()},
        "null_explain": expl0.reset_index().to_dict("records"),
        "auroc_by_k": pd.DataFrame(aucs).groupby("k").auroc.mean().to_dict(),
        "patterns": PATTERNS, "reps": REPS, "per_pattern": PER,
        "timings_s": res0.timings,
    }, open(ROOT / "results/adlb_summary.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
