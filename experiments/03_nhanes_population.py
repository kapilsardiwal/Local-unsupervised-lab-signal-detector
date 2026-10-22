"""Panel C: real population at scale (NHANES 2017-March 2020, 8,253 adults, 43 analytes).

Self-reported diagnoses and pregnancy are never shown to the model; they are
used afterwards to ask whether flags, and the organ panel each flag points to,
line up with known conditions.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

import labsignal as ls

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ["diabetes", "insulin", "kidney_failure", "dialysis", "liver_condition", "cirrhosis",
              "viral_hepatitis", "heart_failure", "cancer", "pregnant"]
PANEL_CONDITIONS = {"renal": ["kidney_failure", "dialysis"], "liver": ["liver_condition", "cirrhosis", "viral_hepatitis"],
                    "metabolic": ["diabetes", "insulin"]}


def enrichment(mask, s, cond):
    a, b = int((mask & s[cond]).sum()), int((mask & ~s[cond]).sum())
    c, d = int((~mask & s[cond]).sum()), int((~mask & ~s[cond]).sum())
    odds, p = fisher_exact([[a, b], [c, d]])
    return {"cases_in_group": a, "group_n": a + b, "prev_group": a / max(1, a + b), "prev_rest": c / max(1, c + d),
            "odds_ratio": odds, "p": p}


def main():
    long = pd.read_csv(ROOT / "data/processed/nhanes_long.csv.gz", dtype={"USUBJID": str})
    subj = pd.read_csv(ROOT / "data/processed/nhanes_subjects.csv", dtype={"USUBJID": str}).set_index("USUBJID")
    long = ls.standardize(long, ls.ColumnMap.generic(subject="USUBJID", param="PARAMCD", value="AVAL"))
    F = ls.subject_features(long)
    res = ls.run(F, ls.Config(seed=42))
    s = res.scores.join(subj)
    for c in CONDITIONS:
        s[c] = s[c].fillna(False).astype(bool)
    expl = ls.explain(res)
    s = s.join(expl[["drivers", "labs"]])

    out = {"n": len(s), "records": len(long), "analytes": int(long.param.nunique()), "features": F.shape[1],
           "panels": {g: len(v) for g, v in res.groups.items()},
           "n_flagged": int(s.flag.sum()), "pct_flagged": float(s.flag.mean() * 100),
           "flag_top_panel": s.loc[s.flag, "top_panel"].value_counts().to_dict(),
           "n_clusters": int(s.cluster.max() + 1), "noise": int((s.cluster == -1).sum()),
           "timings_s": res.timings}
    out["flag_enrichment"] = {c: enrichment(s.flag, s, c) for c in CONDITIONS}
    out["panel_enrichment"] = {
        f"{panel}->{c}": enrichment(s.flag & (s.top_panel == panel), s, c)
        for panel, conds in PANEL_CONDITIONS.items() for c in conds}
    cl = s.groupby("cluster").agg(n=("z", "size"), female=("sex", lambda x: (x == "F").mean()),
                                  age=("age", "median"), **{c: (c, "mean") for c in CONDITIONS})
    out["clusters"] = cl.round(3).reset_index().to_dict("records")

    print(json.dumps({k: v for k, v in out.items() if k != "clusters"}, indent=2, default=float))
    print(cl.round(2).to_string())
    print(s[s.flag].sort_values("z", ascending=False).head(25)[["top_panel", "drivers", "age", "sex"] +
                                                                  [c for c in CONDITIONS]].to_string())
    s.to_csv(ROOT / "results/nhanes_scores.csv")
    np.save(ROOT / "results/nhanes_umap2d.npy", res.embedding2d)
    json.dump(out, open(ROOT / "results/nhanes_summary.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
