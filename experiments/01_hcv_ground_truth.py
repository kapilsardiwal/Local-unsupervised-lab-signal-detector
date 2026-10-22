"""Panel A: recover hidden liver disease in UCI HCV data (labels never seen by the model).

Hoffmann G et al. (2018), UCI ML Repository id 571, CC BY 4.0.
615 people: 533 blood donors, 7 'suspect' donors, 24 hepatitis C, 21 fibrosis, 30 cirrhosis.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, adjusted_rand_score

import labsignal as ls

ROOT = Path(__file__).resolve().parents[1]
LABS = ["ALB", "ALP", "ALT", "AST", "BIL", "CHE", "CHOL", "CREA", "GGT", "PROT"]


def load():
    raw = pd.read_csv(ROOT / "data/raw/hcvdat0.csv", index_col=0)
    raw.index = raw.index.astype(str)
    raw["group"] = raw["Category"].str.split("=").str[1]
    long = raw.reset_index().melt(id_vars=["index"], value_vars=LABS, var_name="param", value_name="value")
    return raw, ls.standardize(long, ls.ColumnMap.generic(subject="index"))


def evaluate(res, raw):
    s = res.scores.join(raw[["group"]])
    s["disease"] = s["group"] != "Blood Donor"
    k = int(s.disease.sum())
    return s, {
        "n_flagged": int(s.flag.sum()),
        "ppv_flagged": float(s.loc[s.flag, "disease"].mean()) if s.flag.any() else None,
        "auroc": roc_auc_score(s.disease, s.z),
        "auprc": average_precision_score(s.disease, s.z),
        "precision_at_k": float(s.sort_values("z", ascending=False).head(k).disease.mean()),
        "detected_by_group": s.groupby("group").flag.agg(["sum", "size"]).astype(int).to_dict("index"),
    }


def main():
    raw, long = load()
    F = ls.subject_features(long)
    res = ls.run(F, ls.Config(seed=42))
    s, out = evaluate(res, raw)
    out.update({
        "n": len(s), "n_disease": int(s.disease.sum()),
        "auroc_glosh": roc_auc_score(s.disease, s.glosh),
        "ari_clusters_vs_group": adjusted_rand_score(s.group, s.cluster),
        "n_clusters": int(s.cluster.max() + 1),
        "cluster_by_group": pd.crosstab(s.group, s.cluster).to_dict("index"),
        "timings_s": res.timings,
    })

    # ablation: original IF+LOF on all features vs the panel ensemble
    ab = {}
    for name, cfg in {
        "IF+LOF, all features": ls.Config(seed=42, run_umap=False, detectors=("if", "lof"), panels={}),
        "IF+LOF+PCA, all features": ls.Config(seed=42, run_umap=False, panels={}),
        "IF+LOF+PCA, organ panels": ls.Config(seed=42, run_umap=False),
    }.items():
        _, m = evaluate(ls.run(F, cfg), raw)
        ab[name] = {k: m[k] for k in ["n_flagged", "ppv_flagged", "auroc", "auprc"]}
    out["ablation"] = ab

    print(json.dumps(out, indent=2, default=float))
    expl = ls.explain(res)
    print(expl.join(s[["group"]]).to_string())
    s.join(expl).to_csv(ROOT / "results/hcv_scores.csv")
    np.save(ROOT / "results/hcv_umap2d.npy", res.embedding2d)
    json.dump(out, open(ROOT / "results/hcv_summary.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
