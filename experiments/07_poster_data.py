"""Collect every number the poster quotes into results/poster_data.json."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
load = lambda f: json.load(open(RES / f))


def main():
    hcv, nh, adlb = load("hcv_summary.json"), load("nhanes_summary.json"), load("adlb_summary.json")
    calib = load("calibration_summary.json") if (RES / "calibration_summary.json").exists() else {}
    scaling = load("scaling.json") if (RES / "scaling.json").exists() else None

    P = pd.read_csv(RES / "adlb_power.csv")
    g = P.groupby(["k", "pattern", "method"]).rate.mean().unstack("method") * 100
    power = {pat: {m: [round(g.loc[(k, pat), m], 1) for k in sorted(P.k.unique())]
                   for m in ["pipeline", "rule_clin_sig", "rule_any_oor"]}
             for pat in ["hys_law", "aki_relative", "plt_decline", "anion_gap", "fabricated"]}
    n_flag = P[P.pattern == "_n_flagged"].groupby("method").rate.mean().round(1).to_dict()
    fdr_new = P[(P.pattern == "_fdr_new") & (P.k == 1.0)].groupby("method").rate.mean().round(3).to_dict()

    d = hcv["detected_by_group"]
    out = {
        "hcv": {"n": hcv["n"], "n_disease": hcv["n_disease"], "n_flagged": hcv["n_flagged"],
                "ppv": hcv["ppv_flagged"], "auroc": hcv["auroc"], "auprc": hcv["auprc"],
                "groups": [{"name": n, "found": d[k]["sum"], "size": d[k]["size"]} for n, k in [
                    ("Blood donor", "Blood Donor"), ("Suspect donor", "suspect Blood Donor"),
                    ("Hepatitis C", "Hepatitis"), ("Fibrosis", "Fibrosis"), ("Cirrhosis", "Cirrhosis")]],
                "ablation": hcv["ablation"], "ari": hcv["ari_clusters_vs_group"]},
        "adlb": {"records": adlb["records"], "subjects": adlb["subjects"], "params": adlb["params"],
                 "features": adlb["features"], "null_flagged": adlb["null_flagged"], "null_rules": adlb["null_rules"],
                 "ks": sorted(P.k.unique().tolist()), "reps": adlb["reps"], "power": power,
                 "mean_flagged": n_flag, "fdr_new_k1": fdr_new, "auroc_by_k": adlb["auroc_by_k"]},
        "nhanes": {"n": nh["n"], "records": nh["records"], "analytes": nh["analytes"], "n_flagged": nh["n_flagged"],
                   "pct_flagged": nh["pct_flagged"], "flag_top_panel": nh["flag_top_panel"],
                   "panel_enrichment": nh["panel_enrichment"], "flag_enrichment": nh["flag_enrichment"],
                   "dialysis_total": None, "runtime_s": sum(v for k, v in nh["timings_s"].items() if k != "boot_trim")},
        "calibration": calib,
        "scaling": scaling,
    }
    subj = pd.read_csv(ROOT / "data/processed/nhanes_subjects.csv")
    out["nhanes"]["dialysis_total"] = int(subj["dialysis"].sum())
    json.dump(out, open(RES / "poster_data.json", "w"), indent=2, default=float)
    print(json.dumps(out, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
