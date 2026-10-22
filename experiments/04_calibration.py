"""Calibration: how often does the pipeline flag anyone when there is nothing to find?

Synthetic null datasets match each real dataset's size, marginals and
correlation (Gaussian copula of the real features) but contain no planted or
natural anomalies beyond that structure.
"""
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import labsignal as ls
from labsignal.detect import preprocess, copula_reference

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
SIMS = int(os.environ.get("SIMS", 20))


def shapes():
    hcv = importlib.import_module("01_hcv_ground_truth")
    adlb = importlib.import_module("02_adlb_spikein")
    yield "HCV-shaped (615 x 10)", ls.subject_features(hcv.load()[1])
    yield "ADLB-shaped (254 x 177)", ls.subject_features(adlb.load_adlb())
    long = pd.read_csv(ROOT / "data/processed/nhanes_long.csv.gz", dtype={"USUBJID": str})
    long = ls.standardize(long, ls.ColumnMap.generic(subject="USUBJID", param="PARAMCD", value="AVAL"))
    F = ls.subject_features(long)
    yield "NHANES-shaped (2000 x 43)", F.sample(2000, random_state=0)


def main():
    out = {}
    for name, F in shapes():
        X = preprocess(F, 25)
        rows = []
        for i in range(SIMS):
            syn = pd.DataFrame(copula_reference(X.to_numpy(), len(X), np.random.default_rng(500 + i)), columns=X.columns)
            r = ls.run(syn, ls.Config(seed=i, run_umap=False))
            p = r.scores["p"]
            rows.append({"n_flags": int(r.scores["flag"].sum()), "p<0.01": float((p < 0.01).mean()),
                         "p<0.05": float((p < 0.05).mean())})
            print(name, i, rows[-1], flush=True)
        d = pd.DataFrame(rows)
        out[name] = {"sims": SIMS, "n": len(X), "features": X.shape[1],
                     "mean_false_flags": d.n_flags.mean(), "max_false_flags": int(d.n_flags.max()),
                     "share_sims_any_flag": float((d.n_flags > 0).mean()),
                     "false_flag_rate_pct": float(d.n_flags.mean() / len(X) * 100),
                     "p<0.01_rate": d["p<0.01"].mean(), "p<0.05_rate": d["p<0.05"].mean()}
        print(json.dumps(out[name], indent=2), flush=True)
    json.dump(out, open(ROOT / "results/calibration_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
