"""Fast end-to-end checks on synthetic data (about 15 s)."""
import numpy as np
import pandas as pd

import labsignal as ls


def _long(n=150, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n):
        base = {"ALT": rng.lognormal(3, .3), "AST": rng.lognormal(3.1, .25), "BILI": rng.lognormal(2, .3),
                "CREAT": rng.normal(90, 12), "BUN": rng.normal(5.5, 1)}
        for day in [-7, 14, 28, 56, 84]:
            for p, v in base.items():
                val = v * rng.normal(1, .05)
                if s < 5 and day > 0 and p in ("ALT", "AST"):
                    val *= 6  # planted liver signal
                rows.append({"USUBJID": f"S{s:03d}", "PARAMCD": p, "AVAL": val, "ADY": day,
                             "A1LO": 0.0, "A1HI": 40.0, "ABLFL": "Y" if day < 0 else ""})
    return pd.DataFrame(rows)


def test_pipeline_flags_planted_subjects():
    F = ls.subject_features(ls.standardize(_long(), ls.ColumnMap.adlb()))
    res = ls.run(F, ls.Config(seed=0, run_umap=False))
    flagged = set(res.flagged)
    assert {f"S{i:03d}" for i in range(5)} <= flagged
    assert len(flagged) < 20
    assert set(ls.explain(res)["panel"]) <= set(res.groups)
