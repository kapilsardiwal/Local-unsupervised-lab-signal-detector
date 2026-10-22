"""Plant realistic safety and data-quality patterns into ADLB-like long data.

Each injector edits post-baseline records of one subject in place. Patterns are
chosen to span what range-based rules can and cannot see.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PATTERNS = {
    "hys_law": "ALT and AST rise to 3-6x ULN with bilirubin 2-3x ULN (positive control)",
    "aki_relative": "creatinine rises to 1.5-2.0x baseline, BUN follows (KDIGO stage 1)",
    "plt_decline": "platelets fall 35-50% from baseline but stay above LLN",
    "anion_gap": "sodium drifts to high-normal, chloride to low-normal; every value in range",
    "fabricated": "post-baseline values copied from baseline (zero within-subject variation)",
}
# Effect size k scales each pattern from absent (0) to its clinical definition (1).


def _ramp(d: pd.DataFrame, rng) -> np.ndarray:
    """0 before a random onset, rising to 1 at the subject's last visit."""
    t = d["time"].to_numpy(float)
    tmax = t.max()
    onset = rng.uniform(0.1, 0.4) * tmax
    return np.clip((t - onset) / max(tmax - onset, 1), 0, 1)


def _edit(long, sid, param, fn, rng):
    m = (long["subject"] == sid) & (long["param"] == param) & (long["blfl"] != "Y") & (long["time"] > 1)
    if m.any():
        d = long.loc[m]
        long.loc[m, "value"] = fn(d, _ramp(d, rng))


def inject(long: pd.DataFrame, pattern: str, sid: str, rng: np.random.Generator, k: float = 1.0) -> None:
    base = long[(long["subject"] == sid) & (long["blfl"] == "Y")].groupby("param")["value"].first()
    b = lambda p: base.get(p, np.nan)

    if pattern == "hys_law":
        for p, (lo, hi) in {"ALT": (3, 6), "AST": (3, 6), "BILI": (2, 3)}.items():
            target = rng.uniform(lo, hi)
            _edit(long, sid, p, lambda d, r: d["value"] + k * r * np.maximum(d["hi"] * target - d["value"], 0), rng)
    elif pattern == "aki_relative":
        fc, fb = rng.uniform(0.5, 1.0) * k, rng.uniform(0.3, 0.6) * k
        _edit(long, sid, "CREAT", lambda d, r: b("CREAT") * (1 + r * fc) * rng.normal(1, .03, len(d)), rng)
        _edit(long, sid, "BUN", lambda d, r: b("BUN") * (1 + r * fb) * rng.normal(1, .05, len(d)), rng)
    elif pattern == "plt_decline":
        f = rng.uniform(0.35, 0.50) * k
        _edit(long, sid, "PLAT",
              lambda d, r: np.maximum(b("PLAT") * (1 - r * f), d["lo"] * 1.05) * rng.normal(1, .03, len(d)), rng)
        _edit(long, sid, "WBC", lambda d, r: d["value"] * (1 - 0.15 * k * r), rng)
    elif pattern == "anion_gap":
        _edit(long, sid, "SODIUM", lambda d, r: d["value"] + k * r * (d["hi"] - 1 - d["value"]).clip(lower=0), rng)
        _edit(long, sid, "CL", lambda d, r: d["value"] + k * r * (d["lo"] + 2 - d["value"]).clip(upper=0), rng)
    elif pattern == "fabricated":
        m = ((long["subject"] == sid) & (long["blfl"] != "Y")).to_numpy().copy()
        m &= rng.random(len(long)) < k  # share k of post-baseline records copied
        long.loc[m, "value"] = long.loc[m, "param"].map(base).fillna(long.loc[m, "value"])
    else:
        raise ValueError(pattern)


def spike(long: pd.DataFrame, per_pattern: int, rng: np.random.Generator, k: float = 1.0):
    """Return a spiked copy and a Series subject -> pattern.

    Only subjects with >= 3 post-baseline visits are eligible, so every planted
    pattern has a trajectory to show.
    """
    long = long.copy()
    nv = long[long["blfl"] != "Y"].groupby("subject")["time"].nunique()
    subjects = rng.choice(nv[nv >= 3].index, per_pattern * len(PATTERNS), replace=False)
    truth = {}
    for i, pat in enumerate(PATTERNS):
        for sid in subjects[i * per_pattern:(i + 1) * per_pattern]:
            inject(long, pat, sid, rng, k)
            truth[sid] = pat
    return long, pd.Series(truth, name="pattern")


def rule_flags(long: pd.DataFrame) -> pd.DataFrame:
    """Range-based comparators on post-baseline records.

    any_oor   : any value outside the reference range
    clin_sig  : any value >3xULN or <0.5xLLN, or Hy's Law (ALT or AST >3xULN with BILI >2xULN at a visit)
    """
    post = long[(long["blfl"] != "Y") & (long["time"] > 1)].copy()
    post["xuln"] = post["value"] / post["hi"]
    post["xlln"] = post["value"] / post["lo"]
    oor = ((post["value"] > post["hi"]) | (post["value"] < post["lo"])).groupby(post["subject"]).any()
    big = ((post["xuln"] > 3) | (post["xlln"] < 0.5)).groupby(post["subject"]).any()
    v = post.pivot_table(index=["subject", "time"], columns="param", values="xuln")
    hy = (((v.get("ALT", 0) > 3) | (v.get("AST", 0) > 3)) & (v.get("BILI", 0) > 2)).groupby("subject").any()
    out = pd.DataFrame({"any_oor": oor, "clin_sig": big | hy.reindex(big.index, fill_value=False)})
    return out.reindex(long["subject"].unique(), fill_value=False)
