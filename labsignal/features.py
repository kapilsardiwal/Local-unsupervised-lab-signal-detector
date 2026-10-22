"""Long-format lab records -> one row per subject.

Works for longitudinal (ADLB-like, many visits) and cross-sectional (one draw
per subject) data. Only `subject`, `param` and `value` are required; time and
reference ranges unlock trajectory and xULN features when present.

Feature names are always "<PARAM>_<stat>".
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ColumnMap:
    subject: str = "USUBJID"
    param: str = "PARAMCD"
    value: str = "AVAL"
    time: str | None = "ADY"
    lo: str | None = "A1LO"
    hi: str | None = "A1HI"
    baseline_flag: str | None = "ABLFL"

    @classmethod
    def adlb(cls) -> "ColumnMap":
        return cls()

    @classmethod
    def generic(cls, subject="subject", param="param", value="value", time=None, lo=None, hi=None):
        return cls(subject, param, value, time, lo, hi, None)


def standardize(df: pd.DataFrame, cm: ColumnMap) -> pd.DataFrame:
    """Rename to internal names, drop non-numeric/empty values."""
    cols = {cm.subject: "subject", cm.param: "param", cm.value: "value"}
    for src, dst in [(cm.time, "time"), (cm.lo, "lo"), (cm.hi, "hi"), (cm.baseline_flag, "blfl")]:
        if src and src in df.columns:
            cols[src] = dst
    out = df[list(cols)].rename(columns=cols).copy()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"])
    out["subject"] = out["subject"].astype(str)
    out["param"] = out["param"].astype(str).str.replace("_", "", regex=False)
    return out


def _log_ratio(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(a / b)
    return r.replace([np.inf, -np.inf], np.nan)


def subject_features(long: pd.DataFrame, min_param_coverage: float = 0.5,
                     min_post_visits: int = 3) -> pd.DataFrame:
    """Engineer subject-level features from standardized long data.

    Longitudinal, per lab:
      peak   log(max / ULN)          trough  log(min / LLN)
      rise   log(max post / baseline) fall   log(min post / baseline)
      cv     within-subject coefficient of variation
    Without reference ranges, peak/trough fall back to log max / log min.
    Cross-sectional: the value (log scale for positive, right-skewed labs).
    """
    longitudinal = "time" in long and long.groupby(["subject", "param"]).size().median() > 1

    keep = long.groupby("param")["subject"].nunique()
    keep = keep[keep >= min_param_coverage * long["subject"].nunique()].index
    long = long[long["param"].isin(keep)]

    if not longitudinal:
        w = long.pivot_table(index="subject", columns="param", values="value", aggfunc="mean")
        logged = (w.skew() > 1) & (w.min() > 0)
        w.loc[:, logged] = np.log(w.loc[:, logged])
        w.columns = [f"{c}_value" for c in w.columns]
        return w

    long = long.sort_values(["subject", "param", "time"])
    first = long.groupby(["subject", "param"])["value"].first()
    if "blfl" in long:
        bl = long[long["blfl"] == "Y"].groupby(["subject", "param"])["value"].first()
        post = long[long["blfl"] != "Y"]
    else:
        bl = long[long["time"] <= 1].groupby(["subject", "param"])["value"].last()
        post = long[long["time"] > 1]
    bl = bl.reindex(first.index).fillna(first)

    # small per-lab offset keeps log ratios stable near zero (e.g. basophil counts of 0.0)
    off = (0.1 * long.groupby("param")["value"].median().abs()).replace(0, 1e-3)
    long = long.assign(value=long["value"] + long["param"].map(off))
    post = post.assign(value=post["value"] + post["param"].map(off))
    bl = bl + bl.index.get_level_values("param").map(off).to_numpy()

    g = long.groupby(["subject", "param"])["value"]
    gp = post.groupby(["subject", "param"])["value"]
    has_range = "hi" in long and long["hi"].notna().any()
    if has_range:
        hi = long.groupby(["subject", "param"])["hi"].median()
        lo = long.groupby(["subject", "param"])["lo"].median() if "lo" in long else hi * np.nan
        o = hi.index.get_level_values("param").map(off).to_numpy()
        peak, trough = _log_ratio(g.max(), hi + o), _log_ratio(g.min(), lo + o)
        # labs with no lower limit: fall back to log min
        trough = trough.fillna(_log_ratio(g.min(), 1))
    else:
        peak, trough = _log_ratio(g.max(), 1), _log_ratio(g.min(), 1)
    f = pd.DataFrame({
        "peak": peak,
        "trough": trough,
        "rise": _log_ratio(gp.max(), bl),
        "fall": _log_ratio(gp.min(), bl),
        "cv": g.std() / g.mean().abs().replace(0, np.nan),
    })
    # trajectory summaries from too few draws are noise, not signal
    thin = gp.size().reindex(f.index, fill_value=0) < min_post_visits
    f.loc[thin, ["rise", "fall", "cv"]] = np.nan
    wide = f.unstack("param")
    wide.columns = [f"{p}_{stat}" for stat, p in wide.columns]
    wide = wide.loc[:, wide.notna().mean() > 0.4]
    wide = wide.loc[:, wide.nunique() > 1]

    # data-quality signals used in risk-based monitoring
    prev = post.groupby(["subject", "param"])["value"].shift()
    rep = (post["value"] == prev).where(prev.notna())
    wide["DQ_repeat"] = rep.groupby(post["subject"]).mean()
    wide["DQ_cvmed"] = f["cv"].groupby("subject").median()
    return wide.sort_index(axis=1)


def feature_lab(name: str) -> str:
    return name.rsplit("_", 1)[0]


# Organ-system panels keyed by CDISC LBTESTCD-style codes (plus common aliases).
PANELS = {
    "liver": ["ALT", "AST", "ALP", "BILI", "GGT", "ALB", "PROT", "CHE", "LDH", "GLOBUL"],
    "renal": ["CREAT", "BUN", "URATE", "EGFR", "ACR", "CYSC"],
    "electrolytes": ["SODIUM", "K", "CL", "CA", "PHOS", "BICARB", "MG", "OSMOL"],
    "metabolic": ["GLUC", "HBA1C", "CHOL", "HDL", "LDL", "TRIG", "INSULIN", "CK", "CRP", "IRON", "FERRITIN"],
    "rbc": ["HGB", "HCT", "RBC", "MCV", "MCH", "MCHC", "RDW", "RETI"],
    "wbc_plt": ["WBC", "NEUT", "LYM", "MONO", "EOS", "BASO", "PLAT", "MPV"],
}
ALIASES = {"BIL": "BILI", "TBIL": "BILI", "CREA": "CREAT", "NA": "SODIUM", "PLT": "PLAT", "GLU": "GLUC",
           "UA": "URATE", "TG": "TRIG", "HB": "HGB", "GHB": "HBA1C", "UREA": "BUN", "CO2": "BICARB"}


def make_groups(columns, panels: dict | None = None, include_global: bool = True,
                min_size: int = 3) -> dict[str, list[str]]:
    """Map features to detector subspaces: organ panels, a variability panel, and all features."""
    panels = PANELS if panels is None else panels
    lookup = {lab: name for name, labs in panels.items() for lab in labs}
    groups = defaultdict(list)
    for c in columns:
        lab, stat = c.rsplit("_", 1)
        if lab == "DQ":
            groups["data_quality"].append(c)
        elif stat == "cv":
            groups["variability"].append(c)
        else:
            groups[lookup.get(ALIASES.get(lab, lab), "other")].append(c)
    # panels too small to have a multivariate shape are covered by the global set only
    groups = {g: v for g, v in groups.items() if len(v) >= min_size}
    if include_global and (len(groups) != 1 or len(next(iter(groups.values()))) < len(columns)):
        groups["global"] = list(columns)
    return groups
