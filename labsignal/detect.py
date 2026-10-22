"""UMAP -> HDBSCAN structure + panel-wise Isolation Forest / LOF anomaly ensemble.

Flags are not a fixed contamination fraction:
  1. Features are split into subspaces (organ panels, within-subject variability,
     and all features together). Isolation Forest and LOF are fit in each.
  2. Each detector score becomes an empirical p-value against a reference
     population drawn from a trimmed, smoothed Gaussian copula of that subspace
     (same marginals and correlation as the bulk), with a generalized Pareto
     tail beyond the reference's top 1%.
  3. All p-values are merged per subject with the Cauchy combination test.
  4. The merged statistic is calibrated by parametric bootstrap: the whole
     pipeline is rerun on synthetic same-size datasets from a Gaussian copula of
     the data, and subjects are flagged at a target FDR (Benjamini-Hochberg).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import hdbscan
import numpy as np
import pandas as pd
import umap
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from .features import make_groups, feature_lab


@dataclass
class Config:
    seed: int = 42
    clip: float = 25.0
    umap_neighbors: int = 15
    umap_cluster_dim: int = 10
    min_cluster_size: int | None = None  # default: max(10, n // 100)
    if_trees: int = 200
    lof_k: int = 20
    detectors: tuple = ("if", "lof", "pca")  # pca = score + orthogonal distance
    trim: float = 0.10           # share of most-anomalous subjects left out of each reference
    ref_size: int | None = None  # per-panel reference draws; default clamp(10 * n, 10_000, 50_000)
    boot_trim: float | None = None  # share of top-z subjects left out of the bootstrap null; None = adaptive
    n_boot: int | None = None    # synthetic null datasets; default ~10,000 null subjects in total
    fdr: float = 0.05
    panels: dict | None = None   # None = built-in CDISC-style panels; {} = global only
    run_umap: bool = True


@dataclass
class Result:
    scores: pd.DataFrame          # per subject: combined z/p, flag, top panel, cluster
    group_p: pd.DataFrame         # per subject x panel combined p-value
    X: pd.DataFrame               # scaled, imputed feature matrix
    groups: dict
    embedding2d: np.ndarray | None
    config: dict
    timings: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict, repr=False)

    @property
    def flagged(self) -> pd.Index:
        return self.scores.index[self.scores["flag"]]


def preprocess(F: pd.DataFrame, clip: float) -> pd.DataFrame:
    """Median-impute, robust-scale (median / IQR as SD units), clip."""
    X = F.astype(float)
    X = X.fillna(X.median())
    med = X.median()
    iqr = X.quantile(0.75) - X.quantile(0.25)
    scale = iqr.where(iqr > 0, X.std()).replace(0, 1.0)
    return ((X - med) / (scale / 1.349)).clip(-clip, clip)


def copula_reference(X: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw from a Gaussian copula with kernel-smoothed empirical marginals of X."""
    n, p = X.shape
    Z = stats.norm.ppf((stats.rankdata(X, axis=0) - 0.5) / n)
    C = np.atleast_2d(np.corrcoef(Z, rowvar=False))
    C = np.nan_to_num(C) * 0.95 + np.eye(p) * 0.05
    U = stats.norm.cdf(rng.multivariate_normal(np.zeros(p), C, size=size, method="cholesky"))
    sd = X.std(axis=0)
    iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
    bw = 0.9 * np.minimum(sd, np.where(iqr > 0, iqr / 1.349, sd)) * n ** (-0.2) + 1e-9
    out = np.empty((size, p))
    for j in range(p):
        out[:, j] = np.quantile(X[:, j], U[:, j]) + rng.normal(0, bw[j], size)
    return out


def empirical_p(s: np.ndarray, ref: np.ndarray, tail: float = 0.01) -> np.ndarray:
    """P(ref score >= s); generalized Pareto extrapolation beyond the top `tail` of ref."""
    ref = np.sort(ref)
    N = len(ref)
    p = (1 + N - np.searchsorted(ref, s, side="left")) / (N + 1)
    tail = max(tail, min(0.1, 50 / N))  # small nulls: still fit the tail on ~50 exceedances
    u = ref[int(N * (1 - tail))]
    exc = ref[ref > u] - u
    over = s > u
    if over.any() and len(exc) >= 30:
        c, _, scale = stats.genpareto.fit(exc, floc=0)
        if c < 0:  # bounded tail would give p = 0 past its endpoint; use exponential
            c, scale = 0.0, exc.mean()
        p[over] = tail * stats.genpareto.sf(s[over] - u, c, 0, scale)
    return np.clip(p, 1e-300, 1 - 1e-12)


def cauchy_combine(P: np.ndarray) -> np.ndarray:
    """Cauchy combination test (Liu & Xie 2020), valid under arbitrary dependence.

    Inputs are capped at 0.99: p-values near 1 map to huge negative Cauchy terms
    that would otherwise cancel a strong signal elsewhere.
    """
    P = np.clip(P, 1e-300, 0.99)
    T = np.tan((0.5 - P) * np.pi).mean(axis=1)
    return np.clip(0.5 - np.arctan(T) / np.pi, 1e-300, 1 - 1e-12)


def bh(p: np.ndarray, q: float) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    passed = p[order] <= q * np.arange(1, n + 1) / n
    k = passed.nonzero()[0].max() + 1 if passed.any() else 0
    flag = np.zeros(n, bool)
    flag[order[:k]] = True
    return flag


def _bh_q(p: np.ndarray) -> np.ndarray:
    n = len(p)
    o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1)
    return out


def _panel_pvalues(X: np.ndarray, cfg: Config, rng, size: int):
    """One empirical p-value per detector for a feature subspace."""
    n = len(X)
    iso = IsolationForest(n_estimators=cfg.if_trees, random_state=cfg.seed, n_jobs=-1).fit(X)
    lof = LocalOutlierFactor(n_neighbors=min(cfg.lof_k, n - 1), novelty=True, n_jobs=-1).fit(X)
    s = {"if": -iso.score_samples(X), "lof": -lof.negative_outlier_factor_}  # LOF training scores exclude self
    # Reference core: drop subjects that ANY lens ranks as extreme (LOF alone is
    # blind to duplicated anomalies; the robust norm catches what it misses).
    prelim = np.maximum.reduce([stats.rankdata(v) for v in (s["if"], s["lof"], (X ** 2).sum(1))])
    core = X[prelim <= np.quantile(prelim, 1 - cfg.trim)]
    R = copula_reference(core, size, rng)
    r = {"if": -iso.score_samples(R), "lof": -lof.score_samples(R)}
    if "pca" in cfg.detectors and X.shape[1] > 1:
        # score distance (within the main components) + orthogonal distance (off them)
        mu = core.mean(0)
        _, sv, Vt = np.linalg.svd(core - mu, full_matrices=False)
        lam = sv ** 2 / (len(core) - 1)
        k = int(np.clip(np.searchsorted(np.cumsum(lam) / lam.sum(), 0.7) + 1, 1, min(10, len(lam) - 1)))
        V, lam_k = Vt[:k].T, lam[:k]
        for M, out in [(X, s), (R, r)]:
            T = (M - mu) @ V
            out["pca_sd"] = (T ** 2 / lam_k).sum(1)
            out["pca_od"] = np.linalg.norm((M - mu) - T @ V.T, axis=1)
    return {d: empirical_p(s[d], r[d]) for d in s}, iso


def _combined(Xdf: pd.DataFrame, groups: dict, cfg: Config, rng):
    """Panel-wise detectors -> one Cauchy-combined z per subject."""
    n = len(Xdf)
    size = cfg.ref_size or int(np.clip(10 * n, 10_000, 50_000))
    cols, gp, models = {}, {}, {}
    for name, feats in groups.items():
        ps, iso = _panel_pvalues(Xdf[feats].to_numpy(), cfg, rng, size)
        cols.update({f"{name}|{d}": v for d, v in ps.items()})
        gp[name] = cauchy_combine(np.column_stack(list(ps.values())))
        models[name] = iso
    z = stats.norm.isf(cauchy_combine(np.column_stack(list(cols.values()))))
    return z, pd.DataFrame(gp, index=Xdf.index), models


def _bootstrap_p(Xdf: pd.DataFrame, z: np.ndarray, groups: dict, cfg: Config, rng, n_boot: int):
    """Parametric bootstrap: rerun the identical pipeline on same-size synthetic
    datasets drawn from a Gaussian copula of the (trimmed) data, and read each
    subject's p-value off the pooled null distribution of z."""
    n = len(Xdf)
    boot_trim = cfg.boot_trim
    if boot_trim is None:  # adaptive: share of clearly atypical subjects, within [10%, 25%]
        med = np.median(z)
        boot_trim = float(np.clip(np.mean(z > med + 3 * stats.median_abs_deviation(z, scale="normal")), 0.10, 0.25))
    core = Xdf.to_numpy()[z <= np.quantile(z, 1 - boot_trim)]
    z_null = [_combined(pd.DataFrame(copula_reference(core, n, rng), columns=Xdf.columns), groups, cfg, rng)[0]
              for _ in range(n_boot)]
    return empirical_p(z, np.concatenate(z_null)), boot_trim, core


def run(F: pd.DataFrame, cfg: Config | None = None) -> Result:
    cfg = cfg or Config()
    rng = np.random.default_rng(cfg.seed)
    t = {}
    t0 = time.perf_counter()
    Xdf = preprocess(F, cfg.clip)
    n = len(Xdf)
    groups = make_groups(Xdf.columns, cfg.panels)
    t["preprocess"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    z, group_p, models = _combined(Xdf, groups, cfg, rng)
    t["detect"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    n_boot = cfg.n_boot or int(np.clip(np.ceil(10_000 / n), 3, 20))
    p, boot_trim, core = _bootstrap_p(Xdf, z, groups, cfg, rng, n_boot)
    t["calibrate"] = time.perf_counter() - t0
    t["boot_trim"] = boot_trim

    flag = bh(p, cfg.fdr)

    panels_only = group_p.drop(columns=["global"], errors="ignore")
    scores = pd.DataFrame({
        "z": z, "p": p, "q": _bh_q(p), "flag": flag,
        "top_panel": panels_only.idxmin(axis=1) if panels_only.shape[1] else "global",
        "top_panel_p": panels_only.min(axis=1) if panels_only.shape[1] else group_p["global"],
    }, index=Xdf.index)

    emb2 = None
    if cfg.run_umap:
        t0 = time.perf_counter()
        X = Xdf.to_numpy()
        k = min(cfg.umap_neighbors, n - 1)
        emb = umap.UMAP(n_components=min(cfg.umap_cluster_dim, X.shape[1]), n_neighbors=k, min_dist=0.0,
                        random_state=cfg.seed).fit_transform(X)
        mcs = cfg.min_cluster_size or max(10, n // 100)
        hdb = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=min(mcs, 10)).fit(emb)
        scores["cluster"] = hdb.labels_
        scores["glosh"] = hdb.outlier_scores_
        emb2 = umap.UMAP(n_components=2, n_neighbors=k, min_dist=0.1, random_state=cfg.seed).fit_transform(X)
        t["umap_hdbscan"] = time.perf_counter() - t0

    return Result(scores, group_p, Xdf, groups, emb2, asdict(cfg), t, models)


def explain(res: Result, subjects=None, top: int = 3) -> pd.DataFrame:
    """Per subject: most anomalous panel, and its top features by Isolation Forest SHAP."""
    import shap
    subjects = res.flagged if subjects is None else pd.Index(subjects)
    rows = []
    for sid in subjects:
        panel = res.scores.at[sid, "top_panel"]
        feats = res.groups[panel]
        x = res.X.loc[[sid], feats]
        contrib = -shap.TreeExplainer(res.models[panel]).shap_values(x.to_numpy())[0]
        best = [feats[i] for i in np.argsort(contrib)[::-1][:top]]
        rows.append({"subject": sid, "panel": panel, "panel_p": res.scores.at[sid, "top_panel_p"],
                     "drivers": "; ".join(f"{c} ({x.iat[0, feats.index(c)]:+.1f})" for c in best),
                     "labs": ", ".join(dict.fromkeys(feature_lab(c) for c in best))})
    cols = ["panel", "panel_p", "drivers", "labs"]
    return pd.DataFrame(rows).set_index("subject") if rows else pd.DataFrame(columns=cols)


def stability(F: pd.DataFrame, seeds=range(10), cfg: Config | None = None) -> dict:
    """Flag-set Jaccard and cluster ARI across random seeds."""
    from sklearn.metrics import adjusted_rand_score
    base = cfg or Config()
    runs = [run(F, Config(**{**asdict(base), "seed": s})) for s in seeds]
    flags = [set(r.flagged) for r in runs]
    jac = [len(a & b) / max(1, len(a | b)) for i, a in enumerate(flags) for b in flags[i + 1:]]
    out = {"flag_jaccard_mean": float(np.mean(jac)), "flag_jaccard_min": float(np.min(jac)),
           "n_flagged": [len(f) for f in flags]}
    if base.run_umap:
        labs = [r.scores["cluster"].to_numpy() for r in runs]
        ari = [adjusted_rand_score(a, b) for i, a in enumerate(labs) for b in labs[i + 1:]]
        out["cluster_ari_mean"] = float(np.mean(ari))
        out["n_clusters"] = [int(l.max() + 1) for l in labs]
    return out
