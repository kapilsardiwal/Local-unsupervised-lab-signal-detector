"""Reading lab extracts and writing an audit manifest for each run."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd

from .features import ColumnMap

PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "umap-learn", "hdbscan", "shap"]


def read_table(path_or_buffer, name: str | None = None) -> pd.DataFrame:
    name = (name or str(path_or_buffer)).lower()
    if name.endswith(".xpt"):
        return pd.read_sas(path_or_buffer, format="xport", encoding="latin1")
    if name.endswith(".sas7bdat"):
        return pd.read_sas(path_or_buffer, format="sas7bdat", encoding="latin1")
    if name.endswith((".parquet", ".pq")):
        return pd.read_parquet(path_or_buffer)
    return pd.read_csv(path_or_buffer)


def guess_columns(df: pd.DataFrame) -> ColumnMap:
    """ADLB names when present, otherwise common long-format names."""
    cols = {c.upper(): c for c in df.columns}
    pick = lambda *names: next((cols[n] for n in names if n in cols), None)
    return ColumnMap(
        subject=pick("USUBJID", "SUBJID", "SUBJECT", "SEQN", "ID") or df.columns[0],
        param=pick("PARAMCD", "LBTESTCD", "PARAM", "TEST", "ANALYTE") or df.columns[1],
        value=pick("AVAL", "LBSTRESN", "VALUE", "RESULT") or df.columns[2],
        time=pick("ADY", "LBDY", "DAY", "TIME", "VISITNUM"),
        lo=pick("A1LO", "ANRLO", "LBSTNRLO", "LO", "LLN"),
        hi=pick("A1HI", "ANRHI", "LBSTNRHI", "HI", "ULN"),
        baseline_flag=pick("ABLFL", "LBBLFL"),
    )


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(res, columns: ColumnMap, input_path=None, input_bytes: bytes | None = None) -> dict:
    """Everything needed to reproduce or audit a run. Contains no patient data."""
    from . import __version__
    digest = sha256(input_path) if input_path else hashlib.sha256(input_bytes).hexdigest() if input_bytes else None
    return {
        "tool": "labsignal", "version": __version__,
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": {"file": Path(input_path).name if input_path else None, "sha256": digest},
        "columns": vars(columns), "config": res.config,
        "subjects": len(res.scores), "features": res.X.shape[1],
        "panels": {g: len(v) for g, v in res.groups.items()},
        "n_flagged": int(res.scores["flag"].sum()),
        "timings_s": {k: round(v, 3) for k, v in res.timings.items()},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        **{p: _version(p) for p in PACKAGES}},
        "network_calls": 0,
    }


def _version(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def write_outputs(res, out_dir, columns: ColumnMap, input_path=None) -> Path:
    from .detect import explain
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scores = res.scores.join(explain(res))
    scores.to_csv(out / "subject_scores.csv")
    scores[scores["flag"]].sort_values("p").to_csv(out / "flagged_subjects.csv")
    json.dump(manifest(res, columns, input_path), open(out / "run_manifest.json", "w"), indent=2, default=str)
    return out
