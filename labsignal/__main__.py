"""Command line: python -m labsignal lab_data.xpt --out results/"""
import argparse

from . import Config, run, standardize, subject_features
from .io import guess_columns, read_table, write_outputs


def main(argv=None):
    ap = argparse.ArgumentParser(prog="labsignal", description="Local unsupervised anomaly detection for lab data.")
    ap.add_argument("input", help="ADLB-like or long-format lab file (.xpt, .sas7bdat, .csv, .parquet)")
    ap.add_argument("--out", default="labsignal_out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--no-umap", action="store_true", help="skip UMAP/HDBSCAN structure step")
    for name in ["subject", "param", "value", "time", "lo", "hi"]:
        ap.add_argument(f"--{name}-col", help=f"override the detected {name} column")
    a = ap.parse_args(argv)

    df = read_table(a.input)
    cm = guess_columns(df)
    for name in ["subject", "param", "value", "time", "lo", "hi"]:
        if getattr(a, f"{name}_col"):
            setattr(cm, name, getattr(a, f"{name}_col"))
    print(f"columns: {vars(cm)}")
    F = subject_features(standardize(df, cm))
    print(f"{len(F):,} subjects x {F.shape[1]} features")
    res = run(F, Config(seed=a.seed, fdr=a.fdr, run_umap=not a.no_umap))
    out = write_outputs(res, a.out, cm, a.input)
    print(f"{res.scores['flag'].sum()} flagged at FDR {a.fdr:.0%}; results in {out}/")


if __name__ == "__main__":
    main()
