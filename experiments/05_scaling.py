"""Laptop scaling: end-to-end runtime and peak memory vs number of subjects.

NHANES adults resampled with replacement (plus tiny jitter so rows are not
exact copies) to 1k .. 50k subjects x 43 analytes.
"""
import json
import resource
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIZES = [1_000, 5_000, 10_000, 25_000, 50_000]

CHILD = r'''
import json, resource, sys, time
import numpy as np, pandas as pd, labsignal as ls
n = int(sys.argv[1])
long = pd.read_csv("data/processed/nhanes_long.csv.gz", dtype={"USUBJID": str})
long = ls.standardize(long, ls.ColumnMap.generic(subject="USUBJID", param="PARAMCD", value="AVAL"))
F = ls.subject_features(long)
rng = np.random.default_rng(0)
F = F.iloc[rng.integers(0, len(F), n)].reset_index(drop=True)
F = F * rng.normal(1, 0.002, F.shape)
F.index = F.index.astype(str)
t = time.perf_counter()
r = ls.run(F, ls.Config(seed=0))
wall = time.perf_counter() - t
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # bytes on macOS -> MB
print(json.dumps({"n": n, "features": F.shape[1], "records_equiv": n * 43, "wall_s": wall, "peak_mb": peak,
                  "flagged": int(r.scores.flag.sum()), "timings": r.timings}))
'''


def main():
    rows = []
    for n in SIZES:
        res = subprocess.run([sys.executable, "-W", "ignore", "-c", CHILD, str(n)], cwd=ROOT,
                             capture_output=True, text=True, env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})
        line = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else res.stderr[-500:]
        print(line, flush=True)
        rows.append(json.loads(line))
    json.dump({"machine": _machine(), "runs": rows}, open(ROOT / "results/scaling.json", "w"), indent=2)


def _machine():
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 2**30
        cores = subprocess.run(["sysctl", "-n", "hw.ncpu"], capture_output=True, text=True).stdout.strip()
        return f"{chip}, {cores} cores, {mem:.0f} GB RAM"
    except Exception:
        return None


if __name__ == "__main__":
    main()
