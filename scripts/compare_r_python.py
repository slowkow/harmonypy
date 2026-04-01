#!/usr/bin/env python
"""Compare R harmony2 and Python harmonypy on the same dataset.

Runs R harmony2 via Rscript, then Python harmonypy, saves both outputs
as parquet, and compares per-PC Pearson correlations.

Usage:
    python scripts/compare_r_python.py

Requires:
    R packages: harmony (>= 2.0), arrow, hdf5r
    Python packages: harmonypy, h5py, pandas, numpy, scipy
"""

import os
import subprocess
import sys
import tempfile
from time import time

import h5py
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_DIR, "data/tahoe-ilya/subsample")
OUTPUT_DIR = os.path.join(DATA_DIR, "results")

META_PATH = os.path.join(DATA_DIR, "meta-1M.parquet")
PCA_PATH = os.path.join(DATA_DIR, "pca-1M.h5")
BATCH_VAR = "sample"

R_OUTPUT = os.path.join(OUTPUT_DIR, "pca-1M_R.parquet")
PY_OUTPUT = os.path.join(OUTPUT_DIR, "pca-1M_python.parquet")


# ---------------------------------------------------------------------------
# Step 1: Run R harmony2
# ---------------------------------------------------------------------------

def run_r_harmony():
    """Run harmony2 in R and save corrected PCs as parquet."""
    print("=" * 60)
    print("Step 1: Running R harmony2")
    print("=" * 60)

    r_script = f"""\
library(harmony)
library(arrow)
library(hdf5r)

cat("harmony version:", as.character(packageVersion("harmony")), "\\n")

# Load data
meta <- read_parquet("{META_PATH}")
f <- H5File$new("{PCA_PATH}", mode = "r")
pca <- f[["pca"]]$read()
f$close()

cat("PCA shape:", nrow(pca), "x", ncol(pca), "\\n")
cat("Meta shape:", nrow(meta), "x", ncol(meta), "\\n")
cat("Batches:", length(unique(meta${BATCH_VAR})), "\\n")

# Run harmony
cat("Running harmony...\\n")
t0 <- proc.time()
ho <- harmony::RunHarmony(
    pca,
    meta,
    "{BATCH_VAR}",
    verbose = FALSE
)
elapsed <- (proc.time() - t0)["elapsed"]
cat(sprintf("Harmony completed in %.1f seconds\\n", elapsed))

# Save result
dir.create("{OUTPUT_DIR}", showWarnings = FALSE, recursive = TRUE)
result <- as.data.frame(ho)
colnames(result) <- paste0("PC", seq_len(ncol(result)))
write_parquet(result, "{R_OUTPUT}")
cat("Saved:", "{R_OUTPUT}", "\\n")
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as f:
        f.write(r_script)
        r_script_path = f.name

    try:
        start = time()
        proc = subprocess.run(
            ["Rscript", r_script_path],
            capture_output=True, text=True
        )
        elapsed = time() - start

        if proc.stdout:
            for line in proc.stdout.strip().splitlines():
                print(f"  {line}")

        if proc.returncode != 0:
            print(f"\n  R FAILED (exit {proc.returncode})")
            if proc.stderr:
                for line in proc.stderr.strip().splitlines()[-10:]:
                    print(f"  {line}")
            sys.exit(1)

        print(f"  Total R wall time: {elapsed:.1f}s")
    finally:
        os.unlink(r_script_path)


# ---------------------------------------------------------------------------
# Step 2: Run Python harmonypy
# ---------------------------------------------------------------------------

def run_python_harmony():
    """Run harmonypy and save corrected PCs as parquet."""
    print()
    print("=" * 60)
    print("Step 2: Running Python harmonypy")
    print("=" * 60)

    import harmonypy as hm

    print(f"  harmonypy version: {hm.__version__}")

    meta = pd.read_parquet(META_PATH)
    with h5py.File(PCA_PATH, "r") as f:
        pca = np.array(f["pca"], dtype=np.float32)

    n_cells, n_pcs = pca.shape
    n_batches = meta[BATCH_VAR].nunique()
    print(f"  PCA shape: {n_cells} x {n_pcs}")
    print(f"  Batches: {n_batches}")

    print("  Running harmony...")
    start = time()
    ho = hm.run_harmony(pca, meta, BATCH_VAR, verbose=False, device="cpu")
    elapsed = time() - start
    print(f"  Harmony completed in {elapsed:.1f} seconds")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result = pd.DataFrame(
        ho.Z_corr,
        columns=[f"PC{i+1}" for i in range(n_pcs)],
    )
    result.to_parquet(PY_OUTPUT)
    print(f"  Saved: {PY_OUTPUT}")


# ---------------------------------------------------------------------------
# Step 3: Compare outputs
# ---------------------------------------------------------------------------

def compare_outputs():
    """Compare R and Python harmony outputs via per-PC Pearson correlation."""
    print()
    print("=" * 60)
    print("Step 3: Comparing R vs Python outputs")
    print("=" * 60)

    r_result = pd.read_parquet(R_OUTPUT)
    py_result = pd.read_parquet(PY_OUTPUT)

    print(f"  R result shape:      {r_result.shape}")
    print(f"  Python result shape: {py_result.shape}")

    assert r_result.shape == py_result.shape, \
        f"Shape mismatch: R {r_result.shape} vs Python {py_result.shape}"

    n_pcs = r_result.shape[1]
    cors = []
    for i in range(n_pcs):
        r_col = r_result.iloc[:, i].values
        py_col = py_result.iloc[:, i].values
        cor, _ = pearsonr(r_col, py_col)
        cors.append(cor)

    print()
    print("  Per-PC Pearson correlations (R vs Python):")
    for i, c in enumerate(cors):
        flag = " <---" if abs(c) < 0.9 else ""
        print(f"    PC{i+1:>2}: {c:>7.4f}{flag}")

    print()
    abs_cors = [abs(c) for c in cors]
    print(f"  Min |correlation|:  {min(abs_cors):.4f}")
    print(f"  Mean |correlation|: {np.mean(abs_cors):.4f}")
    print(f"  PCs with |r| >= 0.9: {sum(1 for c in abs_cors if c >= 0.9)}/{n_pcs}")

    if all(c >= 0.9 for c in abs_cors):
        print("\n  PASSED: All PCs have |correlation| >= 0.9")
    else:
        low = [(i+1, c) for i, c in enumerate(abs_cors) if c < 0.9]
        print(f"\n  WARNING: {len(low)} PCs have |correlation| < 0.9:")
        for pc, c in low:
            print(f"    PC{pc}: {c:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Data: {META_PATH}")
    print(f"      {PCA_PATH}")
    print(f"Batch variable: {BATCH_VAR}")
    print()

    run_r_harmony()
    run_python_harmony()
    compare_outputs()
