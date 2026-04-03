#!/usr/bin/env python
"""Benchmark harmonypy on the Tahoe subsample datasets.

Each benchmark runs in a subprocess (for clean RSS measurement)
using the SAME Python environment that invokes this script.
No separate venvs are created.

Usage:
    # Install harmonypy first, then run:
    uv pip install -e .
    python scripts/benchmark.py

    # Run a single benchmark (internal, used by subprocess):
    python scripts/benchmark.py --worker --meta PATH --pca PATH
"""

import argparse
import json
import os
import resource
import subprocess
import sys
from time import time


# ---------------------------------------------------------------------------
# Worker: runs in a subprocess so peak RSS is isolated per benchmark
# ---------------------------------------------------------------------------

def _get_current_rss_bytes():
    """Return current RSS in bytes.

    Uses /proc/self/status on Linux for accuracy.
    Falls back to resource.getrusage (peak RSS) on macOS.
    """
    if sys.platform == "linux":
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024  # KB -> bytes
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def run_worker(meta_path, pca_path, batch_var, output_dir):
    """Run harmony on one dataset, print JSON with time and memory."""
    import h5py
    import numpy as np
    import pandas as pd
    import harmonypy as hm

    rss_baseline = _get_current_rss_bytes()

    meta = pd.read_parquet(meta_path)
    with h5py.File(pca_path, "r") as f:
        pca = np.array(f["pca"], dtype=np.float32)

    n_cells, n_pcs = pca.shape
    n_batches = meta[batch_var].nunique()

    rss_after_load = _get_current_rss_bytes()

    # Check which backend is being used
    try:
        from harmonypy._harmony_cpp import HarmonyCpp
        backend_available = "cpp"
    except ImportError:
        backend_available = "pytorch"

    start = time()
    ho = hm.run_harmony(pca, meta, batch_var, verbose=False, device="cpu")
    elapsed = time() - start

    rss_after_harmony = _get_current_rss_bytes()

    # Save corrected PCs
    os.makedirs(output_dir, exist_ok=True)
    pca_basename = os.path.basename(pca_path).replace(".h5", "")
    output_path = os.path.join(output_dir, f"{pca_basename}_harmony2.parquet")
    corrected = pd.DataFrame(
        ho.Z_corr,
        columns=[f"PC{i+1}" for i in range(n_pcs)],
    )
    corrected.to_parquet(output_path)

    result = {
        "harmonypy_version": hm.__version__,
        "backend": backend_available,
        "meta": meta_path,
        "pca": pca_path,
        "output": output_path,
        "n_cells": n_cells,
        "n_batches": n_batches,
        "time_seconds": round(elapsed, 1),
        "rss_baseline_gb": round(rss_baseline / 1e9, 2),
        "rss_loading_gb": round((rss_after_load - rss_baseline) / 1e9, 2),
        "rss_harmony_gb": round((rss_after_harmony - rss_after_load) / 1e9, 2),
        "rss_total_gb": round(rss_after_harmony / 1e9, 2),
    }
    print(json.dumps(result))


# ---------------------------------------------------------------------------
# Driver: spawns workers using the current Python, collects results, plots
# ---------------------------------------------------------------------------

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBSAMPLE_DIR = os.path.join(REPO_DIR, "data/tahoe-ilya/subsample")
OUTPUT_DIR = os.path.join(REPO_DIR, "data/tahoe-ilya/subsample/results")
BATCH_VAR = "sample"

# Vary batches (fixed 1M cells)
BATCH_DATASETS = [
    ("meta-50B.parquet", "pca-50B.h5"),
    ("meta-100B.parquet", "pca-100B.h5"),
    ("meta-200B.parquet", "pca-200B.h5"),
    ("meta-400B.parquet", "pca-400B.h5"),
    ("meta-800B.parquet", "pca-800B.h5"),
]

# Vary cells (fixed 800 batches)
CELL_DATASETS = [
    ("meta-1M.parquet", "pca-1M.h5"),
    ("meta-2M.parquet", "pca-2M.h5"),
    ("meta-4M.parquet", "pca-4M.h5"),
    ("meta-8M.parquet", "pca-8M.h5"),
    ("meta-16M.parquet", "pca-16M.h5"),
]


def run_benchmark(meta_file, pca_file):
    """Spawn a subprocess to benchmark one dataset."""
    meta_path = os.path.join(SUBSAMPLE_DIR, meta_file)
    pca_path = os.path.join(SUBSAMPLE_DIR, pca_file)

    if not os.path.exists(meta_path) or not os.path.exists(pca_path):
        print(f"    Skipping (file not found): {meta_file}")
        return None

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Use the SAME Python that is running this script
    proc = subprocess.run(
        [
            sys.executable, os.path.abspath(__file__),
            "--worker",
            "--meta", meta_path,
            "--pca", pca_path,
            "--batch-var", BATCH_VAR,
            "--output-dir", OUTPUT_DIR,
        ],
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0:
        print(f"    FAILED (exit {proc.returncode})")
        if proc.stderr:
            for line in proc.stderr.strip().splitlines()[-5:]:
                print(f"    {line}")
        return None

    # Last line of stdout is the JSON result
    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            result = json.loads(line)
            print(f"    {result['n_cells']:>12,} cells, {result['n_batches']:>5} batches "
                  f"-> {result['time_seconds']:>7.1f}s, "
                  f"baseline {result['rss_baseline_gb']:.2f} + "
                  f"load {result['rss_loading_gb']:.2f} + "
                  f"harmony {result['rss_harmony_gb']:.2f} = "
                  f"{result['rss_total_gb']:.2f} GB"
                  f"  [{result['backend']}]")
            return result

    print("    FAILED (no JSON output)")
    return None


def plot_results(all_results, output_path):
    """Create a 2-panel figure (memory + time) for both sweeps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    color = "#00BFC4"

    batch_results = all_results.get("batch", [])
    cell_results = all_results.get("cell", [])

    if batch_results:
        x = [r["n_batches"] for r in batch_results]
        axes[0, 0].plot(x, [r["rss_harmony_gb"] for r in batch_results], "o-", color=color)
        axes[0, 0].set_xlabel("Number of batches")
        axes[0, 0].set_ylabel("Memory (GB)")
        axes[0, 1].plot(x, [r["time_seconds"] / 60 for r in batch_results], "o-", color=color)
        axes[0, 1].set_xlabel("Number of batches")
        axes[0, 1].set_ylabel("Runtime (minutes)")

    if cell_results:
        x = [r["n_cells"] / 1e6 for r in cell_results]
        axes[1, 0].plot(x, [r["rss_harmony_gb"] for r in cell_results], "o-", color=color)
        axes[1, 0].set_xlabel("Millions of cells")
        axes[1, 0].set_ylabel("Memory (GB)")
        axes[1, 1].plot(x, [r["time_seconds"] / 60 for r in cell_results], "o-", color=color)
        axes[1, 1].set_xlabel("Millions of cells")
        axes[1, 1].set_ylabel("Runtime (minutes)")

    for ax, letter in zip(axes.flat, "abcd"):
        ax.set_title(letter, loc="left", fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {output_path}")


def main():
    results_path = os.path.join(REPO_DIR, "data/tahoe-ilya/benchmark_results.json")

    # Show what we're benchmarking
    import harmonypy as hm
    print(f"harmonypy {hm.__version__}")
    print(f"Python: {sys.executable}")
    try:
        from harmonypy._harmony_cpp import HarmonyCpp
        print("Backend: C++ (Armadillo)")
    except ImportError:
        print("Backend: PyTorch (C++ extension not available)")
    print()

    all_results = {}

    print("=== Vary batches (1M cells) ===")
    batch_results = []
    for meta_file, pca_file in BATCH_DATASETS:
        print(f"  {meta_file}")
        result = run_benchmark(meta_file, pca_file)
        if result:
            batch_results.append(result)
    all_results["batch"] = batch_results

    print(f"\n=== Vary cells (800 batches) ===")
    cell_results = []
    for meta_file, pca_file in CELL_DATASETS:
        print(f"  {meta_file}")
        result = run_benchmark(meta_file, pca_file)
        if result:
            cell_results.append(result)
    all_results["cell"] = cell_results

    # Save results
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # Plot
    plot_results(all_results, os.path.join(REPO_DIR, "data/tahoe-ilya/benchmark.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true",
                        help="Run a single benchmark (internal)")
    parser.add_argument("--meta", help="Path to metadata parquet file")
    parser.add_argument("--pca", help="Path to PCA h5 file")
    parser.add_argument("--batch-var", default="sample",
                        help="Batch variable name")
    parser.add_argument("--output-dir", default=".",
                        help="Directory to save corrected PCs")
    args = parser.parse_args()

    if args.worker:
        run_worker(args.meta, args.pca, args.batch_var, args.output_dir)
    else:
        main()
