#!/usr/bin/env python
"""Benchmark harmonypy on the Tahoe subsample datasets.

Compares two versions of harmonypy:
  - harmony1: version 0.2.0 from PyPI
  - harmony2: current local code (version 2.0.0)

Each benchmark runs in an isolated subprocess with its own venv,
so peak RSS measurements are per-run.

Usage:
    python scripts/benchmark.py                # run all benchmarks and plot
    python scripts/benchmark.py --worker ...   # (internal) run a single benchmark

Setup (done automatically by the driver):
    uv venv .venv-harmony1 && uv pip install --python .venv-harmony1 harmonypy==0.2.0 ...
    uv venv .venv-harmony2 && uv pip install --python .venv-harmony2 -e . ...
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
    """Return current RSS in bytes (not peak).

    Uses /proc/self/status on Linux for accuracy.
    Falls back to resource.getrusage (peak RSS) on macOS.
    """
    if sys.platform == "linux":
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024  # KB -> bytes
    # macOS fallback: ru_maxrss is peak, not current, but it's the best
    # we can do without psutil. Since each phase only grows memory, peak
    # at each checkpoint is close to current.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def run_worker(meta_path, pca_path, batch_var, output_dir, version_label):
    """Run harmony on one dataset, print JSON with time and memory.

    Measures current RSS at three points to isolate costs:
      1. After imports, before loading data (baseline)
      2. After loading data (loading cost = point 2 - point 1)
      3. After harmony completes (harmony cost = point 3 - point 2)

    Saves corrected PCs as parquet in output_dir for comparison with R.
    """
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

    start = time()
    ho = hm.run_harmony(pca, meta, batch_var, verbose=False, device="cpu")
    elapsed = time() - start

    rss_after_harmony = _get_current_rss_bytes()

    # Save corrected PCs
    os.makedirs(output_dir, exist_ok=True)
    pca_basename = os.path.basename(pca_path).replace(".h5", "")
    output_path = os.path.join(output_dir, f"{pca_basename}_{version_label}.parquet")
    corrected = pd.DataFrame(
        ho.Z_corr,
        columns=[f"PC{i+1}" for i in range(n_pcs)],
    )
    corrected.to_parquet(output_path)

    result = {
        "version": version_label,
        "harmonypy_version": hm.__version__,
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
# Driver: sets up venvs, spawns workers, collects results, plots
# ---------------------------------------------------------------------------

# Anchor all paths to the repo root (parent of scripts/)
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBSAMPLE_DIR = os.path.join(REPO_DIR, "data/tahoe-ilya/subsample")
OUTPUT_DIR = os.path.join(REPO_DIR, "data/tahoe-ilya/subsample/results")
BATCH_VAR = "sample"

VENVS = {
    "harmony1": {
        "dir": os.path.join(REPO_DIR, ".venv-harmony1"),
        "install": ["harmonypy==0.2.0"],
        "color": "#F8766D",
    },
    "harmony2": {
        "dir": os.path.join(REPO_DIR, ".venv-harmony2"),
        "install": ["-e", REPO_DIR],
        "color": "#00BFC4",
    },
}

DEPS = ["pandas", "numpy", "h5py", "pyarrow", "torch", "scikit-learn", "scipy"]

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


def _get_python(venv_dir):
    """Get the Python executable path for a venv."""
    if sys.platform == "win32":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def setup_venvs():
    """Create venvs and install dependencies for each version."""
    for label, cfg in VENVS.items():
        venv_dir = cfg["dir"]
        python = _get_python(venv_dir)

        if os.path.exists(python):
            # Check if harmonypy is installed
            result = subprocess.run(
                [python, "-c", "import harmonypy; print(harmonypy.__version__)"],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                ver = result.stdout.strip()
                print(f"  {label}: venv ready (harmonypy {ver})")
                continue

        print(f"  {label}: setting up {venv_dir}...")
        subprocess.run(["uv", "venv", venv_dir], check=True,
                        capture_output=True)
        subprocess.run(
            ["uv", "pip", "install", "--python", python]
            + DEPS + cfg["install"],
            check=True, capture_output=True,
        )
        result = subprocess.run(
            [python, "-c", "import harmonypy; print(harmonypy.__version__)"],
            capture_output=True, text=True, check=True,
        )
        print(f"  {label}: installed harmonypy {result.stdout.strip()}")


def run_benchmark(meta_file, pca_file, version_label):
    """Spawn a subprocess to benchmark one dataset with one version."""
    meta_path = os.path.join(SUBSAMPLE_DIR, meta_file)
    pca_path = os.path.join(SUBSAMPLE_DIR, pca_file)

    if not os.path.exists(meta_path) or not os.path.exists(pca_path):
        print(f"    Skipping (file not found): {meta_file}")
        return None

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    python = _get_python(VENVS[version_label]["dir"])
    proc = subprocess.run(
        [
            python, os.path.abspath(__file__),
            "--worker",
            "--meta", meta_path,
            "--pca", pca_path,
            "--batch-var", BATCH_VAR,
            "--output-dir", OUTPUT_DIR,
            "--version-label", version_label,
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
                  f"{result['rss_total_gb']:.2f} GB")
            return result

    print("    FAILED (no JSON output)")
    return None


def plot_results_4panel(all_results, output_path):
    """Create a 2x2 figure comparing harmony1 and harmony2."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for label, cfg in VENVS.items():
        color = cfg["color"]
        batch_results = all_results.get(f"{label}_batch", [])
        cell_results = all_results.get(f"{label}_cell", [])

        # --- Row 1: Vary batches (fixed 1M cells) ---
        if batch_results:
            x = [r["n_batches"] for r in batch_results]

            ax = axes[0, 0]
            ax.plot(x, [r["rss_harmony_gb"] for r in batch_results],
                    "o-", color=color, label=label)

            ax = axes[0, 1]
            ax.plot(x, [r["time_seconds"] / 60 for r in batch_results],
                    "o-", color=color, label=label)

        # --- Row 2: Vary cells (fixed 800 batches) ---
        if cell_results:
            x = [r["n_cells"] / 1e6 for r in cell_results]

            ax = axes[1, 0]
            ax.plot(x, [r["rss_harmony_gb"] for r in cell_results],
                    "o-", color=color, label=label)

            ax = axes[1, 1]
            ax.plot(x, [r["time_seconds"] / 60 for r in cell_results],
                    "o-", color=color, label=label)

    # Labels and legends
    for i, (ylabel, xlabel_top, xlabel_bot) in enumerate([
        ("Memory (GB)", "Number of batches", "Millions of cells"),
        ("Runtime (minutes)", "Number of batches", "Millions of cells"),
    ]):
        axes[0, i].set_xlabel(xlabel_top)
        axes[0, i].set_ylabel(ylabel)
        axes[0, i].legend()
        axes[1, i].set_xlabel(xlabel_bot)
        axes[1, i].set_ylabel(ylabel)
        axes[1, i].legend()

    for ax, letter in zip(axes.flat, "abcd"):
        ax.set_title(letter, loc="left", fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {output_path}")


def main():
    results_path = os.path.join(REPO_DIR, "data/tahoe-ilya/benchmark_results.json")

    print("=== Setting up virtual environments ===")
    setup_venvs()

    all_results = {}
    for label in VENVS:
        print(f"\n=== Benchmarking {label}: vary batches (1M cells) ===")
        batch_results = []
        for meta_file, pca_file in BATCH_DATASETS:
            print(f"  [{label}] {meta_file}")
            result = run_benchmark(meta_file, pca_file, label)
            if result:
                batch_results.append(result)
        all_results[f"{label}_batch"] = batch_results

        print(f"\n=== Benchmarking {label}: vary cells (800 batches) ===")
        cell_results = []
        for meta_file, pca_file in CELL_DATASETS:
            print(f"  [{label}] {meta_file}")
            result = run_benchmark(meta_file, pca_file, label)
            if result:
                cell_results.append(result)
        all_results[f"{label}_cell"] = cell_results

    # Save results as JSON
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # Plot
    plot_results_4panel(all_results, os.path.join(REPO_DIR, "data/tahoe-ilya/benchmark.png"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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
    parser.add_argument("--version-label", default="harmony2",
                        help="Version label for output filenames")
    args = parser.parse_args()

    if args.worker:
        run_worker(args.meta, args.pca, args.batch_var,
                   args.output_dir, args.version_label)
    else:
        main()
