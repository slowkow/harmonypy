#!/usr/bin/env python
"""Quick benchmark: run harmony on 1M cells, report time and memory.

Usage:
    python scripts/benchmark_1M.py
    python scripts/benchmark_1M.py --backend cpp
    python scripts/benchmark_1M.py --backend pytorch
"""

import argparse
import os
import resource
import sys
from time import time


def get_rss_gb():
    """Return current RSS in GB."""
    if sys.platform == "linux":
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024 / 1e9
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="auto", choices=["auto", "cpp", "pytorch"])
    parser.add_argument("--meta", default="data/tahoe-ilya/subsample/meta-1M.parquet")
    parser.add_argument("--pca", default="data/tahoe-ilya/subsample/pca-1M.h5")
    parser.add_argument("--batch-var", default="sample")
    args = parser.parse_args()

    rss_baseline = get_rss_gb()
    print(f"Baseline RSS: {rss_baseline:.2f} GB")

    import h5py
    import numpy as np
    import pandas as pd

    meta = pd.read_parquet(args.meta)
    with h5py.File(args.pca, "r") as f:
        pca = np.array(f["pca"], dtype=np.float32)

    n_cells, n_pcs = pca.shape
    n_batches = meta[args.batch_var].nunique()

    rss_after_load = get_rss_gb()
    print(f"After load:   {rss_after_load:.2f} GB (load cost: {rss_after_load - rss_baseline:.2f} GB)")
    print(f"Data: {n_cells:,} cells, {n_pcs} PCs, {n_batches} batches")
    print(f"Backend: {args.backend}")

    import harmonypy as hm
    print(f"harmonypy version: {hm.__version__}")

    # Check if C++ backend is available
    try:
        from harmonypy._harmony_cpp import HarmonyCpp
        print("C++ backend: available")
    except ImportError:
        print("C++ backend: NOT available (will use PyTorch)")

    print(f"\nRunning harmony (backend={args.backend})...")
    start = time()
    ho = hm.run_harmony(pca, meta, args.batch_var, verbose=True, device="cpu", backend=args.backend)
    elapsed = time() - start

    rss_after = get_rss_gb()
    print(f"\nResults:")
    print(f"  Time:          {elapsed:.1f}s")
    print(f"  RSS baseline:  {rss_baseline:.2f} GB")
    print(f"  RSS load:      {rss_after_load - rss_baseline:.2f} GB")
    print(f"  RSS harmony:   {rss_after - rss_after_load:.2f} GB")
    print(f"  RSS total:     {rss_after:.2f} GB")
    print(f"  Z_corr shape:  {ho.Z_corr.shape}")


if __name__ == "__main__":
    main()
