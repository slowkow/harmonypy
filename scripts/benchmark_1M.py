#!/usr/bin/env python
"""Quick benchmark: run harmony on 1M cells, report time and memory.

Usage:
    python scripts/benchmark_1M.py
    python scripts/benchmark_1M.py --meta path/to/meta.parquet --pca path/to/pca.h5
    python scripts/benchmark_1M.py --ncores 4
"""

import argparse
import os
import sys
from time import time


def get_rss_gb():
    """Return current RSS in GB (not peak).

    Uses mach_task_basic_info on macOS and /proc/self/status on Linux.
    """
    if sys.platform == "darwin":
        import ctypes, ctypes.util
        libc = ctypes.CDLL(ctypes.util.find_library("c"))

        MACH_TASK_BASIC_INFO = 20
        class mach_task_basic_info(ctypes.Structure):
            _fields_ = [
                ("virtual_size", ctypes.c_uint64),
                ("resident_size", ctypes.c_uint64),
                ("resident_size_max", ctypes.c_uint64),
                ("user_time_seconds", ctypes.c_uint32),
                ("user_time_microseconds", ctypes.c_uint32),
                ("system_time_seconds", ctypes.c_uint32),
                ("system_time_microseconds", ctypes.c_uint32),
                ("policy", ctypes.c_int32),
                ("suspend_count", ctypes.c_int32),
            ]

        info = mach_task_basic_info()
        count = ctypes.c_uint32(ctypes.sizeof(info) // 4)
        libc.task_info(
            libc.mach_task_self(),
            MACH_TASK_BASIC_INFO,
            ctypes.byref(info),
            ctypes.byref(count),
        )
        return info.resident_size / 1e9
    elif sys.platform == "linux":
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024 / 1e9
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def main():
    parser = argparse.ArgumentParser(description="Benchmark harmonypy on a large dataset")
    parser.add_argument("--meta", default="data/tahoe-ilya/subsample/meta-1M.parquet",
                        help="Path to metadata parquet file")
    parser.add_argument("--pca", default="data/tahoe-ilya/subsample/pca-1M.h5",
                        help="Path to PCA h5 file")
    parser.add_argument("--batch-var", default="sample",
                        help="Batch variable name in metadata")
    parser.add_argument("--ncores", type=int, default=1,
                        help="Number of BLAS threads (Linux only)")
    args = parser.parse_args()

    if not os.path.exists(args.meta):
        print(f"Error: {args.meta} not found")
        sys.exit(1)
    if not os.path.exists(args.pca):
        print(f"Error: {args.pca} not found")
        sys.exit(1)

    import gc
    gc.collect()
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

    gc.collect()
    rss_after_load = get_rss_gb()
    print(f"After load:   {rss_after_load:.2f} GB (+{rss_after_load - rss_baseline:.2f} GB)")
    print(f"Data:         {n_cells:,} cells, {n_pcs} PCs, {n_batches} batches")

    import harmonypy as hm
    print(f"harmonypy:    {hm.__version__}")
    print(f"ncores:       {args.ncores}")

    print(f"\nRunning harmony...")
    gc.collect()
    rss_before_harmony = get_rss_gb()
    start = time()
    ho = hm.run_harmony(pca, meta, args.batch_var, verbose=True, ncores=args.ncores)
    elapsed = time() - start
    rss_after_harmony = get_rss_gb()

    print(f"\nResults:")
    print(f"  Time:          {elapsed:.1f}s")
    print(f"  RSS before:    {rss_before_harmony:.2f} GB")
    print(f"  RSS after:     {rss_after_harmony:.2f} GB")
    print(f"  RSS harmony:   {rss_after_harmony - rss_before_harmony:.2f} GB")
    print(f"  Z_corr shape:  {ho.Z_corr.shape}")


if __name__ == "__main__":
    main()
