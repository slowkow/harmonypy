# Test harmonypy
"""
Run just the small test (good for Github actions):

    uv run pytest

The small test data is available on Github.

Or run all of the tests (good for local development):

    uv run tests/test_harmony.py

The medium and large data will be downloaded automatically.
"""
from time import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import os
import sys
import harmonypy as hm


def _get_current_rss_mb():
    """Get current RSS (resident set size) in MB. Works on macOS and Linux.

    Unlike resource.getrusage(RUSAGE_SELF).ru_maxrss which only reports the
    lifetime high-water mark, this returns the *current* RSS — so before/after
    deltas correctly capture memory that was allocated and freed by C++ code.
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
        return info.resident_size / 1024 / 1024
    else:
        # Linux: read from /proc/self/status
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # kB -> MB
        return 0.0


def test_run_harmony_small():
    harmonized = "data/pbmc_3500_pcs_harmony2.tsv.gz"
    if not os.path.exists(harmonized):
        harmonized = "data/pbmc_3500_pcs_harmonized.tsv.gz"
    run_harmony(
        meta_tsv="data/pbmc_3500_meta.tsv.gz",
        pcs_tsv="data/pbmc_3500_pcs.tsv.gz",
        harmonized_tsv=harmonized,
        batch_var="donor"
    )


def test_random_seed():
    print("\n" + "=" * 60)
    print("TEST: test_random_seed")
    print("=" * 60)

    meta_data = pd.read_csv("data/pbmc_3500_meta.tsv.gz", sep="\t")
    data_mat = pd.read_csv("data/pbmc_3500_pcs.tsv.gz", sep="\t")

    def run(random_state):
        ho = hm.run_harmony(data_mat,
                            meta_data, ['donor'],
                            max_iter_harmony=2,
                            max_iter_kmeans=2,
                            verbose=False,
                            random_state=random_state)
        return ho.Z_corr

    # Assert same results when random_state is set.
    # Note: MPS (Apple Silicon) has slight non-determinism, so we use relaxed tolerance
    print("\n--- Testing reproducibility with random_state=42 ---")
    result1 = run(42)
    result2 = run(42)
    diff_same_seed = np.abs(result1 - result2).sum()
    print(f"Difference between two runs with same seed: {diff_same_seed:.6f}")
    np.testing.assert_allclose(result1, result2, rtol=1e-3, atol=1e-4)
    print("✓ Same seed produces similar results (PASSED)")

    # Assert different values when random_state is different
    print("\n--- Testing variability with different seeds ---")
    result3 = run(123)
    result4 = run(456)
    diff_diff_seed = np.abs(result3 - result4).sum()
    print(f"Difference between runs with different seeds: {diff_diff_seed:.2f}")
    assert diff_diff_seed > 1000, f"Expected diff > 1000, got {diff_diff_seed}"
    print("✓ Different seeds produce different results (PASSED)")


def run_harmony(meta_tsv, pcs_tsv, harmonized_tsv, batch_var):
    print("\n" + "=" * 60)
    print("TEST: test_run_harmony")
    print("=" * 60)

    if not os.path.exists(meta_tsv):
        return {"time": 0, "rss_delta_mb": 0}

    # Load input data
    meta_data = pd.read_csv(meta_tsv, sep="\t", low_memory=False)
    data_mat = pd.read_csv(pcs_tsv, sep="\t", low_memory=False)
    data_mat = data_mat.select_dtypes(include=[np.number])

    print("\n--- Input Data ---")
    print(f"data_mat shape: {data_mat.shape} (cells × PCs)")
    print(f"meta_data shape: {meta_data.shape}")
    print(f"meta_data columns: {list(meta_data.columns)[:10]}...")
    print(f"Batch variable '{batch_var}' unique values: {meta_data[batch_var].unique()}")
    print(f"Cells per {batch_var}:\n{meta_data[batch_var].value_counts()}")

    print("\n--- Running Harmony ---")
    import gc, sys
    gc.collect()
    rss_before = _get_current_rss_mb()
    start = time()
    ho = hm.run_harmony(data_mat, meta_data, [batch_var])
    end = time()
    rss_after = _get_current_rss_mb()
    rss_delta_mb = rss_after - rss_before
    print(f"\n✓ Harmony completed in {end - start:.2f} seconds")
    print(f"  RSS before: {rss_before:.1f} MB")
    print(f"  RSS after:  {rss_after:.1f} MB")
    print(f"  RSS delta:  {rss_delta_mb:.1f} MB")

    print("\n--- Harmony Object Info ---")
    print(f"Number of clusters (K): {ho.K}")
    print(f"Number of harmony iterations: {len(ho.objective_harmony)}")
    print(f"K-means rounds per iteration: {ho.kmeans_rounds}")
    print(f"Z_corr shape: {ho.Z_corr.shape} (cells × PCs)")
    print(f"Z_orig shape: {ho.Z_orig.shape}")

    # Check convergence
    print("\n--- Convergence ---")
    print(f"Objective (harmony) history: {[f'{x:.2f}' for x in ho.objective_harmony]}")

    # Z_corr is now cells × PCs (same as input)
    res = pd.DataFrame(ho.Z_corr)
    res.columns = ['PC{}'.format(i + 1) for i in range(res.shape[1])]

    # Compare to expected results from R
    harm = pd.read_csv(harmonized_tsv, sep="\t")
    harm = harm.select_dtypes(include=[np.number])
    print("\n--- Comparison with R Results ---")
    print(f"Expected result shape: {harm.shape}")

    cors = []
    for i in range(res.shape[1]):
        cors.append(pearsonr(res.iloc[:, i].values, harm.iloc[:, i].values))
    
    cors_values = [x[0] for x in cors]
    print(f"Correlations (Python vs R) per PC: {[f'{x:.3f}' for x in cors_values]}")
    print(f"Min correlation: {min(cors_values):.3f}")
    print(f"Mean correlation: {np.mean(cors_values):.3f}")

    # Correlation between test PCs and observed PCs is high
    assert np.all(np.array(cors_values) >= 0.9), f"Some correlations < 0.9: {cors_values}"
    print("✓ All correlations >= 0.9 (PASSED)")
    
    return {"time": end - start, "rss_delta_mb": rss_delta_mb}


def download_data():
    if not os.path.exists("data"):
        os.makedirs("data")
    remote_url = "https://immunogenomics.io/downloads"
    files = [
        "acute_myeloid_obs.tsv.gz",
        "acute_myeloid_pcs.tsv.gz",
        "acute_myeloid_pcs_harmonized.tsv.gz",
    ]
    for file in files:
        if not os.path.exists(f"data/{file}"):
            import wget
            print(f"Downloading {file}")
            wget.download(f"{remote_url}/{file}", f"data/{file}")


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("# Running harmonypy tests")
    print("#" * 60)
    print()
    
    timings = {}

    download_data()
    
    timings['small'] = run_harmony(
        meta_tsv="data/pbmc_3500_meta.tsv.gz",
        pcs_tsv="data/pbmc_3500_pcs.tsv.gz",
        harmonized_tsv="data/pbmc_3500_pcs_harmonized.tsv.gz",
        batch_var="donor"
    )
    
    timings['medium'] = run_harmony(
        meta_tsv="data/ircolitis_blood_cd8_obs.tsv.gz",
        pcs_tsv="data/ircolitis_blood_cd8_pcs.tsv.gz",
        harmonized_tsv="data/ircolitis_blood_cd8_pcs_harmonized.tsv.gz",
        batch_var="batch"
    )
    
    timings['large'] = run_harmony(
        meta_tsv="data/acute_myeloid_obs.tsv.gz",
        pcs_tsv="data/acute_myeloid_pcs.tsv.gz",
        harmonized_tsv="data/acute_myeloid_pcs_harmonized.tsv.gz",
        batch_var="batch"
    )
    
    test_random_seed()
    
    print("\n" + "#" * 60)
    print("# Performance Summary")
    print("#" * 60)
    print(f"  {'Dataset':<22} {'Time':>8} {'RSS delta':>12}")
    print(f"  {'-'*22} {'-'*8} {'-'*12}")
    for label, key in [("Small (3.5k cells)", "small"),
                       ("Medium (69k cells)", "medium"),
                       ("Large (858k cells)", "large")]:
        t = timings[key]
        print(f"  {label:<22} {t['time']:>7.2f}s {t['rss_delta_mb']:>8.1f} MB")

    print("\n" + "#" * 60)
    print("# All tests passed!")
    print("#" * 60)
