# vprof -c p tests/test_harmony.py
from time import time

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from scipy.stats import pearsonr
import sys

import harmonypy as hm


def test_run_harmony(meta_tsv, pcs_tsv, harmonized_tsv, batch_var):
    print("\n" + "=" * 60)
    print("TEST: test_run_harmony")
    print("=" * 60)

    # Load input data
    meta_data = pd.read_csv(meta_tsv, sep="\t", low_memory=False)
    data_mat = pd.read_csv(pcs_tsv, sep="\t", low_memory=False)

    if data_mat.iloc[:,0].dtype == 'object':
        data_mat = data_mat.iloc[:, 1:]

    print("\n--- Input Data ---")
    print(f"data_mat shape: {data_mat.shape} (cells × PCs)")
    print(f"meta_data shape: {meta_data.shape}")
    print(f"meta_data columns: {list(meta_data.columns)}")
    print(f"Batch variable '{batch_var}' unique values: {meta_data[batch_var].unique()}")
    print(f"Cells per {batch_var}:\n{meta_data[batch_var].value_counts()}")

    print("\n--- Running Harmony ---")
    start = time()
    ho = hm.run_harmony(data_mat, meta_data, [batch_var])
    end = time()
    print(f"\n✓ Harmony completed in {end - start:.2f} seconds")

    print("\n--- Harmony Object Info ---")
    print(f"Number of clusters (K): {ho.K}")
    print(f"Number of harmony iterations: {len(ho.objective_harmony)}")
    print(f"K-means rounds per iteration: {ho.kmeans_rounds}")
    print(f"Z_corr shape: {ho.Z_corr.shape} (PCs × cells)")
    print(f"Z_orig shape: {ho.Z_orig.shape}")

    # Check convergence
    print("\n--- Convergence ---")
    print(f"Objective (harmony) history: {[f'{x:.2f}' for x in ho.objective_harmony]}")

    res = pd.DataFrame(ho.Z_corr).T
    res.columns = ['PC{}'.format(i + 1) for i in range(res.shape[1])]
    # res.to_csv("data/pbmc_3500_pcs_harmonized_python.tsv.gz", sep = "\t", index = False)

    # Compare to expected results from R
    harm = pd.read_csv(harmonized_tsv, sep="\t")
    if harm.iloc[:,0].dtype == 'object':
        harm = harm.iloc[:, 1:]
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
                            random_state=random_state)
        return ho.Z_corr

    # Assert same results when random_state is set.
    print("\n--- Testing reproducibility with random_state=42 ---")
    result1 = run(42)
    result2 = run(42)
    diff_same_seed = np.abs(result1 - result2).sum()
    print(f"Difference between two runs with same seed: {diff_same_seed:.6f}")
    np.testing.assert_allclose(result1, result2)
    print("✓ Same seed produces identical results (PASSED)")

    # Assert different values when random_state is None. Absolute differences
    # in multiple runs are usually > 2000
    print("\n--- Testing variability with random_state=None ---")
    result3 = run(None)
    result4 = run(None)
    diff_no_seed = np.abs(result3 - result4).sum()
    print(f"Difference between two runs without seed: {diff_no_seed:.2f}")
    assert diff_no_seed > 1000, f"Expected diff > 1000, got {diff_no_seed}"
    print("✓ No seed produces different results (PASSED)")


def test_cluster_fn():
    print("\n" + "=" * 60)
    print("TEST: test_cluster_fn")
    print("=" * 60)

    meta_data = pd.read_csv("data/pbmc_3500_meta.tsv.gz", sep="\t")
    data_mat = pd.read_csv("data/pbmc_3500_pcs.tsv.gz", sep="\t")

    if sys.version_info.major == 3 and sys.version_info.minor == 6:
        print("⚠ Skipping test on Python 3.6")
        return

    print(f"\nPython version: {sys.version_info.major}.{sys.version_info.minor}")

    def cluster_fn(data, K):
        print(f"  Custom cluster_fn called: data shape={data.shape}, K={K}")
        centroid, label = kmeans2(data, K, minit='++', seed=0)
        print(f"  Returning centroids shape: {centroid.shape}")
        return centroid

    def run(cluster_fn):
        ho = hm.run_harmony(data_mat,
                            meta_data, ['donor'],
                            max_iter_harmony=2,
                            max_iter_kmeans=2,
                            cluster_fn=cluster_fn)
        return ho.Z_corr

    # Assert same results when random_state is set.
    print("\n--- Testing custom cluster function reproducibility ---")
    result1 = run(cluster_fn)
    print("  First run complete")
    result2 = run(cluster_fn)
    print("  Second run complete")
    
    diff = np.abs(result1 - result2).sum()
    print(f"\nDifference between runs: {diff:.6f}")
    np.testing.assert_equal(result1, result2)
    print("✓ Custom cluster function produces identical results (PASSED)")


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("# Running harmonypy tests")
    print("#" * 60)
    
    test_run_harmony(
        meta_tsv="data/pbmc_3500_meta.tsv.gz",
        pcs_tsv="data/pbmc_3500_pcs.tsv.gz",
        harmonized_tsv="data/pbmc_3500_pcs_harmonized.tsv.gz",
        batch_var="donor"
    )
    test_run_harmony(
        meta_tsv="data/ircolitis_blood_cd8_obs.tsv.gz",
        pcs_tsv="data/ircolitis_blood_cd8_pcs.tsv.gz",
        harmonized_tsv="data/ircolitis_blood_cd8_pcs_harmonized.tsv.gz",
        batch_var="batch"
    )
    test_random_seed()
    test_cluster_fn()
    
    print("\n" + "#" * 60)
    print("# All tests passed!")
    print("#" * 60)

