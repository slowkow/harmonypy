# Test harmonypy with NumPy implementation
from time import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import harmonypy as hm


def test_run_harmony(meta_tsv, pcs_tsv, harmonized_tsv, batch_var):
    print("\n" + "=" * 60)
    print("TEST: test_run_harmony (NumPy)")
    print("=" * 60)

    # Load input data
    meta_data = pd.read_csv(meta_tsv, sep="\t", low_memory=False)
    data_mat = pd.read_csv(pcs_tsv, sep="\t", low_memory=False)

    if data_mat.iloc[:,0].dtype == 'object':
        data_mat = data_mat.iloc[:, 1:]

    print("\n--- Input Data ---")
    print(f"data_mat shape: {data_mat.shape} (cells × PCs)")
    print(f"meta_data shape: {meta_data.shape}")
    print(f"meta_data columns: {list(meta_data.columns)[:10]}...")
    print(f"Batch variable '{batch_var}' unique values: {meta_data[batch_var].unique()}")
    print(f"Cells per {batch_var}:\n{meta_data[batch_var].value_counts()}")

    print("\n--- Running Harmony (NumPy) ---")
    start = time()
    ho = hm.run_harmony(data_mat, meta_data, [batch_var])
    end = time()
    print(f"\n✓ Harmony completed in {end - start:.2f} seconds")

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
    
    return end - start


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
    print("\n--- Testing reproducibility with random_state=42 ---")
    result1 = run(42)
    result2 = run(42)
    diff_same_seed = np.abs(result1 - result2).sum()
    print(f"Difference between two runs with same seed: {diff_same_seed:.6f}")
    np.testing.assert_allclose(result1, result2)
    print("✓ Same seed produces identical results (PASSED)")

    # Assert different values when random_state is different
    print("\n--- Testing variability with different seeds ---")
    result3 = run(123)
    result4 = run(456)
    diff_diff_seed = np.abs(result3 - result4).sum()
    print(f"Difference between runs with different seeds: {diff_diff_seed:.2f}")
    assert diff_diff_seed > 1000, f"Expected diff > 1000, got {diff_diff_seed}"
    print("✓ Different seeds produce different results (PASSED)")


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("# Running harmonypy tests (NumPy implementation)")
    print("#" * 60)
    
    timings = {}
    
    timings['small'] = test_run_harmony(
        meta_tsv="data/pbmc_3500_meta.tsv.gz",
        pcs_tsv="data/pbmc_3500_pcs.tsv.gz",
        harmonized_tsv="data/pbmc_3500_pcs_harmonized.tsv.gz",
        batch_var="donor"
    )
    
    timings['medium'] = test_run_harmony(
        meta_tsv="data/ircolitis_blood_cd8_obs.tsv.gz",
        pcs_tsv="data/ircolitis_blood_cd8_pcs.tsv.gz",
        harmonized_tsv="data/ircolitis_blood_cd8_pcs_harmonized.tsv.gz",
        batch_var="batch"
    )
    
    timings['large'] = test_run_harmony(
        meta_tsv="data/acute_myeloid_obs.tsv.gz",
        pcs_tsv="data/acute_myeloid_pcs.tsv.gz",
        harmonized_tsv="data/acute_myeloid_pcs_harmonized.tsv.gz",
        batch_var="batch"
    )
    
    test_random_seed()
    
    print("\n" + "#" * 60)
    print("# Performance Summary")
    print("#" * 60)
    print(f"  Small (3.5k cells):    {timings['small']:.2f}s")
    print(f"  Medium (69k cells):    {timings['medium']:.2f}s")
    print(f"  Large (858k cells):    {timings['large']:.2f}s")
    
    print("\n" + "#" * 60)
    print("# All tests passed!")
    print("#" * 60)
