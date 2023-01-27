# vprof -c p tests/test_harmony.py

import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr
import harmonypy as hm
from time import time


def test_run_harmony():

    meta_data = pd.read_csv("data/pbmc_3500_meta.tsv.gz", sep = "\t")
    data_mat = pd.read_csv("data/pbmc_3500_pcs.tsv.gz", sep = "\t")

    start = time()
    ho = hm.run_harmony(data_mat, meta_data, ['donor'])
    end = time()
    print("{:.2f} seconds elapsed".format(end - start))

    res = pd.DataFrame(ho.Z_corr).T
    res.columns = ['PC{}'.format(i + 1) for i in range(res.shape[1])]
    # res.to_csv("data/pbmc_3500_pcs_harmonized_python.tsv.gz", sep = "\t", index = False)

    harm = pd.read_csv("data/pbmc_3500_pcs_harmonized.tsv.gz", sep = "\t")

    cors = []
    for i in range(res.shape[1]):
        cors.append(pearsonr(res.iloc[:,i].values, harm.iloc[:,i].values))
    print([np.round(x[0], 3) for x in cors])

    # Correlation between test PCs and observed PCs is high
    assert np.all(np.array([x[0] for x in cors]) >= 0.9)

def test_random_seed():
    meta_data = pd.read_csv("data/pbmc_3500_meta.tsv.gz", sep = "\t")
    data_mat = pd.read_csv("data/pbmc_3500_pcs.tsv.gz", sep = "\t")

    def run(random_state):
        ho = hm.run_harmony(data_mat, meta_data, ['donor'],
                            max_iter_harmony = 2,
                            max_iter_kmeans = 2,
                            random_state=random_state
        )
        return ho.Z_corr

    # Assert same results when random_state is set.
    np.testing.assert_allclose(run(42), run(42))

    # Assert different values when random_state is None. Absolute differences
    # in multiple runs are usually > 2000
    assert np.abs(run(None) - run(None)).sum() > 1000
