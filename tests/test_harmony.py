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
    res.to_csv("data/pbmc_3500_pcs_harmonized_python.tsv.gz", sep = "\t", index = False)

    harm = pd.read_csv("data/pbmc_3500_pcs_harmonized.tsv.gz", sep = "\t")

    cors = []
    for i in range(res.shape[1]):
        cors.append(pearsonr(res.iloc[:,i].values, harm.iloc[:,i].values))
    print([np.round(x[0], 3) for x in cors])

    # Correlation between test PCs and observed PCs is high
    assert np.all(np.array([x[0] for x in cors]) >= 0.9)

