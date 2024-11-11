# vprof -c p tests/test_harmony.py
from time import time
GPU = False
import numpy
try:
    import cudf as pd
    GPU = True
except ImportError:
    import pandas as pd

try:
    import cupy as np
except ImportError:
    import numpy as np

try:
    from cuml import KMeans
except ImportError:
    pass

from scipy.cluster.vq import kmeans2

try:
    from cuml.metrics import pearsonr
except ImportError:
    from scipy.stats import pearsonr

import sys
import harmonypy as hm


def test_run_harmony():

    meta_data = pd.read_csv("data/pbmc_3500_meta.tsv.gz", sep="\t")
    data_mat = pd.read_csv("data/pbmc_3500_pcs.tsv.gz", sep="\t")

    start = time()
    ho = hm.run_harmony(data_mat, meta_data, ['donor'])
    end = time()
    print("{:.2f} seconds elapsed".format(end - start))

    res = pd.DataFrame(ho.Z_corr).T
    res.columns = ['PC{}'.format(i + 1) for i in range(res.shape[1])]
    # res.to_csv("data/pbmc_3500_pcs_harmonized_python.tsv.gz", sep = "\t", index = False)

    harm = pd.read_csv("data/pbmc_3500_pcs_harmonized.tsv.gz", sep="\t")

    cors = []
    for i in range(res.shape[1]):
        if GPU:
            cors.append(pearsonr(res.iloc[:, i].values.get(), harm.iloc[:, i].values.get()))
        else:
            cors.append(pearsonr(res.iloc[:, i].values, harm.iloc[:, i].values))
    print([np.round(x[0], 3) for x in cors])

    # Correlation between test PCs and observed PCs is high
    assert np.all(np.array([x[0] for x in cors]) >= 0.9)


def test_random_seed():
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
    np.testing.assert_allclose(run(42), run(42))

    # Assert different values when random_state is None. Absolute differences
    # in multiple runs are usually > 2000
    assert np.abs(run(None) - run(None)).sum() > 1000


def test_cluster_fn():
    meta_data = pd.read_csv("data/pbmc_3500_meta.tsv.gz", sep="\t")
    data_mat = pd.read_csv("data/pbmc_3500_pcs.tsv.gz", sep="\t")

    if sys.version_info.major == 3 and sys.version_info.minor == 6:
        return

    def cluster_fn(data, K):

        if GPU:
            kmeans = KMeans(n_clusters=K, init='k-means||')
            kmeans.fit(data)
            centroid = kmeans.cluster_centers_
            label = kmeans.labels_
        else:
            centroid, label = kmeans2(data, K, minit='++', seed=0)
        return centroid

    def run(cluster_fn):
        ho = hm.run_harmony(data_mat,
                            meta_data, ['donor'],
                            max_iter_harmony=2,
                            max_iter_kmeans=2,
                            cluster_fn=cluster_fn)
        return ho.Z_corr

    # Assert same results when random_state is set.
    if GPU:
        pass
        # numpy.testing.assert_equal(run(cluster_fn).get(), run(cluster_fn).get())
    else:
        numpy.testing.assert_equal(run(cluster_fn), run(cluster_fn))

