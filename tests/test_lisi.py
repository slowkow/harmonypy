import harmonypy as hm
import pandas as pd
import numpy as np
import pytest

def test_lisi():

    X = pd.read_csv("data/lisi_x.tsv.gz", sep = "\t")
    metadata = pd.read_csv("data/lisi_metadata.tsv.gz", sep = "\t")
    label_colnames = metadata.columns
    perplexity = 30

    lisi = hm.compute_lisi(X, metadata, label_colnames, perplexity)

    lisi_test = pd.read_csv("data/lisi_lisi.tsv.gz", sep="\t")
    lisi_test = lisi_test.iloc[:,-2:].to_numpy()

    # Tolerance accounts for differences between kd-tree implementations
    assert np.allclose(lisi, lisi_test, atol=0.01)

    # Batched metadata must retain the values and order from separate calls,
    # including a repeated column and columns with different category counts.
    columns = [label_colnames[1], label_colnames[0], label_colnames[1]]
    batched = hm.compute_lisi(X, metadata, columns, perplexity)
    separate = np.column_stack([
        hm.compute_lisi(X, metadata, [column], perplexity)[:, 0]
        for column in columns
    ])
    np.testing.assert_array_equal(batched, separate)


def test_lisi_native_label_validation():
    from harmonypy._harmony_cpp import compute_lisi_columns_cpp

    X = np.array([[0., 1.], [2., 3.]])
    labels = np.array([[0, 1]], dtype=np.int32)
    counts = np.array([2], dtype=np.int32)
    for bad_labels, bad_counts in [
        (labels[:, :1].copy(), counts),
        (labels, np.array([], dtype=np.int32)),
        (np.array([[-1, 1]], dtype=np.int32), counts),
        (np.array([[0, 2]], dtype=np.int32), counts),
        (labels, np.array([0], dtype=np.int32)),
    ]:
        with pytest.raises(ValueError):
            compute_lisi_columns_cpp(X, bad_labels, bad_counts, 1.)
    for perplexity in (0., np.inf, np.iinfo(np.int32).max / 3):
        with pytest.raises(ValueError):
            compute_lisi_columns_cpp(X, labels, counts, perplexity)

# def timereps(reps, func):
#     from time import time
#     start = time()
#     for i in range(0, reps):
#         func()
#     end = time()
#     return (end - start) / reps
# 
# # 0.3 seconds per loop (too slow)
# timereps(10, lambda: hm.compute_lisi(X, metadata, label_colnames, perplexity))
# 
# 
# # Try https://github.com/rkern/line_profiler
