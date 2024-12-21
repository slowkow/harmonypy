# LISI - The Local Inverse Simpson Index
# Copyright (C) 2018  Ilya Korsunsky
#               2019  Kamil Slowikowski <kslowikowski@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
GPU = False
try:
    if os.environ.get('HARMONYPY_CPU', '0') == '1':
        raise ModuleNotFoundError("HARMONYPY_CPU is set to 1")

    import cudf as _pd
    import cupy as _np
    from cudf.core.dtypes import CategoricalDtype as _Categorical
    from cuml.neighbors import NearestNeighbors as _NearestNeighbors
    GPU = True
except ModuleNotFoundError:
    import pandas as _pd
    import numpy as _np
    from pandas import Categorical as _Categorical
    from sklearn.neighbors import NearestNeighbors as _NearestNeighbors

from typing import Iterable


def compute_lisi(
    X: _np.array,
    metadata: _pd.DataFrame,
    label_colnames: Iterable[str],
    perplexity: float=30
):
    """Compute the Local Inverse Simpson Index (LISI) for each column in metadata.

    LISI is a statistic computed for each item (row) in the data matrix X.

    The following example may help to interpret the LISI values.

    Suppose one of the columns in metadata is a categorical variable with 3 categories.

        - If LISI is approximately equal to 3 for an item in the data matrix,
          that means that the item is surrounded by neighbors from all 3
          categories.

        - If LISI is approximately equal to 1, then the item is surrounded by
          neighbors from 1 category.

    The LISI statistic is useful to evaluate whether multiple datasets are
    well-integrated by algorithms such as Harmony [1].

    [1]: Korsunsky et al. 2019 doi: 10.1038/s41592-019-0619-0
    """
    n_cells = metadata.shape[0]
    n_labels = len(label_colnames)
    # We need at least 3 * n_neigbhors to compute the perplexity
    algo = 'auto' if GPU else 'kd_tree'
    knn = _NearestNeighbors(n_neighbors = perplexity * 3, algorithm = algo).fit(X)
    distances, indices = knn.kneighbors(X)
    # Don't count yourself
    if GPU:
        indices = indices.iloc[:,1:]
        distances = distances.iloc[:,1:]
    else:
        indices = indices[:,1:]
        distances = distances[:,1:]

    # Save the result
    lisi_df = _np.zeros((n_cells, n_labels))
    for i, label in enumerate(label_colnames):
        labels = _Categorical(metadata[label])
        n_categories = len(labels.categories)
        simpson = compute_simpson(distances.T, indices.T, labels, n_categories, perplexity)
        lisi_df[:,i] = 1 / simpson
    return lisi_df


def compute_simpson(
    distances: _np.ndarray,
    indices: _np.ndarray,
    labels: _Categorical,
    n_categories: int,
    perplexity: float,
    tol: float=1e-5
):
    n = distances.shape[1]
    P = _np.zeros(distances.shape[0])
    simpson = _np.zeros(n)
    logU = _np.log(perplexity)
    # Loop through each cell.
    for i in range(n):
        beta = 1
        betamin = -_np.inf
        betamax = _np.inf
        # Compute Hdiff
        if GPU:
            P = _np.exp(_np.array(-distances.iloc[:,i] * beta))
        else:
            P = _np.exp(_np.array(-distances[:,i] * beta))
        P_sum = _np.sum(P)
        if P_sum == 0:
            H = 0
            P = _np.zeros(distances.shape[0])
        else:
            if GPU:
                H = _np.log(P_sum) + beta * _np.sum(_np.array(distances.iloc[:,i]) * P) / P_sum
            else:
                H = _np.log(P_sum) + beta * _np.sum(_np.array(distances[:,i]) * P) / P_sum
            P = P / P_sum
        Hdiff = H - logU
        n_tries = 50
        for t in range(n_tries):
            # Stop when we reach the tolerance
            if abs(Hdiff) < tol:
                break
            # Update beta
            if Hdiff > 0:
                betamin = beta
                if not _np.isfinite(betamax):
                    beta *= 2
                else:
                    beta = (beta + betamax) / 2
            else:
                betamax = beta
                if not _np.isfinite(betamin):
                    beta /= 2
                else:
                    beta = (beta + betamin) / 2
            # Compute Hdiff
            if GPU:
                P = _np.exp(_np.array(-distances.iloc[:,i] * beta))
            else:
                P = _np.exp(_np.array(-distances[:,i] * beta))
            P_sum = _np.sum(P)
            if P_sum == 0:
                H = 0
                P = _np.zeros(distances.shape[0])
            else:
                if GPU:
                    H = _np.log(P_sum) + beta * _np.sum(_np.array(distances.iloc[:,i]) * P) / P_sum
                else:
                    H = _np.log(P_sum) + beta * _np.sum(_np.array(distances[:,i]) * P) / P_sum
                P = P / P_sum
            Hdiff = H - logU
        # distancesefault value
        if H == 0:
            simpson[i] = -1
        # Simpson's index
        if GPU:
            labels_alias = labels.categories
            categories = labels_alias.unique().to_pandas()
        else:
            categories = labels.categories
            labels_alias = labels

        for label_category in categories:
            if GPU:
                ix = indices.iloc[:,i]
            else:
                ix = indices[:,i]
            q = labels_alias[ix] == label_category
            if _np.any(q):
                P_sum = _np.sum(P[q])
                simpson[i] += P_sum * P_sum
    return simpson

