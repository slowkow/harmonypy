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

import numpy as np
from typing import Iterable
from harmonypy._harmony_cpp import compute_lisi_cpp


def compute_lisi(
    X: np.ndarray,
    metadata,
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
    n_cells = np.asarray(X).shape[0]
    n_labels = len(label_colnames)
    X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    lisi_df = np.zeros((n_cells, n_labels))
    for i, label in enumerate(label_colnames):
        col = np.asarray(metadata[label])
        uniques, codes = np.unique(col, return_inverse=True)
        n_categories = len(uniques)
        lisi_df[:, i] = compute_lisi_cpp(X_arr, codes.astype(np.int32), n_categories, perplexity)
    return lisi_df
