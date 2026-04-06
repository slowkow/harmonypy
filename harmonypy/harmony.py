# harmonypy - A data alignment algorithm.
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
import pandas as pd
import numpy as np
from harmonypy._harmony_cpp import HarmonyCpp

# Prevent OpenMP runtime conflicts on macOS between openblas (via Armadillo)
# and other libraries. Setting threads to 1 avoids the conflict.
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"
if "OPENBLAS_NUM_THREADS" not in os.environ:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
import logging

# create logger
logger = logging.getLogger('harmonypy')
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def run_harmony(
    data_mat: np.ndarray,
    meta_data: pd.DataFrame,
    vars_use,
    theta=None,
    lamb=None,
    sigma=0.1,
    nclust=None,
    tau=0,
    block_size=0.05,
    max_iter_harmony=10,
    max_iter_kmeans=4,
    epsilon_cluster=1e-3,
    epsilon_harmony=1e-2,
    alpha=0.2,
    batch_prop_cutoff=1e-5,
    verbose=True,
    random_state=0,
):
    """Run Harmony batch effect correction.

    Parameters
    ----------
    data_mat : np.ndarray
        PCA embedding matrix (cells x PCs or PCs x cells)
    meta_data : pd.DataFrame
        Metadata with batch variables (cells x variables)
    vars_use : str or list
        Column name(s) in meta_data to use for batch correction
    theta : float or list, optional
        Diversity penalty parameter(s). Default is 2 for each batch.
    lamb : float or list, optional
        Ridge regression penalty. Default is None (auto-estimation).
        Set to a positive value for fixed lambda.
    sigma : float, optional
        Kernel bandwidth for soft clustering. Default is 0.1.
    nclust : int, optional
        Number of clusters. Default is min(N/30, 100).
    tau : float, optional
        Protection against overcorrection. Default is 0.
    block_size : float, optional
        Proportion of cells to update in each block. Default is 0.05.
    max_iter_harmony : int, optional
        Maximum Harmony iterations. Default is 10.
    max_iter_kmeans : int, optional
        Maximum k-means iterations per Harmony iteration. Default is 4.
    epsilon_cluster : float, optional
        K-means convergence threshold. Default is 1e-3.
    epsilon_harmony : float, optional
        Harmony convergence threshold. Default is 1e-2.
    alpha : float, optional
        Alpha parameter for lambda estimation. Default is 0.2.
    batch_prop_cutoff : float, optional
        Minimum batch proportion in a cluster for correction. Default is 1e-5.
    verbose : bool, optional
        Print progress messages. Default is True.
    random_state : int, optional
        Random seed for reproducibility. Default is 0.

    Returns
    -------
    Harmony
        Harmony object with corrected data in Z_corr attribute.
    """
    N = meta_data.shape[0]
    if data_mat.shape[1] != N:
        data_mat = data_mat.T

    assert data_mat.shape[1] == N, \
       "data_mat and meta_data do not have the same number of cells"

    if nclust is None:
        nclust = int(min(round(N / 30.0), 100))

    if isinstance(sigma, float) and nclust > 1:
        sigma = np.repeat(sigma, nclust)

    if isinstance(vars_use, str):
        vars_use = [vars_use]

    # Build compact batch-of-cell index (n_covariates x N, int64)
    # instead of dense B x N one-hot matrix — O(N) vs O(B*N) memory
    batch_of_cell = np.empty((len(vars_use), N), dtype=np.int64)
    phi_n = np.empty(len(vars_use), dtype=int)
    offset = 0
    for c, var in enumerate(vars_use):
        codes, uniques = pd.factorize(meta_data[var])
        n_levels = len(uniques)
        batch_of_cell[c] = codes + offset
        phi_n[c] = n_levels
        offset += n_levels

    # Theta handling - default is 2 (matches R package)
    if theta is None:
        theta = np.repeat([2] * len(phi_n), phi_n).astype(np.float32)
    elif isinstance(theta, (float, int)):
        theta = np.repeat([theta] * len(phi_n), phi_n).astype(np.float32)
    elif len(theta) == len(phi_n):
        theta = np.repeat([theta], phi_n).astype(np.float32)
    else:
        theta = np.asarray(theta, dtype=np.float32)

    assert len(theta) == np.sum(phi_n), \
        "each batch variable must have a theta"

    # Lambda handling (matches R harmony2: NULL = auto-estimation)
    lambda_estimation = False
    if lamb is None or lamb == -1:
        lambda_estimation = True
        lamb = np.zeros(1, dtype=np.float32)
    elif isinstance(lamb, (float, int)):
        lamb = np.repeat([lamb] * len(phi_n), phi_n).astype(np.float32)
        lamb = np.insert(lamb, 0, 0).astype(np.float32)
    elif len(lamb) == len(phi_n):
        lamb = np.repeat([lamb], phi_n).astype(np.float32)
        lamb = np.insert(lamb, 0, 0).astype(np.float32)
    else:
        lamb = np.asarray(lamb, dtype=np.float32)
        if len(lamb) == np.sum(phi_n):
            lamb = np.insert(lamb, 0, 0).astype(np.float32)

    # Number of items in each category
    B = int(np.sum(phi_n))
    N_b = np.bincount(batch_of_cell.ravel(), minlength=B).astype(np.float32)
    Pr_b = (N_b / N).astype(np.float32)

    if tau > 0:
        theta = theta * (1 - np.exp(-(N_b / (nclust * tau)) ** 2))

    if verbose:
        logger.info("Running Harmony")
        logger.info("  Parameters:")
        logger.info(f"    max_iter_harmony: {max_iter_harmony}")
        logger.info(f"    max_iter_kmeans: {max_iter_kmeans}")
        logger.info(f"    epsilon_cluster: {epsilon_cluster}")
        logger.info(f"    epsilon_harmony: {epsilon_harmony}")
        logger.info(f"    nclust: {nclust}")
        logger.info(f"    block_size: {block_size}")
        if lambda_estimation:
            logger.info(f"    lamb: dynamic (alpha={alpha})")
        else:
            logger.info(f"    lamb: {lamb[1:]}")
        logger.info(f"    theta: {theta}")
        logger.info(f"    sigma: {sigma[:5]}..." if len(sigma) > 5 else f"    sigma: {sigma}")
        logger.info(f"    verbose: {verbose}")
        logger.info(f"    random_state: {random_state}")
        logger.info(f"  Data: {data_mat.shape[0]} PCs × {N} cells")
        logger.info(f"  Batch variables: {vars_use}")

    # Ensure data_mat is a proper numpy array
    if hasattr(data_mat, 'values'):
        data_mat = data_mat.values

    # Prepare arrays for C++ backend
    data_f64 = np.ascontiguousarray(data_mat.astype(np.float64))
    batch_of_cell_c = np.ascontiguousarray(batch_of_cell)

    # Signal lambda estimation with sentinel [-1]
    if lambda_estimation:
        lamb_cpp = np.array([-1.0], dtype=np.float64)
    else:
        lamb_cpp = lamb.astype(np.float64)

    cpp_harmony = HarmonyCpp(
        data_f64,
        batch_of_cell_c,
        Pr_b.astype(np.float64),
        sigma.astype(np.float64),
        theta.astype(np.float64),
        lamb_cpp,
        float(alpha),
        max_iter_harmony,
        max_iter_kmeans,
        float(epsilon_cluster),
        float(epsilon_harmony),
        nclust,
        float(block_size),
        phi_n.tolist(),
        float(batch_prop_cutoff),
        verbose,
        random_state if random_state is not None else 0
    )
    return Harmony(cpp_harmony)


class Harmony:
    """Harmony result object.

    Attributes
    ----------
    Z_corr : np.ndarray
        Corrected embedding matrix (N x d).
    Z_orig : np.ndarray
        Original embedding matrix (N x d).
    R : np.ndarray
        Soft cluster assignment matrix (N x K).
    Y : np.ndarray
        Cluster centroids matrix (d x K).
    K : int
        Number of clusters.
    objective_harmony : list
        Harmony objective values per iteration.
    objective_kmeans : list
        K-means objective values.
    kmeans_rounds : list
        Number of k-means rounds per harmony iteration.
    """

    def __init__(self, cpp_harmony):
        self._cpp = cpp_harmony

    @property
    def Z_corr(self):
        """Corrected embedding matrix (N x d)."""
        return self._cpp.Z_corr.T

    @property
    def Z_orig(self):
        """Original embedding matrix (N x d)."""
        return self._cpp.Z_orig.T

    @property
    def Z_cos(self):
        """L2-normalized embedding matrix (N x d)."""
        return self._cpp.Z_cos.T

    @property
    def R(self):
        """Soft cluster assignment matrix (N x K)."""
        return self._cpp.R.T

    @property
    def Y(self):
        """Cluster centroids matrix (d x K)."""
        return self._cpp.Y

    @property
    def K(self):
        """Number of clusters."""
        return self._cpp.K

    @property
    def objective_harmony(self):
        """Harmony objective values per iteration."""
        return self._cpp.objective_harmony

    @property
    def objective_kmeans(self):
        """K-means objective values."""
        return self._cpp.objective_kmeans

    @property
    def kmeans_rounds(self):
        """Number of k-means rounds per harmony iteration."""
        return self._cpp.kmeans_rounds

    def result(self):
        """Return corrected data as NumPy array."""
        return self._cpp.Z_corr.T
