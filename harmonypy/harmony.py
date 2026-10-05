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

import numpy as np
from harmonypy._harmony_cpp import HarmonyCpp
from collections import defaultdict
import itertools
import logging
import operator
import os
import sys

# create logger
logger = logging.getLogger('harmonypy')
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def _available_cores():
    """Number of CPUs this process may run on.

    Respects CPU affinity (e.g. Slurm, taskset) where the platform reports it.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _physical_cores():
    """Number of physical cores this process may run on, or None if unknown.

    The hardware threads of one core (hyperthreads) count once. On Linux,
    these are the cores of the CPUs in the process's affinity mask (e.g.
    Slurm, taskset); on macOS, all cores.
    """
    if hasattr(os, "sched_getaffinity"):
        cores = set()
        try:
            for cpu in os.sched_getaffinity(0):
                with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list", encoding="ascii") as f:
                    cores.add(f.read().strip())
        except OSError:
            return None
        return len(cores) or None
    if sys.platform == "darwin":
        import ctypes
        n, size = ctypes.c_int(0), ctypes.c_size_t(ctypes.sizeof(ctypes.c_int))
        status = ctypes.CDLL(None).sysctlbyname(
            b"hw.physicalcpu", ctypes.byref(n), ctypes.byref(size), None, ctypes.c_size_t(0)
        )
        return n.value if status == 0 and n.value > 0 else None
    return None


def _missing(label):
    """Whether label is None, NaN, NaT or pandas.NA."""
    try:
        return label is None or bool(label != label)
    except TypeError:  # pandas.NA cannot be converted to bool
        return True


def _factorize(labels, name="labels"):
    """Number the distinct labels in sorted order.

    Returns (codes, n): codes[i] is the rank of labels[i] among the n distinct
    labels, as np.unique(labels, return_inverse=True) gives. np.unique sorts
    every label, which is slow for strings (7-23 s for 16 million), so labels
    other than numbers are numbered in one pass through a hash table, and then
    only the n distinct labels are sorted, as pandas.factorize(sort=True)
    does. A pandas Categorical's codes are renumbered without reading the
    labels. Missing labels (None, NaN, NaT or pandas.NA) raise ValueError,
    naming the labels as name.
    """
    missing = ValueError(f"{name} has missing labels (None, NaN or NaT); label those cells or remove them")
    categorical = getattr(labels, "cat", labels)  # a pandas Series's categorical values
    codes, categories = getattr(categorical, "codes", None), getattr(categorical, "categories", None)
    if codes is not None and categories is not None:
        codes = np.asarray(codes)
        if codes.size and codes.min() < 0:
            raise missing
        if codes.size:
            used = np.flatnonzero(np.bincount(codes, minlength=len(categories)))
            uniques, inverse = np.unique(np.asarray(categories)[used], return_inverse=True)
            rank = np.empty(len(categories), dtype=np.intp)
            rank[used] = inverse.reshape(-1)
            return rank[codes], len(uniques)
    labels = np.asarray(labels).reshape(-1)
    kind = labels.dtype.kind
    if (kind in "fc" and np.isnan(labels).any()) or (kind in "mM" and np.isnat(labels).any()):
        raise missing
    if kind not in "biufcmM":  # np.unique is fast for numbers
        try:
            # Number the labels in the order they first appear (in C: dict
            # lookups through map).
            first_seen = defaultdict(itertools.count().__next__)
            codes = np.fromiter(map(first_seen.__getitem__, labels.tolist()), dtype=np.intp, count=labels.size)
            distinct = list(first_seen)
        except TypeError:  # labels that cannot be hashed
            distinct = None
        if distinct is not None:
            if any(map(_missing, distinct)):
                raise missing
            try:
                # Renumber them in sorted order. np.unique merges equal labels
                # that hash differently (np.float32(0.1) and 0.1), so leave
                # those to it.
                order = sorted(range(len(distinct)), key=distinct.__getitem__)
                ordered = [distinct[i] for i in order]
                if all(map(operator.lt, ordered, ordered[1:])):
                    rank = np.empty(len(distinct), dtype=np.intp)
                    rank[order] = np.arange(len(distinct))
                    return rank[codes], len(distinct)
            except (TypeError, ValueError, OverflowError):  # labels that cannot be sorted
                pass
    uniques, codes = np.unique(labels, return_inverse=True)
    return codes.reshape(-1), len(uniques)


def _numbers(values, name):
    """values (a number, sequence, NumPy array or NumPy scalar) as a float32 array.

    An (n, 1) or (1, n) array is read as a sequence of n numbers. Anything that
    is not real numbers raises ValueError.
    """
    try:
        array = np.asarray(values)
        if array.dtype.kind == "O":
            array = array.astype(np.float64)
    except (TypeError, ValueError) as err:
        raise ValueError(f"{name} must be a number or a sequence of numbers") from err
    if array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be a number or a sequence of numbers, not {array.dtype}")
    if array.ndim > 1 and array.size == max(array.shape):
        array = array.reshape(-1)
    return array.astype(np.float32)


def _per_batch(values, phi_n, name, other=None):
    """Expand a parameter to one float32 value per batch.

    values is one number for all batches, one per batch variable, or one per
    batch. phi_n holds the number of batches of each variable. other describes
    any other accepted form, for the error message.
    """
    values = _numbers(values, name)
    if values.ndim == 0:
        return np.repeat(values, np.sum(phi_n))
    if values.ndim == 1 and len(values) == len(phi_n):
        return np.repeat(values, phi_n)
    if values.ndim == 1 and len(values) == np.sum(phi_n):
        return values
    forms = ["one number", f"one per batch variable ({len(phi_n)})", f"one per batch ({np.sum(phi_n)})"]
    forms += [other] if other else []
    raise ValueError(f"{name} must be {', '.join(forms[:-1])}, or {forms[-1]}; got shape {values.shape}")


def run_harmony(
    data_mat: np.ndarray,
    meta_data,
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
    ncores=0,
):
    """Run Harmony batch effect correction.

    Parameters
    ----------
    data_mat : np.ndarray
        PCA embedding matrix (cells x PCs or PCs x cells)
    meta_data : dict-like
        Metadata with batch variables (cells x variables).
        Accepts pandas DataFrame, dict of arrays, or any object
        supporting ``meta_data[var]`` column access and ``.shape[0]``.
    vars_use : str or list
        Column name(s) in meta_data to use for batch correction
    theta : float or array-like, optional
        Diversity penalty: one number for all batches, one per batch
        variable (in the order of vars_use), or one per batch (each
        variable's levels in sorted order). Default is 2.
    lamb : float or array-like, optional
        Ridge regression penalty, given like theta, with positive values; one
        per batch may also be preceded by the intercept's penalty (otherwise
        0). Default is None, which estimates it in each cluster from alpha;
        -1 does the same.
    sigma : float or array-like, optional
        Kernel bandwidth for soft clustering, one value for all clusters or
        one per cluster. Default is 0.1.
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
    ncores : int, optional
        Number of threads, at most the number of CPUs this process may use
        (its CPU affinity, where the platform reports it). Default is 0,
        which uses one thread per physical core of those CPUs, since a
        second thread per core (hyperthreading) adds little speed. In a
        container limited by a CPU quota, set it explicitly. Results are the
        same for any value.

    Returns
    -------
    Harmony
        Harmony object with corrected data in Z_corr attribute.
    """
    # Ensure data_mat is a proper numpy array (drop non-numeric columns)
    if hasattr(data_mat, 'select_dtypes'):
        data_mat = data_mat.select_dtypes(include=[np.number]).values
    elif hasattr(data_mat, 'values'):
        data_mat = data_mat.values

    if hasattr(meta_data, 'shape'):
        N = meta_data.shape[0]
    else:
        # dict-like: get length from first column
        first_key = vars_use[0] if isinstance(vars_use, list) else vars_use
        N = len(meta_data[first_key])
    if data_mat.shape[1] == N:
        pass
    elif data_mat.shape[0] == N:
        data_mat = data_mat.T
    else:
        raise ValueError(
            f"data_mat has shape {data_mat.shape}, but meta_data has {N} cells. "
            f"One dimension of data_mat must match the number of cells."
        )

    if nclust is None:
        nclust = int(min(round(N / 30.0), 100))

    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.ndim == 0:
        sigma = np.repeat(sigma, nclust)

    if isinstance(vars_use, str):
        vars_use = [vars_use]

    # Build compact batch-of-cell index (n_covariates x N, int64).
    # Passed directly to C++ — no sparse Phi construction needed.
    batch_of_cell = np.empty((len(vars_use), N), dtype=np.int64)
    phi_n = np.empty(len(vars_use), dtype=int)
    offset = 0
    for c, var in enumerate(vars_use):
        codes, n_levels = _factorize(meta_data[var], f"meta_data[{var!r}]")
        batch_of_cell[c] = codes + offset
        phi_n[c] = n_levels
        offset += n_levels

    # Theta handling - default is 2 (matches R package)
    theta = _per_batch(2.0 if theta is None else theta, phi_n, "theta")

    # Lambda handling (matches R harmony2: NULL = auto-estimation). A single
    # -1, also in a one-element list or array, selects estimation too.
    lamb = None if lamb is None else _numbers(lamb, "lamb")
    lambda_estimation = lamb is None or (lamb.size == 1 and lamb.reshape(-1)[0] == -1)
    if lambda_estimation:
        lamb = np.zeros(1, dtype=np.float32)
    else:
        # One value per batch, with the intercept's penalty (0) in front, or
        # that full vector as given.
        if lamb.ndim != 1 or len(lamb) != np.sum(phi_n) + 1:
            other = f"the intercept's penalty and then one per batch ({np.sum(phi_n) + 1})"
            lamb = np.insert(_per_batch(lamb, phi_n, "lamb", other), 0, 0)
        if not np.all(np.isfinite(lamb)) or np.any(lamb < 0):
            raise ValueError("lamb must be finite and not negative")

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
            intercept = f" (intercept: {lamb[0]})" if lamb[0] else ""
            logger.info(f"    lamb: {lamb[1:]}{intercept}")
        logger.info(f"    theta: {theta}")
        logger.info(f"    sigma: {sigma[:5]}..." if len(sigma) > 5 else f"    sigma: {sigma}")
        logger.info(f"    verbose: {verbose}")
        logger.info(f"    random_state: {random_state}")
        logger.info(f"  Data: {data_mat.shape[0]} PCs × {N} cells")
        logger.info(f"  Batch variables: {vars_use}")

    # Prepare arrays for C++ backend: one row per cell
    data_f64 = np.ascontiguousarray(data_mat.T, dtype=np.float64)
    batch_of_cell_c = np.ascontiguousarray(batch_of_cell)

    # Signal lambda estimation with sentinel [-1]
    if lambda_estimation:
        lamb_cpp = np.array([-1.0], dtype=np.float64)
    else:
        lamb_cpp = lamb.astype(np.float64)

    # ncores threads, or by default one per physical core, at most one per CPU
    cpus = _available_cores()
    n_threads = min(int(ncores) if ncores > 0 else _physical_cores() or cpus, cpus)

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
        random_state if random_state is not None else 0,
        n_threads,
        logger.info,
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
        return self._cpp.Z_corr

    @property
    def Z_orig(self):
        """Original embedding matrix (N x d)."""
        return self._cpp.Z_orig

    @property
    def Z_cos(self):
        """Corrected embedding matrix with each cell scaled to unit length (N x d).

        These are the cosine-normalized coordinates that Harmony clusters on.
        """
        return self._cpp.Z_cos

    @property
    def R(self):
        """Soft cluster assignment matrix (N x K)."""
        return self._cpp.R

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
        return self._cpp.Z_corr
