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
import torch
from sklearn.cluster import KMeans

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


def get_device(device=None):
    """Get the appropriate device for PyTorch operations."""
    if device is not None:
        return torch.device(device)
    
    # Check for available accelerators
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


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
    device=None,
    backend="auto"
):
    """Run Harmony batch effect correction.
    
    This is a PyTorch implementation matching the R package formulas.
    Supports CPU and GPU (CUDA, MPS) acceleration.
    
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
    device : str, optional
        Device to use ('cpu', 'cuda', 'mps'). Default is auto-detect.
        
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

    # Get device
    device_obj = get_device(device)

    if verbose:
        logger.info(f"Running Harmony (PyTorch on {device_obj})")
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

    # Resolve backend
    use_cpp = False
    if backend == "auto":
        try:
            from harmonypy._harmony_cpp import HarmonyCpp
            use_cpp = True
        except ImportError:
            use_cpp = False
    elif backend == "cpp":
        from harmonypy._harmony_cpp import HarmonyCpp
        use_cpp = True

    if use_cpp:
        # C++ backend (Armadillo)
        from harmonypy._harmony_cpp import HarmonyCpp

        if verbose:
            logger.info("Using C++ backend (Armadillo)")

        # Pass batch_of_cell directly — C++ builds sparse Phi, no dense B×N array
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
        return HarmonyCpp_Wrapper(cpp_harmony)

    # PyTorch backend
    data_mat = np.asarray(data_mat, dtype=np.float32)

    # Set random seeds
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    ho = Harmony(
        data_mat, batch_of_cell, phi_n, N_b, Pr_b, sigma.astype(np.float32),
        theta, lamb, alpha, lambda_estimation, batch_prop_cutoff,
        max_iter_harmony, max_iter_kmeans,
        epsilon_cluster, epsilon_harmony, nclust, block_size, verbose,
        random_state, device_obj
    )

    return ho


class HarmonyCpp_Wrapper:
    """Wrapper around C++ HarmonyCpp to provide the same interface as Harmony."""

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


class Harmony:
    """Harmony class for batch effect correction using PyTorch.
    
    Supports CPU and GPU acceleration.
    """
    
    def __init__(
            self, Z, batch_of_cell, B_vec, N_b, Pr_b, sigma, theta, lamb, alpha,
            lambda_estimation, batch_prop_cutoff,
            max_iter_harmony, max_iter_kmeans,
            epsilon_kmeans, epsilon_harmony, K, block_size, verbose,
            random_state, device
    ):
        self.device = device

        # Convert to PyTorch tensors on device
        # Store with underscore prefix internally, expose as properties returning NumPy arrays
        self._Z_corr = torch.tensor(Z, dtype=torch.float32, device=device)
        self._Z_orig = torch.tensor(Z, dtype=torch.float32, device=device)

        # Normalize Z_corr in-place (serves as Z_cos; no separate buffer — matches C++)
        self._Z_corr /= torch.linalg.norm(self._Z_corr, ord=2, dim=0)

        # Compact batch assignment (n_covariates x N, int64) — O(N) memory
        self._batch_of_cell = torch.tensor(batch_of_cell, dtype=torch.int64, device=device)
        self._n_covariates = batch_of_cell.shape[0]
        self._Pr_b = torch.tensor(Pr_b, dtype=torch.float32, device=device)

        self.N = self._Z_corr.shape[1]
        self.B = int(np.sum(B_vec))
        self.d = self._Z_corr.shape[0]

        # Covariate structure
        self.B_vec = B_vec
        self._batch_sizes = torch.tensor(N_b, dtype=torch.float32, device=device)
        self.batch_prop_cutoff = batch_prop_cutoff
        # Covariate bounds: cumulative sums for mapping batch index -> covariate
        self.covariate_bounds = np.cumsum(B_vec)

        # Build batch index for fast ridge correction
        cov_of_batch = np.repeat(np.arange(len(B_vec)), B_vec)
        self._batch_index = []
        for b in range(self.B):
            c = cov_of_batch[b]
            idx = np.where(batch_of_cell[c] == b)[0]
            self._batch_index.append(torch.tensor(idx, dtype=torch.int64, device=device))

        self.window_size = 3
        self.epsilon_kmeans = epsilon_kmeans
        self.epsilon_harmony = epsilon_harmony

        self._lamb = torch.tensor(lamb, dtype=torch.float32, device=device)
        self.alpha = alpha
        self.lambda_estimation = lambda_estimation
        self._sigma = torch.tensor(sigma, dtype=torch.float32, device=device)
        self.block_size = block_size
        self.K = K
        self.max_iter_harmony = max_iter_harmony
        self.max_iter_kmeans = max_iter_kmeans
        self.verbose = verbose
        self._theta = torch.tensor(theta, dtype=torch.float32, device=device)

        self.objective_harmony = []
        self.objective_kmeans = []
        self.objective_kmeans_dist = []
        self.objective_kmeans_entropy = []
        self.objective_kmeans_cross = []
        self.kmeans_rounds = []

        self.allocate_buffers()
        self.init_cluster(random_state)
        self.harmonize(self.max_iter_harmony, self.verbose)

    # =========================================================================
    # Properties - Return NumPy arrays for inspection and tutorials
    # =========================================================================
    
    @property
    def Z_corr(self):
        """Corrected embedding matrix (N x d). Batch effects removed."""
        return self._Z_corr.cpu().numpy().T
    
    @property
    def Z_orig(self):
        """Original embedding matrix (N x d). Input data before correction."""
        return self._Z_orig.cpu().numpy().T
    
    @property
    def Z_cos(self):
        """L2-normalized embedding matrix (N x d). Used for clustering."""
        return self._Z_corr.cpu().numpy().T
    
    @property
    def R(self):
        """Soft cluster assignment matrix (N x K). R[i,k] = P(cell i in cluster k)."""
        return self._R.cpu().numpy().T
    
    @property
    def Y(self):
        """Cluster centroids matrix (d x K). Columns are cluster centers."""
        return self._Y.cpu().numpy()
    
    @property
    def O(self):
        """Observed batch-cluster counts (K x B). O[k,b] = sum of R[k,:] for batch b."""
        return self._O.cpu().numpy()
    
    @property
    def E(self):
        """Expected batch-cluster counts (K x B). E[k,b] = cluster_size[k] * batch_proportion[b]."""
        return self._E.cpu().numpy()
    
    @property
    def Phi(self):
        """Batch indicator matrix (N x B). One-hot encoding of batch membership."""
        phi = torch.zeros(self.B, self.N, dtype=torch.float32)
        batch_cpu = self._batch_of_cell.cpu()
        for c in range(self._n_covariates):
            phi.scatter_(0, batch_cpu[c:c+1], 1.0)
        return phi.numpy().T

    @property
    def Phi_moe(self):
        """Batch indicator with intercept (N x (B+1)). First column is all ones."""
        phi = self.Phi  # N x B numpy
        ones = np.ones((self.N, 1), dtype=np.float32)
        return np.hstack([ones, phi])
    
    @property
    def Pr_b(self):
        """Batch proportions (B,). Pr_b[b] = cells in batch b / total cells."""
        return self._Pr_b.cpu().numpy()
    
    @property
    def theta(self):
        """Diversity penalty parameters (B,). Higher = more mixing encouraged."""
        return self._theta.cpu().numpy()
    
    @property
    def sigma(self):
        """Clustering bandwidth parameters (K,). Soft assignment kernel width."""
        return self._sigma.cpu().numpy()
    
    @property 
    def lamb(self):
        """Ridge regression penalty ((B+1),). Regularization for batch correction."""
        return self._lamb.cpu().numpy()

    def result(self):
        """Return corrected data as NumPy array."""
        return self._Z_corr.cpu().numpy().T

    def allocate_buffers(self):
        self._dist_mat = torch.zeros((self.K, self.N), dtype=torch.float32, device=self.device)
        self._O = torch.zeros((self.K, self.B), dtype=torch.float32, device=self.device)
        self._E = torch.zeros((self.K, self.B), dtype=torch.float32, device=self.device)
        self._R = torch.zeros((self.K, self.N), dtype=torch.float32, device=self.device)
        self._Y = torch.zeros((self.d, self.K), dtype=torch.float32, device=self.device)

    def _compute_O_from_R(self):
        """Compute O[k,b] = sum of R[k, cells_in_batch_b] via scatter_add."""
        self._O.zero_()
        for c in range(self._n_covariates):
            self._O.scatter_add_(1, self._batch_of_cell[c:c+1].expand(self.K, -1), self._R)

    def _scatter_add_to_O(self, block_batches, R_block, sign=1.0):
        """Accumulate R_block contributions into O via scatter_add."""
        src = R_block if sign > 0 else -R_block
        for c in range(self._n_covariates):
            self._O.scatter_add_(1, block_batches[c:c+1].expand(self.K, -1), src)

    def init_cluster(self, random_state):
        logger.info("Computing initial centroids with sklearn.KMeans...")
        # KMeans needs CPU numpy array (Z_corr is L2-normalized at this point)
        Z_cos_np = self._Z_corr.cpu().numpy()
        model = KMeans(n_clusters=self.K, init='k-means++',
                       n_init=1, max_iter=25, random_state=random_state)
        model.fit(Z_cos_np.T)
        self._Y = torch.tensor(model.cluster_centers_.T, dtype=torch.float32, device=self.device)
        logger.info("KMeans initialization complete.")
        
        # Normalize centroids
        self._Y = self._Y / torch.linalg.norm(self._Y, ord=2, dim=0)
        
        # Compute distance matrix: dist = 2 * (1 - Y.T @ Z_cos)
        self._dist_mat = 2 * (1 - self._Y.T @ self._Z_corr)
        
        # Compute R
        self._R = -self._dist_mat / self._sigma[:, None]
        self._R = torch.exp(self._R)
        self._R = self._R / self._R.sum(dim=0)
        
        # Batch diversity statistics
        self._E = torch.outer(self._R.sum(dim=1), self._Pr_b)
        self._compute_O_from_R()
        
        self.compute_objective()
        self.objective_harmony.append(self.objective_kmeans[-1])

    def compute_objective(self):
        norm_const = 2000.0 / self.N

        # K-means error: sum(R * dist_mat) via dot on flat views (no K x N temp)
        kmeans_error = torch.dot(self._R.reshape(-1), self._dist_mat.reshape(-1)).item()

        # Entropy: sum(xlogy(R, R) * sigma)
        # = sum_k sigma_k * sum_j R[k,j]*log(R[k,j])
        # Compute per-row sums of xlogy then dot with sigma (no K x N sigma broadcast)
        entropy_per_k = torch.xlogy(self._R, self._R).sum(dim=1)
        _entropy = torch.dot(entropy_per_k, self._sigma).item()

        # Cross entropy: sum((R * sigma) * (theta_log @ Phi))
        # Since R @ Phi.T = O, we have (R * sigma) @ Phi.T = sigma[:,None] * O
        # This reduces the K x N matmul to a K x B element-wise product
        ratio = (self._O + self._E + 1) / (2 * self._E + 1)
        theta_log = self._theta.unsqueeze(0).expand(self.K, -1) * torch.log(ratio)
        _cross_entropy = torch.sum(self._sigma[:, None] * self._O * theta_log).item()

        self.objective_kmeans.append((kmeans_error + _entropy + _cross_entropy) * norm_const)
        self.objective_kmeans_dist.append(kmeans_error * norm_const)
        self.objective_kmeans_entropy.append(_entropy * norm_const)
        self.objective_kmeans_cross.append(_cross_entropy * norm_const)

    def harmonize(self, iter_harmony=10, verbose=True):
        converged = False
        for i in range(1, iter_harmony + 1):
            if verbose:
                logger.info(f"Iteration {i} of {iter_harmony}")
            
            self.cluster()
            self.moe_correct_ridge()
            
            converged = self.check_convergence(1)
            if converged:
                if verbose:
                    logger.info(f"Converged after {i} iteration{'s' if i > 1 else ''}")
                break
                
        if verbose and not converged:
            logger.info("Stopped before convergence")

    def cluster(self):
        # Cold-start R re-estimation after correction (harmony2)
        # On iterations after the first, R is stale because Z_corr changed
        if len(self.objective_harmony) > 1:
            self._Z_corr /= torch.linalg.norm(self._Z_corr, ord=2, dim=0)
            torch.mm(self._Y.T, self._Z_corr, out=self._dist_mat)
            self._dist_mat.mul_(-2).add_(2)  # dist = 2*(1 - Y.T @ Z_cos)
            # Compute R in-place from dist_mat
            torch.div(self._dist_mat, self._sigma[:, None], out=self._R)
            self._R.neg_().exp_()
            self._R /= self._R.sum(dim=0)
            self._E = torch.outer(self._R.sum(dim=1), self._Pr_b)
            self._compute_O_from_R()

        rounds = 0
        for i in range(self.max_iter_kmeans):
            # harmony2: no Y/dist_mat update inside k-means loop
            # Centroids are updated in moe_correct_ridge() instead

            # Update R
            self.update_R()

            # Compute objective and check convergence
            self.compute_objective()

            if i > self.window_size:
                if self.check_convergence(0):
                    rounds = i + 1
                    break
            rounds = i + 1

        self.kmeans_rounds.append(rounds)
        self.objective_harmony.append(self.objective_kmeans[-1])

    def update_R(self):
        # Create shuffled update order
        update_order = torch.randperm(self.N, device=self.device)

        # Shuffle R and dist_mat in-place (no separate _scale_dist buffer — matches C++)
        self._R[:] = self._R[:, update_order]
        self._dist_mat[:] = self._dist_mat[:, update_order]

        # Process in blocks
        n_blocks = int(np.ceil(1.0 / self.block_size))
        cells_per_block = int(self.N * self.block_size)

        for blk in range(n_blocks):
            idx_min = blk * cells_per_block
            idx_max = self.N if blk == n_blocks - 1 else (blk + 1) * cells_per_block
            block_cells = update_order[idx_min:idx_max]

            R_block = self._R[:, idx_min:idx_max]
            dist_block = self._dist_mat[:, idx_min:idx_max]
            block_batches = self._batch_of_cell[:, block_cells]

            # Remove cells from statistics
            self._E -= torch.outer(R_block.sum(dim=1), self._Pr_b)
            self._scatter_add_to_O(block_batches, R_block, sign=-1.0)

            # Recompute R for this block from dist_mat (no _scale_dist buffer)
            scale_block = torch.exp(-dist_block / self._sigma[:, None])
            scale_block /= scale_block.sum(dim=0)

            # Diversity penalty (harmony2 formula)
            ratio = (2 * self._E + 1) / (self._O + self._E + 1)
            ratio_powered = harmony_pow_torch(ratio, self._theta)
            diversity = ratio_powered[:, block_batches[0]]
            for c in range(1, self._n_covariates):
                diversity = diversity + ratio_powered[:, block_batches[c]]
            R_block_new = scale_block * diversity
            R_block_sum = R_block_new.sum(dim=0)
            R_block_sum = torch.clamp(R_block_sum, min=1e-8)
            R_block_new = R_block_new / R_block_sum

            # Put cells back
            self._E += torch.outer(R_block_new.sum(dim=1), self._Pr_b)
            self._scatter_add_to_O(block_batches, R_block_new, sign=1.0)

            self._R[:, idx_min:idx_max] = R_block_new

        # Restore original order in-place
        inverse_order = torch.argsort(update_order)
        self._R[:] = self._R[:, inverse_order]
        self._dist_mat[:] = self._dist_mat[:, inverse_order]

    def check_convergence(self, i_type):
        if i_type == 0:
            if len(self.objective_kmeans) <= self.window_size + 1:
                return False
            
            w = self.window_size
            obj_old = sum(self.objective_kmeans[-w-1:-1])
            obj_new = sum(self.objective_kmeans[-w:])
            return abs(obj_old - obj_new) / abs(obj_old) < self.epsilon_kmeans
        
        if i_type == 1:
            if len(self.objective_harmony) < 2:
                return False
            
            obj_old = self.objective_harmony[-2]
            obj_new = self.objective_harmony[-1]
            return (obj_old - obj_new) / abs(obj_old) < self.epsilon_harmony
        
        return True

    def moe_correct_ridge(self):
        """Ridge regression correction for batch effects (harmony2)."""
        self._Z_corr.copy_(self._Z_orig)

        for k in range(self.K):
            # Determine which batches have sufficient representation
            avg_R = self._O[k, :] / self._batch_sizes
            keep = []
            cov_levels = [0] * len(self.B_vec)
            current_cov = 0
            for b in range(self.B):
                # Map batch index to covariate
                if current_cov < len(self.covariate_bounds) - 1 and b >= self.covariate_bounds[current_cov]:
                    current_cov += 1
                if avg_R[b].item() > self.batch_prop_cutoff:
                    cov_levels[current_cov] += 1

            # Count active covariates (those with >1 qualifying level)
            active_covariates = sum(1 for l in cov_levels if l > 1)

            # Collect qualifying batches (must have >1 level in its covariate)
            current_cov = 0
            for b in range(self.B):
                if current_cov < len(self.covariate_bounds) - 1 and b >= self.covariate_bounds[current_cov]:
                    current_cov += 1
                if avg_R[b].item() > self.batch_prop_cutoff and cov_levels[current_cov] > 1:
                    keep.append(b)

            if active_covariates == 0:
                # No covariates qualify, skip correction for this cluster
                continue

            if len(keep) == self.B:
                # All batches qualify - use full data (no subsetting needed)
                if self.lambda_estimation:
                    lamb_vec = find_lambda_torch(self.alpha, self._E[k, :], self.device)
                else:
                    lamb_vec = self._lamb

                # Build cov_mat directly from O[k,:] -- no (B+1) x N Phi_Rk needed
                Ok = self._O[k, :]
                cov_mat = torch.zeros(self.B + 1, self.B + 1, dtype=torch.float32, device=self.device)
                cov_mat[0, 0] = Ok.sum()
                cov_mat[0, 1:] = Ok
                cov_mat[1:, 0] = Ok
                cov_mat[1:, 1:] = torch.diag(Ok)
                cov_mat += torch.diag(lamb_vec)

                # Arrowhead optimization for single covariate
                if len(self.B_vec) > 1:
                    inv_cov = torch.linalg.inv(cov_mat)
                else:
                    inv_cov = _arrowhead_inv(cov_mat)

                # Compute W via per-batch matrix-vector products -- no d x N Z_tmp
                Rk = self._R[k, :]
                z_sums = []
                for b in range(self.B):
                    idx = self._batch_index[b]
                    z_sums.append(self._Z_orig[:, idx] @ Rk[idx])

                z_sum_all = sum(z_sums)
                W = inv_cov[:, 0:1] @ z_sum_all.unsqueeze(0)
                for b in range(self.B):
                    W = W + inv_cov[:, b+1:b+2] @ z_sums[b].unsqueeze(0)

                # Update centroid from intercept, then zero it out
                self._Y[:, k] = W[0, :]
                W[0, :] = 0

                # Batch-wise in-place correction -- no d x N product
                for b in range(self.B):
                    idx = self._batch_index[b]
                    self._Z_corr[:, idx] -= W[b+1:b+2, :].T @ self._R[k:k+1, idx]
            else:
                # Subset to qualifying batches
                n_keep = len(keep)

                # Lambda for subsetted batches
                if self.lambda_estimation:
                    E_sub = self._E[k, torch.tensor(keep, device=self.device)]
                    lamb_vec = find_lambda_torch(self.alpha, E_sub, self.device)
                else:
                    keep_lamb_idx = [0] + [b + 1 for b in keep]
                    lamb_vec = self._lamb[keep_lamb_idx]

                # Build cov_mat directly from O[k, keep] -- no subsetting matrices needed
                Ok_sub = self._O[k, torch.tensor(keep, device=self.device)]
                cov_mat = torch.zeros(n_keep + 1, n_keep + 1, dtype=torch.float32, device=self.device)
                cov_mat[0, 0] = Ok_sub.sum()
                cov_mat[0, 1:] = Ok_sub
                cov_mat[1:, 0] = Ok_sub
                cov_mat[1:, 1:] = torch.diag(Ok_sub)
                cov_mat += torch.diag(lamb_vec)

                if len(self.B_vec) > 1:
                    inv_cov = torch.linalg.inv(cov_mat)
                else:
                    inv_cov = _arrowhead_inv(cov_mat)

                # Compute W via per-batch matrix-vector products -- no d x N Z_tmp
                Rk = self._R[k, :]
                z_sums = []
                for b in keep:
                    idx = self._batch_index[b]
                    z_sums.append(self._Z_orig[:, idx] @ Rk[idx])

                z_sum_all = sum(z_sums)
                W = inv_cov[:, 0:1] @ z_sum_all.unsqueeze(0)
                for i in range(n_keep):
                    W = W + inv_cov[:, i+1:i+2] @ z_sums[i].unsqueeze(0)

                # Update centroid from intercept, then zero it out
                self._Y[:, k] = W[0, :]
                W[0, :] = 0

                # Batch-wise in-place correction -- no d x N product
                for i, b in enumerate(keep):
                    idx = self._batch_index[b]
                    self._Z_corr[:, idx] -= W[i+1:i+2, :].T @ self._R[k:k+1, idx]

        # Normalize centroids
        self._Y = self._Y / torch.linalg.norm(self._Y, ord=2, dim=0)
        # Z_corr will be re-normalized at the start of the next cluster() call


def harmony_pow_torch(A, T):
    """Element-wise power with different exponents per column."""
    return torch.pow(A, T.unsqueeze(0))


def find_lambda_torch(alpha, cluster_E, device):
    """Compute dynamic lambda based on cluster expected counts."""
    lamb = torch.zeros(len(cluster_E) + 1, dtype=torch.float32, device=device)
    lamb[1:] = cluster_E * alpha
    return lamb


def _arrowhead_inv(cov_mat):
    """Compute inverse of an arrowhead matrix analytically.

    When there is only one covariate, the covariance matrix Phi_cov has
    arrowhead structure which allows analytical inversion instead of
    full matrix inversion. This is faster and more numerically stable.
    """
    ac = -cov_mat[0, :].clone()
    ac[0] = 1.0
    b0 = cov_mat[0, 0]
    b = 1.0 / torch.diag(cov_mat)
    b[0] = 0.0
    u = b0 - torch.sum(ac * ac * b)
    ac_b = ac * b
    ac_b[0] = 1.0
    inv_cov = (1.0 / u) * torch.outer(ac_b, ac_b)
    inv_cov += torch.diag(b)
    return inv_cov
