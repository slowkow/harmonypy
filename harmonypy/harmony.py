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

from functools import partial
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
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
    max_iter_kmeans=20,
    epsilon_cluster=1e-5,
    epsilon_harmony=1e-4, 
    alpha=0.2,
    verbose=True,
    random_state=0
):
    """Run Harmony batch effect correction.
    
    This is a pure NumPy implementation matching the R package formulas.
    
    Parameters
    ----------
    data_mat : np.ndarray
        PCA embedding matrix (cells x PCs or PCs x cells)
    meta_data : pd.DataFrame
        Metadata with batch variables
    vars_use : str or list
        Column name(s) in meta_data to use for batch correction
    theta : float or list, optional
        Diversity penalty parameter(s). Default is 2 for each batch.
    lamb : float or list, optional
        Ridge regression penalty. Default is 1 for each batch.
        If -1, lambda is estimated automatically (matches R package).
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
        Maximum k-means iterations per Harmony iteration. Default is 20.
    epsilon_cluster : float, optional
        K-means convergence threshold. Default is 1e-5.
    epsilon_harmony : float, optional
        Harmony convergence threshold. Default is 1e-4.
    alpha : float, optional
        Alpha parameter for lambda estimation (when lamb=-1). Default is 0.2.
    verbose : bool, optional
        Print progress messages. Default is True.
    random_state : int, optional
        Random seed for reproducibility. Default is 0.
        
    Returns
    -------
    Harmony
        Harmony object with corrected data in Z_corr attribute (cells × PCs).
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

    # Create batch indicator matrix (one-hot encoded)
    phi = pd.get_dummies(meta_data[vars_use]).to_numpy().T.astype(np.float64)
    phi_n = meta_data[vars_use].describe().loc['unique'].to_numpy().astype(int)

    # Theta handling - default is 2 (matches R package)
    if theta is None:
        theta = np.repeat([2] * len(phi_n), phi_n).astype(np.float64)
    elif isinstance(theta, (float, int)):
        theta = np.repeat([theta] * len(phi_n), phi_n).astype(np.float64)
    elif len(theta) == len(phi_n):
        theta = np.repeat([theta], phi_n).astype(np.float64)
    else:
        theta = np.asarray(theta, dtype=np.float64)

    assert len(theta) == np.sum(phi_n), \
        "each batch variable must have a theta"

    # Lambda handling (matches R package)
    # If lamb is None, use default of 1
    # If lamb is -1, enable lambda estimation
    lambda_estimation = False
    if lamb is None:
        lamb = np.repeat([1] * len(phi_n), phi_n).astype(np.float64)
        lamb = np.insert(lamb, 0, 0).astype(np.float64)
    elif lamb == -1:
        # Lambda estimation mode (matches R package)
        lambda_estimation = True
        lamb = np.zeros(1)  # Placeholder
    elif isinstance(lamb, (float, int)):
        lamb = np.repeat([lamb] * len(phi_n), phi_n).astype(np.float64)
        lamb = np.insert(lamb, 0, 0).astype(np.float64)
    elif len(lamb) == len(phi_n):
        lamb = np.repeat([lamb], phi_n).astype(np.float64)
        lamb = np.insert(lamb, 0, 0).astype(np.float64)
    else:
        lamb = np.asarray(lamb, dtype=np.float64)
        if len(lamb) == np.sum(phi_n):
            lamb = np.insert(lamb, 0, 0).astype(np.float64)

    # Number of items in each category
    N_b = phi.sum(axis=1)
    # Proportion of items in each category
    Pr_b = (N_b / N).astype(np.float64)

    if tau > 0:
        theta = theta * (1 - np.exp(-(N_b / (nclust * tau)) ** 2))

    np.random.seed(random_state)

    if verbose:
        logger.info("Running Harmony (NumPy, R package compatible)")
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
            logger.info(f"    lamb: {lamb[1:]}")  # Skip intercept
        logger.info(f"    theta: {theta}")
        logger.info(f"    sigma: {sigma[:5]}..." if len(sigma) > 5 else f"    sigma: {sigma}")
        logger.info(f"    verbose: {verbose}")
        logger.info(f"    random_state: {random_state}")
        logger.info(f"  Data: {data_mat.shape[0]} PCs × {N} cells")
        logger.info(f"  Batch variables: {vars_use}")

    ho = Harmony(
        data_mat, phi, Pr_b, sigma, theta, lamb, alpha, lambda_estimation,
        max_iter_harmony, max_iter_kmeans,
        epsilon_cluster, epsilon_harmony, nclust, block_size, verbose,
        random_state
    )

    return ho


class Harmony:
    """Harmony class for batch effect correction.
    
    Updated to match R package implementation for improved performance.
    
    Attributes
    ----------
    Z_corr : np.ndarray
        Batch-corrected embedding matrix (cells × PCs). This is the main output.
    Z_orig : np.ndarray
        Original embedding matrix (cells × PCs).
    Z_cos : np.ndarray
        L2-normalized embedding (cells × PCs), used for soft clustering.
    R : np.ndarray
        Soft cluster assignment matrix (cells × clusters). R[i,k] is the 
        probability that cell i belongs to cluster k.
    Y : np.ndarray
        Cluster centroid matrix (PCs × clusters).
    O : np.ndarray
        Observed counts matrix (clusters × batches). O[k,b] is the sum of 
        cluster k assignment probabilities for cells in batch b.
    E : np.ndarray
        Expected counts matrix (clusters × batches). E[k,b] is the expected 
        count if cells were distributed proportionally across batches.
    Phi : np.ndarray
        Batch indicator matrix (cells × batches). Phi[i,b] = 1 if cell i 
        belongs to batch b.
    """
    
    def __init__(
            self, Z, Phi, Pr_b, sigma, theta, lamb, alpha, lambda_estimation,
            max_iter_harmony, max_iter_kmeans, 
            epsilon_kmeans, epsilon_harmony, K, block_size, verbose,
            random_state=None
    ):
        # Store original data as PCs × cells (internal representation)
        self.Z_orig = np.array(Z, dtype=np.float64)
        self.Z_corr = np.array(Z, dtype=np.float64)

        # L2 normalization for cosine distance clustering (matches R package)
        self.Z_cos = self.Z_orig / np.linalg.norm(self.Z_orig, ord=2, axis=0)

        # Batch indicator matrix: Phi[b,i] = 1 if cell i is in batch b
        self.Phi = sp.csc_matrix(Phi)
        self.Phi_dense = Phi  # Dense version for matrix operations
        
        # Pr_b[b] = proportion of cells in batch b
        self.Pr_b = Pr_b
        
        self.N = self.Z_corr.shape[1]  # Number of cells
        self.B = Phi.shape[0]           # Number of batches
        self.d = self.Z_corr.shape[0]   # Number of PCs
        
        # Pre-compute cell indices for each batch (speeds up ridge correction)
        self.batch_index = []
        for b in range(self.B):
            self.batch_index.append(np.where(Phi[b, :] > 0)[0])
        
        # Create Phi_moe with intercept
        self.Phi_moe = np.vstack((np.ones(self.N), Phi)).astype(np.float64)
        
        self.window_size = 3
        self.epsilon_kmeans = epsilon_kmeans
        self.epsilon_harmony = epsilon_harmony

        self.lamb = lamb
        self.alpha = alpha
        self.lambda_estimation = lambda_estimation
        self.sigma = np.asarray(sigma, dtype=np.float64)
        self.block_size = block_size
        self.K = K
        self.max_iter_harmony = max_iter_harmony
        self.max_iter_kmeans = max_iter_kmeans
        self.verbose = verbose
        self.theta = theta

        self.objective_harmony = []
        self.objective_kmeans = []
        self.objective_kmeans_dist = []
        self.objective_kmeans_entropy = []
        self.objective_kmeans_cross = []
        self.kmeans_rounds = []

        self.allocate_buffers()
        self.init_cluster(random_state)
        self.harmonize(self.max_iter_harmony, self.verbose)

    def result(self):
        return self.Z_corr

    def allocate_buffers(self):
        """Allocate memory for intermediate matrices."""
        self._scale_dist = np.zeros((self.K, self.N), dtype=np.float64)  # Scaled distances
        self.dist_mat = np.zeros((self.K, self.N), dtype=np.float64)     # Distance to centroids
        self.O = np.zeros((self.K, self.B), dtype=np.float64)  # Observed batch counts per cluster
        self.E = np.zeros((self.K, self.B), dtype=np.float64)  # Expected batch counts per cluster
        self.W = np.zeros((self.B + 1, self.d), dtype=np.float64)  # Ridge regression coefficients
        self.R = np.zeros((self.K, self.N), dtype=np.float64)  # Soft cluster assignments
        self.Y = np.zeros((self.d, self.K), dtype=np.float64)  # Cluster centroids

    def init_cluster(self, random_state):
        logger.info("Computing initial centroids with sklearn.KMeans...")
        model = KMeans(n_clusters=self.K, init='k-means++',
                       n_init=1, max_iter=25, random_state=random_state)
        model.fit(self.Z_cos.T)
        self.Y = model.cluster_centers_.T
        logger.info("KMeans initialization complete.")
        
        # Normalize centroids (matches R package)
        self.Y = self.Y / np.linalg.norm(self.Y, ord=2, axis=0)
        
        # Compute distance matrix: dist = 2 * (1 - Y.T @ Z_cos)
        self.dist_mat = 2 * (1 - self.Y.T @ self.Z_cos)
        
        # Compute R (matches R package - no max subtraction for stability)
        self.R = -self.dist_mat / self.sigma[:, None]
        np.exp(self.R, out=self.R)
        self.R /= self.R.sum(axis=0)
        
        # Batch diversity statistics
        self.E = np.outer(self.R.sum(axis=1), self.Pr_b)
        self.O = self.R @ self.Phi.T.toarray()
        
        self.compute_objective()
        self.objective_harmony.append(self.objective_kmeans[-1])

    def compute_objective(self):
        # Normalization constant (matches R package)
        norm_const = 2000.0 / self.N
        
        # K-means error
        kmeans_error = np.sum(self.R * self.dist_mat)
        
        # Entropy: sum(safe_entropy(R) * sigma)
        _entropy = np.sum(safe_entropy(self.R) * self.sigma[:, None])
        
        # Cross entropy (matches R package formula)
        # R package: log((O + E) / E) instead of log((E+1)/(O+1))
        R_sigma = self.R * self.sigma[:, None]
        theta_log = np.tile(self.theta, (self.K, 1)) * np.log((self.O + self.E) / self.E)
        _cross_entropy = np.sum(R_sigma * (theta_log @ self.Phi_dense))
        
        # Store with normalization constant
        self.objective_kmeans.append((kmeans_error + _entropy + _cross_entropy) * norm_const)
        self.objective_kmeans_dist.append(kmeans_error * norm_const)
        self.objective_kmeans_entropy.append(_entropy * norm_const)
        self.objective_kmeans_cross.append(_cross_entropy * norm_const)

    def harmonize(self, iter_harmony=10, verbose=True):
        converged = False
        for i in range(1, iter_harmony + 1):
            if verbose:
                logger.info(f"Iteration {i} of {iter_harmony}")
            
            # STEP 1: Clustering
            self.cluster()
            
            # STEP 2: Ridge regression correction
            self.moe_correct_ridge()
            
            # STEP 3: Check for convergence
            converged = self.check_convergence(1)
            if converged:
                if verbose:
                    logger.info(f"Converged after {i} iteration{'s' if i > 1 else ''}")
                break
                
        if verbose and not converged:
            logger.info("Stopped before convergence")
        
        # Transpose matrices so cells are always the first dimension
        # This makes the output more intuitive: Z_corr[i, :] is cell i's corrected PCs
        self.Z_corr = self.Z_corr.T   # PCs × cells → cells × PCs
        self.Z_orig = self.Z_orig.T   # PCs × cells → cells × PCs
        self.Z_cos = self.Z_cos.T     # PCs × cells → cells × PCs
        self.R = self.R.T             # clusters × cells → cells × clusters
        self.Phi = self.Phi_dense.T   # batches × cells → cells × batches

    def cluster(self):
        self.dist_mat = 2 * (1 - self.Y.T @ self.Z_cos)
        
        rounds = 0
        for i in range(self.max_iter_kmeans):
            # STEP 1: Update Y
            self.Y = self.Z_cos @ self.R.T
            self.Y /= np.linalg.norm(self.Y, ord=2, axis=0)
            
            # STEP 2: Update distance matrix
            self.dist_mat = 2 * (1 - self.Y.T @ self.Z_cos)
            
            # STEP 3: Update R
            self.update_R()
            
            # STEP 4: Compute objective and check convergence
            self.compute_objective()
            
            if i > self.window_size:
                if self.check_convergence(0):
                    rounds = i + 1
                    break
            rounds = i + 1
            
        self.kmeans_rounds.append(rounds)
        self.objective_harmony.append(self.objective_kmeans[-1])

    def update_R(self):
        # Compute scaled distances (matches R package)
        self._scale_dist = -self.dist_mat / self.sigma[:, None]
        np.exp(self._scale_dist, out=self._scale_dist)
        self._scale_dist /= self._scale_dist.sum(axis=0)  # L1 normalize columns
        
        # Create shuffled update order
        update_order = np.arange(self.N)
        np.random.shuffle(update_order)
        
        # Process in blocks
        n_blocks = int(np.ceil(1.0 / self.block_size))
        cells_per_block = int(self.N * self.block_size)
        
        # Permute matrices for block processing
        R_perm = self.R[:, update_order]
        scale_perm = self._scale_dist[:, update_order]
        Phi_perm = self.Phi_dense[:, update_order]
        
        for blk in range(n_blocks):
            idx_min = blk * cells_per_block
            idx_max = self.N if blk == n_blocks - 1 else (blk + 1) * cells_per_block
            
            R_block = R_perm[:, idx_min:idx_max]
            scale_block = scale_perm[:, idx_min:idx_max]
            Phi_block = Phi_perm[:, idx_min:idx_max]
            
            # STEP 1: Remove cells from statistics
            self.E -= np.outer(R_block.sum(axis=1), self.Pr_b)
            self.O -= R_block @ Phi_block.T
            
            # STEP 2: Recompute R for this block (matches R package formula)
            # R package: E / (O + E) raised to power theta
            ratio = self.E / (self.O + self.E)
            ratio_powered = harmony_pow(ratio, self.theta)
            R_block_new = scale_block * (ratio_powered @ Phi_block)
            R_block_new /= R_block_new.sum(axis=0)  # L1 normalize columns
            
            # STEP 3: Put cells back
            self.E += np.outer(R_block_new.sum(axis=1), self.Pr_b)
            self.O += R_block_new @ Phi_block.T
            
            # Update
            R_perm[:, idx_min:idx_max] = R_block_new
        
        # Restore original order
        inverse_order = np.argsort(update_order)
        self.R = R_perm[:, inverse_order]

    def check_convergence(self, i_type):
        if i_type == 0:
            # K-means convergence
            if len(self.objective_kmeans) <= self.window_size + 1:
                return False
            
            w = self.window_size
            obj_old = sum(self.objective_kmeans[-w-1:-1])
            obj_new = sum(self.objective_kmeans[-w:])
            return abs(obj_old - obj_new) / abs(obj_old) < self.epsilon_kmeans
        
        if i_type == 1:
            # Harmony convergence
            if len(self.objective_harmony) < 2:
                return False
            
            obj_old = self.objective_harmony[-2]
            obj_new = self.objective_harmony[-1]
            return (obj_old - obj_new) / abs(obj_old) < self.epsilon_harmony
        
        return True

    def moe_correct_ridge(self):
        """Ridge regression correction for batch effects.
        
        Updated to match R package implementation with batch index optimization.
        """
        self.Z_corr = self.Z_orig.copy()
        
        for k in range(self.K):
            # Compute lambda if estimating
            if self.lambda_estimation:
                lamb_vec = find_lambda(self.alpha, self.E[k, :])
            else:
                lamb_vec = self.lamb
            
            # Phi_Rk = Phi_moe scaled by R[k,:]
            Phi_Rk = self.Phi_moe * self.R[k, :]
            
            # Compute covariance: Phi_Rk @ Phi_moe.T + diag(lambda)
            cov_mat = Phi_Rk @ self.Phi_moe.T + np.diag(lamb_vec)
            
            # Invert
            inv_cov = np.linalg.inv(cov_mat)
            
            # Calculate R-scaled PCs
            Z_tmp = self.Z_orig * self.R[k, :]
            
            # Generate betas using the batch index (matches R package optimization)
            W = inv_cov[:, 0:1] @ Z_tmp.sum(axis=1, keepdims=True).T  # Intercept contribution
            
            for b in range(self.B):
                batch_sum = Z_tmp[:, self.batch_index[b]].sum(axis=1, keepdims=True)
                W += inv_cov[:, b+1:b+2] @ batch_sum.T
            
            W[0, :] = 0  # Do not remove intercept
            self.Z_corr -= W.T @ Phi_Rk
        
        # Update Z_cos (matches R package)
        self.Z_cos = self.Z_corr / np.linalg.norm(self.Z_corr, ord=2, axis=0)


def safe_entropy(x):
    """Compute x * log(x), returning 0 where x is 0 or negative."""
    result = x * np.log(x)
    result = np.where(np.isfinite(result), result, 0.0)
    return result


def harmony_pow(A, T):
    """Element-wise power with different exponents per column.
    
    Matches R package's harmony_pow function.
    """
    result = np.empty_like(A)
    for c in range(A.shape[1]):
        result[:, c] = np.power(A[:, c], T[c])
    return result


def find_lambda(alpha, cluster_E):
    """Compute dynamic lambda based on cluster expected counts.
    
    Matches R package's find_lambda_cpp.
    """
    lamb = np.zeros(len(cluster_E) + 1)
    lamb[1:] = cluster_E * alpha
    return lamb
