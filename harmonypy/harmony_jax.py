# harmonypy - A data alignment algorithm.
# Copyright (C) 2018  Ilya Korsunsky
#               2019  Kamil Slowikowski <kslowikowski@gmail.com>
#
# JAX-accelerated implementation for high performance.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
JAX-accelerated Harmony implementation.

This module provides a JAX backend for the Harmony algorithm, offering
significant speedups especially on GPU. The key optimization is using
jax.vmap to parallelize the K-cluster loop in the ridge regression step.

Usage:
    import harmonypy as hm
    
    # Use JAX backend (falls back to NumPy if JAX not available)
    ho = hm.run_harmony(data_mat, meta_data, ['batch'], use_jax=True)
    
    # Or use the JAX-specific function directly
    from harmonypy.harmony_jax import run_harmony_jax
    ho = run_harmony_jax(data_mat, meta_data, ['batch'])
"""

from functools import partial
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import logging

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    import jax.scipy.linalg
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np  # Fallback to numpy

# create logger
logger = logging.getLogger('harmonypy')


def check_jax_available():
    """Check if JAX is available and raise informative error if not."""
    if not JAX_AVAILABLE:
        raise ImportError(
            "JAX is required for the JAX-accelerated backend. "
            "Install it with: pip install harmonypy[jax] "
            "or: pip install jax jaxlib"
        )


def get_jax_device_info():
    """Get information about available JAX devices.
    
    Returns:
        dict with keys: 'device_type', 'device_name', 'message', 'gpu_available'
    """
    if not JAX_AVAILABLE:
        return {
            'device_type': 'none',
            'device_name': 'JAX not installed',
            'message': 'JAX not available',
            'gpu_available': False
        }
    
    devices = jax.devices()
    default_device = devices[0] if devices else None
    backend = jax.default_backend()
    
    # Determine device type
    if backend == 'gpu' or 'gpu' in str(default_device).lower():
        device_type = 'GPU'
        gpu_available = True
    elif backend == 'METAL' or 'metal' in str(default_device).lower():
        device_type = 'GPU (Metal)'
        gpu_available = True
    elif backend == 'tpu':
        device_type = 'TPU'
        gpu_available = True
    else:
        device_type = 'CPU'
        gpu_available = False
    
    device_name = str(default_device) if default_device else 'unknown'
    
    # Create helpful message
    if gpu_available:
        message = f"🚀 Running on {device_type}: {device_name}"
        if 'metal' in device_type.lower():
            message += " (experimental)"
    else:
        message = f"Running on {device_type}: {device_name}"
        # Add hint for GPU acceleration
        import platform
        if platform.system() == 'Darwin' and platform.machine() == 'arm64':
            message += "\n   💡 Tip: Install jax-metal for GPU acceleration on Apple Silicon"
        elif platform.system() == 'Linux':
            message += "\n   💡 Tip: Install jax[cuda12] for NVIDIA GPU acceleration"
    
    return {
        'device_type': device_type,
        'device_name': device_name,
        'message': message,
        'gpu_available': gpu_available
    }


def _test_jax_backend():
    """Test if the current JAX backend works by creating a simple array.
    
    Returns:
        bool: True if backend works, False otherwise
        str: Error message if backend fails, empty string otherwise
    """
    try:
        # Simple test: create a small array and do a basic operation
        test = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        _ = test.sum()
        return True, ""
    except Exception as e:
        return False, str(e)


# =============================================================================
# JAX-optimized core functions
# =============================================================================

if JAX_AVAILABLE:
    @partial(jit, static_argnums=(2,))
    def _compute_single_correction(R_row, args, d):
        """Compute ridge regression correction for a single cluster.
        
        This function is designed to be vmapped over all K clusters.
        
        Args:
            R_row: Cluster assignment probabilities for one cluster (N,)
            args: Tuple of (Phi_moe, Phi_moe_T, Z_orig, lamb)
            d: Number of dimensions (static)
            
        Returns:
            Correction matrix (d, N) for this cluster
        """
        Phi_moe, Phi_moe_T, Z_orig, lamb = args
        
        # Phi_Rk = Phi_moe * R_row (broadcast over rows)
        Phi_Rk = Phi_moe * R_row[None, :]
        
        # x = Phi_Rk @ Phi_moe_T + lamb
        x = Phi_Rk @ Phi_moe_T + lamb
        
        # rhs = Phi_Rk @ Z_orig.T
        rhs = Phi_Rk @ Z_orig.T
        
        # Solve x @ W = rhs for W
        W = jax.scipy.linalg.solve(x, rhs, assume_a='pos')
        
        # Zero out intercept
        W = W.at[0, :].set(0.0)
        
        # Compute correction: W.T @ Phi_Rk
        correction = W.T @ Phi_Rk
        
        return correction


    @jit
    def _moe_correct_ridge_jax(Z_orig, R, Phi_moe, Phi_moe_T, lamb):
        """JAX-accelerated ridge regression correction.
        
        Uses vmap to parallelize over all K clusters simultaneously.
        This is the key optimization - instead of a sequential loop over K=100
        clusters, we compute all corrections in parallel.
        
        Args:
            Z_orig: Original data (d, N)
            R: Cluster probabilities (K, N)
            Phi_moe: Design matrix with intercept (B+1, N)
            Phi_moe_T: Transpose of Phi_moe (N, B+1)
            lamb: Regularization matrix (B+1, B+1)
            
        Returns:
            Z_cos: L2-normalized corrected data
            Z_corr: Corrected data
        """
        d = Z_orig.shape[0]
        args = (Phi_moe, Phi_moe_T, Z_orig, lamb)
        
        # Vectorize over K clusters (first axis of R)
        # This computes all K corrections in parallel!
        compute_corrections = vmap(
            lambda r: _compute_single_correction(r, args, d)
        )
        
        # corrections shape: (K, d, N)
        corrections = compute_corrections(R)
        
        # Sum all corrections and subtract from original
        total_correction = corrections.sum(axis=0)
        Z_corr = Z_orig - total_correction
        
        # L2 normalize
        Z_cos = Z_corr / jnp.linalg.norm(Z_corr, axis=0, keepdims=True)
        
        return Z_cos, Z_corr


    @jit
    def _compute_dist_mat(Y, Z_cos):
        """Compute distance matrix between centroids and data."""
        return 2.0 * (1.0 - Y.T @ Z_cos)


    @jit
    def _update_Y(Z_cos, R):
        """Update cluster centroids."""
        Y = Z_cos @ R.T
        Y = Y / jnp.linalg.norm(Y, axis=0, keepdims=True)
        return Y


    @jit
    def _compute_R_softmax(dist_mat, sigma):
        """Compute soft cluster assignments with softmax."""
        R = -dist_mat / sigma[:, None]
        R = R - R.max(axis=0, keepdims=True)
        R = jnp.exp(R)
        R = R / R.sum(axis=0, keepdims=True)
        return R


    @jit 
    def _safe_entropy_jax(x):
        """Compute x * log(x), returning 0 where x <= 0."""
        # Use where to handle zeros safely
        return jnp.where(x > 0, x * jnp.log(x), 0.0)


class HarmonyJAX:
    """JAX-accelerated Harmony algorithm.
    
    This class provides the same interface as the NumPy Harmony class
    but uses JAX for accelerated computation, especially on GPU.
    """
    
    def __init__(
            self, Z, Phi, Phi_moe, Pr_b, sigma,
            theta, max_iter_harmony, max_iter_kmeans, 
            epsilon_kmeans, epsilon_harmony, K, block_size,
            lamb, verbose, random_state=None, cluster_fn='kmeans'
    ):
        check_jax_available()
        
        # Convert inputs to JAX arrays
        self.Z_corr = jnp.array(Z, dtype=jnp.float32)
        self.Z_orig = jnp.array(Z, dtype=jnp.float32)
        
        # Initial normalization
        Z_scaled = self.Z_orig / self.Z_orig.max(axis=0, keepdims=True)
        self.Z_cos = Z_scaled / jnp.linalg.norm(Z_scaled, axis=0, keepdims=True)
        
        # Store Phi as dense JAX array (sparse not well supported in JAX)
        if hasattr(Phi, 'toarray'):
            self.Phi = jnp.array(Phi.toarray(), dtype=jnp.float32)
        else:
            self.Phi = jnp.array(Phi, dtype=jnp.float32)
        
        self.Phi_moe = jnp.array(Phi_moe, dtype=jnp.float32)
        self.Phi_moe_T = jnp.array(Phi_moe.T, dtype=jnp.float32)
        
        self.N = self.Z_corr.shape[1]
        self.Pr_b = jnp.array(Pr_b, dtype=jnp.float32)
        self.B = self.Phi.shape[0]
        self.d = self.Z_corr.shape[0]
        self.window_size = 3
        self.epsilon_kmeans = epsilon_kmeans
        self.epsilon_harmony = epsilon_harmony
        
        self.lamb = jnp.array(lamb, dtype=jnp.float32)
        self.sigma = jnp.array(sigma, dtype=jnp.float32)
        self.sigma_prior = sigma
        self.block_size = block_size
        self.K = K
        self.max_iter_harmony = max_iter_harmony
        self.max_iter_kmeans = max_iter_kmeans
        self.verbose = verbose
        self.theta = jnp.array(theta, dtype=jnp.float32)
        
        # Tracking
        self.objective_harmony = []
        self.objective_kmeans = []
        self.objective_kmeans_dist = []
        self.objective_kmeans_entropy = []
        self.objective_kmeans_cross = []
        self.kmeans_rounds = []
        
        # Precompute theta tile
        self._theta_tile = jnp.tile(self.theta[:, None], (1, self.K)).T
        
        # Random key for JAX
        if random_state is None:
            random_state = 0
        self._key = jax.random.PRNGKey(random_state)
        
        # Initialize and run
        if cluster_fn == 'kmeans':
            cluster_fn = partial(self._cluster_kmeans, random_state=random_state)
        self._init_cluster(cluster_fn)
        self._harmonize(self.max_iter_harmony, self.verbose)
    
    def result(self):
        """Return corrected data as NumPy array."""
        return np.array(self.Z_corr)
    
    @staticmethod
    def _cluster_kmeans(data, K, random_state):
        """Initialize clusters with sklearn KMeans."""
        logger.info("Computing initial centroids with sklearn.KMeans...")
        # Use NumPy for sklearn
        data_np = np.array(data)
        model = KMeans(n_clusters=K, init='k-means++',
                       n_init=1, max_iter=25, random_state=random_state)
        model.fit(data_np)
        logger.info("sklearn.KMeans initialization complete.")
        return model.cluster_centers_
    
    def _init_cluster(self, cluster_fn):
        """Initialize cluster centroids and assignments."""
        # KMeans on CPU (sklearn), then convert to JAX
        centroids = cluster_fn(np.array(self.Z_cos.T), self.K)
        self.Y = jnp.array(centroids.T, dtype=jnp.float32)
        
        # Normalize centroids
        self.Y = self.Y / jnp.linalg.norm(self.Y, axis=0, keepdims=True)
        
        # Compute initial cluster assignments
        self.dist_mat = _compute_dist_mat(self.Y, self.Z_cos)
        self.R = _compute_R_softmax(self.dist_mat, self.sigma)
        
        # Batch diversity statistics
        self.E = jnp.outer(self.R.sum(axis=1), self.Pr_b)
        self.O = self.R @ self.Phi.T
        
        self._compute_objective()
        self.objective_harmony.append(self.objective_kmeans[-1])
    
    def _compute_objective(self):
        """Compute and store objective function value."""
        kmeans_error = float(jnp.sum(self.R * self.dist_mat))
        
        # Entropy
        _entropy = float(jnp.sum(_safe_entropy_jax(self.R) * self.sigma[:, None]))
        
        # Cross Entropy
        x = self.R * self.sigma[:, None]
        z = jnp.log((self.O + 1) / (self.E + 1))
        w = (self._theta_tile * z) @ self.Phi
        _cross_entropy = float(jnp.sum(x * w))
        
        self.objective_kmeans.append(kmeans_error + _entropy + _cross_entropy)
        self.objective_kmeans_dist.append(kmeans_error)
        self.objective_kmeans_entropy.append(_entropy)
        self.objective_kmeans_cross.append(_cross_entropy)
    
    def _harmonize(self, iter_harmony=10, verbose=True):
        """Main Harmony iteration loop."""
        converged = False
        
        for i in range(1, iter_harmony + 1):
            if verbose:
                logger.info(f"Iteration {i} of {iter_harmony}")
            
            # STEP 1: Clustering
            self._cluster()
            
            # STEP 2: Ridge regression correction (JAX-accelerated!)
            self.Z_cos, self.Z_corr = _moe_correct_ridge_jax(
                self.Z_orig, self.R, self.Phi_moe, self.Phi_moe_T, self.lamb
            )
            
            # STEP 3: Check convergence
            converged = self._check_convergence(1)
            if converged:
                if verbose:
                    logger.info(f"Converged after {i} iteration{'s' if i > 1 else ''}")
                break
        
        if verbose and not converged:
            logger.info("Stopped before convergence")
    
    def _cluster(self):
        """Soft clustering step."""
        self.dist_mat = _compute_dist_mat(self.Y, self.Z_cos)
        
        for i in range(self.max_iter_kmeans):
            # Update centroids
            self.Y = _update_Y(self.Z_cos, self.R)
            
            # Update distances
            self.dist_mat = _compute_dist_mat(self.Y, self.Z_cos)
            
            # Update R with block updates for diversity
            self._update_R()
            
            # Check convergence
            self._compute_objective()
            if i > self.window_size:
                if self._check_convergence(0):
                    break
        
        self.kmeans_rounds.append(i)
        self.objective_harmony.append(self.objective_kmeans[-1])
    
    def _update_R(self):
        """Update cluster assignments with diversity regularization."""
        # Compute base softmax
        scale_dist = -self.dist_mat / self.sigma[:, None]
        scale_dist = scale_dist - scale_dist.max(axis=0, keepdims=True)
        scale_dist = jnp.exp(scale_dist)
        
        # Block updates for diversity
        self._key, subkey = jax.random.split(self._key)
        update_order = jax.random.permutation(subkey, self.N)
        n_blocks = int(np.ceil(1 / self.block_size))
        block_size = self.N // n_blocks
        
        # Convert to numpy for block iteration (JAX arrays are immutable)
        R_np = np.array(self.R)
        E_np = np.array(self.E)
        O_np = np.array(self.O)
        scale_dist_np = np.array(scale_dist)
        Phi_np = np.array(self.Phi)
        Pr_b_np = np.array(self.Pr_b)
        theta_np = np.array(self.theta)
        
        for block_idx in range(n_blocks):
            start = block_idx * block_size
            end = min(start + block_size, self.N)
            b = np.array(update_order[start:end])
            
            # Remove cells
            E_np -= np.outer(R_np[:, b].sum(axis=1), Pr_b_np)
            O_np -= R_np[:, b] @ Phi_np[:, b].T
            
            # Recompute R
            R_np[:, b] = scale_dist_np[:, b] * (
                np.power((E_np + 1) / (O_np + 1), theta_np) @ Phi_np[:, b]
            )
            R_np[:, b] /= R_np[:, b].sum(axis=0)
            
            # Add cells back
            E_np += np.outer(R_np[:, b].sum(axis=1), Pr_b_np)
            O_np += R_np[:, b] @ Phi_np[:, b].T
        
        # Convert back to JAX
        self.R = jnp.array(R_np, dtype=jnp.float32)
        self.E = jnp.array(E_np, dtype=jnp.float32)
        self.O = jnp.array(O_np, dtype=jnp.float32)
    
    def _check_convergence(self, i_type):
        """Check if algorithm has converged."""
        if i_type == 0:  # Clustering
            w = self.window_size
            if len(self.objective_kmeans) < w + 1:
                return False
            obj_old = sum(self.objective_kmeans[-w-1:-1])
            obj_new = sum(self.objective_kmeans[-w:])
            return abs(obj_old - obj_new) / abs(obj_old) < self.epsilon_kmeans
        else:  # Harmony
            if len(self.objective_harmony) < 2:
                return False
            obj_old = self.objective_harmony[-2]
            obj_new = self.objective_harmony[-1]
            return (obj_old - obj_new) / abs(obj_old) < self.epsilon_harmony


def run_harmony_jax(
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
    verbose=True,
    random_state=0,
):
    """Run Harmony with JAX acceleration.
    
    This function provides the same interface as run_harmony but uses
    JAX for accelerated computation. The main speedup comes from:
    1. JIT compilation of core functions
    2. Vectorization of the K-cluster loop with vmap
    3. Automatic GPU utilization if available
    
    Args:
        data_mat: Data matrix (cells x features or features x cells)
        meta_data: DataFrame with batch/covariate information
        vars_use: Column name(s) to use for batch correction
        theta: Diversity penalty parameter (default: 1 for each batch)
        lamb: Ridge regression penalty (default: 1 for each batch)
        sigma: Bandwidth for soft clustering (default: 0.1)
        nclust: Number of clusters (default: min(N/30, 100))
        tau: Expected frequencies penalty (default: 0)
        block_size: Fraction of cells per block in update_R (default: 0.05)
        max_iter_harmony: Maximum Harmony iterations (default: 10)
        max_iter_kmeans: Maximum clustering iterations (default: 20)
        epsilon_cluster: Clustering convergence threshold (default: 1e-5)
        epsilon_harmony: Harmony convergence threshold (default: 1e-4)
        verbose: Print progress messages (default: True)
        random_state: Random seed (default: 0)
        
    Returns:
        HarmonyJAX object with corrected data in .Z_corr
    """
    check_jax_available()
    
    # Log device info
    device_info = get_jax_device_info()
    if verbose:
        logger.info(device_info['message'])
    
    # Test if JAX backend works (Metal can have issues)
    backend_works, error_msg = _test_jax_backend()
    if not backend_works:
        if 'UNIMPLEMENTED' in error_msg or 'default_memory_space' in error_msg:
            raise RuntimeError(
                f"JAX Metal backend error: {error_msg}\n\n"
                "The Metal GPU backend is experimental and may not work with all JAX versions.\n"
                "Try one of these solutions:\n"
                "  1. Force CPU mode by setting environment variable before importing JAX:\n"
                "     import os; os.environ['JAX_PLATFORMS'] = 'cpu'\n"
                "  2. Upgrade jax-metal: pip install --upgrade jax-metal\n"
                "  3. Use the NumPy version: harmonypy.run_harmony() instead of run_harmony_jax()"
            )
        else:
            raise RuntimeError(f"JAX backend test failed: {error_msg}")
    
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
    
    phi = pd.get_dummies(meta_data[vars_use]).to_numpy().T
    phi_n = meta_data[vars_use].describe().loc['unique'].to_numpy().astype(int)
    
    if theta is None:
        theta = np.repeat([1] * len(phi_n), phi_n)
    elif isinstance(theta, (float, int)):
        theta = np.repeat([theta] * len(phi_n), phi_n)
    elif len(theta) == len(phi_n):
        theta = np.repeat([theta], phi_n)
    
    assert len(theta) == np.sum(phi_n), "each batch variable must have a theta"
    
    if lamb is None:
        lamb = np.repeat([1] * len(phi_n), phi_n)
    elif isinstance(lamb, (float, int)):
        lamb = np.repeat([lamb] * len(phi_n), phi_n)
    elif len(lamb) == len(phi_n):
        lamb = np.repeat([lamb], phi_n)
    
    assert len(lamb) == np.sum(phi_n), "each batch variable must have a lambda"
    
    N_b = phi.sum(axis=1)
    Pr_b = N_b / N
    
    if tau > 0:
        theta = theta * (1 - np.exp(-(N_b / (nclust * tau)) ** 2))
    
    lamb_mat = np.diag(np.insert(lamb, 0, 0))
    phi_moe = np.vstack((np.repeat(1, N), phi))
    
    np.random.seed(random_state)
    
    ho = HarmonyJAX(
        data_mat, phi, phi_moe, Pr_b, sigma, theta,
        max_iter_harmony, max_iter_kmeans,
        epsilon_cluster, epsilon_harmony, nclust, block_size,
        lamb_mat, verbose, random_state, 'kmeans'
    )
    
    return ho

