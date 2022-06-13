from typing import Tuple, TypeVar

import numpy as np
import structlog
from typeguard import typechecked

logger = structlog.get_logger()
# kmeans does not always return k centroids, but kmeans2 does
from scipy.cluster.vq import kmeans2

THarmony = TypeVar("THarmony", bound="Harmony")


class Harmony(object):
    def __init__(
        self: THarmony,
        Z: np.ndarray,
        Phi: np.ndarray,
        Phi_moe: np.ndarray,
        Pr_b: np.ndarray,
        sigma: np.ndarray,
        theta: np.ndarray,
        max_iter_harmony: int,
        max_iter_kmeans: int,
        epsilon_kmeans: float,
        epsilon_harmony: float,
        K: int,
        block_size: float,
        lamb: np.ndarray,
        verbose: bool,
    ) -> THarmony:
        self.Z_corr = np.array(Z)
        self.Z_orig = np.array(Z)

        self.Z_cos = self.Z_orig / self.Z_orig.max(axis=0)
        self.Z_cos = self.Z_cos / np.linalg.norm(self.Z_cos, ord=2, axis=0)

        self.Phi = Phi
        self.Phi_moe = Phi_moe
        self.N = self.Z_corr.shape[1]
        self.Pr_b = Pr_b
        self.B = self.Phi.shape[0]  # number of batch variables
        self.d = self.Z_corr.shape[0]
        self.window_size = 3
        self.epsilon_kmeans = epsilon_kmeans
        self.epsilon_harmony = epsilon_harmony

        self.lamb = lamb
        self.sigma = sigma
        self.sigma_prior = sigma
        self.block_size = block_size
        self.K = K  # number of clusters
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
        self.init_cluster()
        self.harmonize(self.max_iter_harmony, self.verbose)

    def result(self: THarmony) -> np.ndarray:
        return self.Z_corr

    def allocate_buffers(self: THarmony) -> None:
        self._scale_dist = np.zeros((self.K, self.N))
        self.dist_mat = np.zeros((self.K, self.N))
        self.O = np.zeros((self.K, self.B))
        self.E = np.zeros((self.K, self.B))
        self.W = np.zeros((self.B + 1, self.d))
        self.Phi_Rk = np.zeros((self.B + 1, self.N))

    def init_cluster(self: THarmony) -> None:
        # Start with cluster centroids
        km_centroids, _ = kmeans2(self.Z_cos.T, self.K, minit="++")
        self.Y = km_centroids.T
        # (1) Normalize
        self.Y = self.Y / np.linalg.norm(self.Y, ord=2, axis=0)
        # (2) Assign cluster probabilities
        self.dist_mat = 2 * (1 - np.dot(self.Y.T, self.Z_cos))
        self.R = -self.dist_mat
        self.R = self.R / self.sigma[:, None]
        self.R -= np.max(self.R, axis=0)
        self.R = np.exp(self.R)
        self.R = self.R / np.sum(self.R, axis=0)
        # (3) Batch diversity statistics
        self.E = np.outer(np.sum(self.R, axis=1), self.Pr_b)
        self.O = np.inner(self.R, self.Phi)
        self.compute_objective()
        # Save results
        self.objective_harmony.append(self.objective_kmeans[-1])

    def compute_objective(self: THarmony) -> None:
        kmeans_error = np.sum(np.multiply(self.R, self.dist_mat))
        # Entropy
        _entropy = np.sum(safe_entropy(self.R) * self.sigma[:, np.newaxis])
        # Cross Entropy
        x = self.R * self.sigma[:, np.newaxis]
        y = np.tile(self.theta[:, np.newaxis], self.K).T
        z = np.log((self.O + 1) / (self.E + 1))
        w = np.dot(y * z, self.Phi)
        _cross_entropy = np.sum(x * w)
        # Save results
        self.objective_kmeans.append(kmeans_error + _entropy + _cross_entropy)
        self.objective_kmeans_dist.append(kmeans_error)
        self.objective_kmeans_entropy.append(_entropy)
        self.objective_kmeans_cross.append(_cross_entropy)

    def harmonize(self: THarmony, iter_harmony: int = 10, verbose: bool = True) -> int:
        converged = False
        for i in range(1, iter_harmony + 1):
            if verbose:
                logger.info("Iteration {} of {}".format(i, iter_harmony))
            # STEP 1: Clustering
            self.cluster()
            # STEP 2: Regress out covariates
            # self.moe_correct_ridge()
            self.Z_cos, self.Z_corr, self.W, self.Phi_Rk = moe_correct_ridge(
                self.Z_orig,
                self.Z_cos,
                self.Z_corr,
                self.R,
                self.W,
                self.K,
                self.Phi_Rk,
                self.Phi_moe,
                self.lamb,
            )
            # STEP 3: Check for convergence
            converged = self.check_convergence(1)
            if converged:
                if verbose:
                    logger.info(
                        "Converged after {} iteration{}".format(i, "s" if i > 1 else "")
                    )
                break
        if verbose and not converged:
            logger.info("Stopped before convergence")
        return 0  # not sure why this returns a value at all

    def cluster(self: THarmony) -> int:
        # Z_cos has changed
        # R is assumed to not have changed
        # Update Y to match new integrated data
        self.dist_mat = 2 * (1 - np.dot(self.Y.T, self.Z_cos))
        for i in range(self.max_iter_kmeans):
            # print("kmeans {}".format(i))
            # STEP 1: Update Y
            self.Y = np.dot(self.Z_cos, self.R.T)
            self.Y = self.Y / np.linalg.norm(self.Y, ord=2, axis=0)
            # STEP 2: Update dist_mat
            self.dist_mat = 2 * (1 - np.dot(self.Y.T, self.Z_cos))
            # STEP 3: Update R
            self.update_R()
            # STEP 4: Check for convergence
            self.compute_objective()
            if i > self.window_size:
                converged = self.check_convergence(0)
                if converged:
                    break
        self.kmeans_rounds.append(i)
        self.objective_harmony.append(self.objective_kmeans[-1])
        return 0  # not sure why this returns a value at all

    def update_R(self: THarmony) -> int:
        self._scale_dist = -self.dist_mat
        self._scale_dist = self._scale_dist / self.sigma[:, None]
        self._scale_dist -= np.max(self._scale_dist, axis=0)
        self._scale_dist = np.exp(self._scale_dist)
        # Update cells in blocks
        update_order = np.arange(self.N)
        np.random.shuffle(update_order)
        n_blocks = np.ceil(1 / self.block_size).astype(int)
        blocks = np.array_split(update_order, n_blocks)
        for b in blocks:
            # STEP 1: Remove cells
            self.E -= np.outer(np.sum(self.R[:, b], axis=1), self.Pr_b)
            self.O -= np.dot(self.R[:, b], self.Phi[:, b].T)
            # STEP 2: Recompute R for removed cells
            self.R[:, b] = self._scale_dist[:, b]
            self.R[:, b] = np.multiply(
                self.R[:, b],
                np.dot(
                    np.power((self.E + 1) / (self.O + 1), self.theta), self.Phi[:, b]
                ),
            )
            self.R[:, b] = self.R[:, b] / np.linalg.norm(self.R[:, b], ord=1, axis=0)
            # STEP 3: Put cells back
            self.E += np.outer(np.sum(self.R[:, b], axis=1), self.Pr_b)
            self.O += np.dot(self.R[:, b], self.Phi[:, b].T)
        return 0

    def check_convergence(self: THarmony, i_type: int) -> bool:
        # TODO: change `i_type` to an enum
        obj_old = 0.0
        obj_new = 0.0
        # Clustering, compute new window mean
        if i_type == 0:
            okl = len(self.objective_kmeans)
            for i in range(self.window_size):
                obj_old += self.objective_kmeans[okl - 2 - i]
                obj_new += self.objective_kmeans[okl - 1 - i]
            if abs(obj_old - obj_new) / abs(obj_old) < self.epsilon_kmeans:
                return True
            return False
        # Harmony
        if i_type == 1:
            obj_old = self.objective_harmony[-2]
            obj_new = self.objective_harmony[-1]
            if (obj_old - obj_new) / abs(obj_old) < self.epsilon_harmony:
                return True
            return False
        return True


@typechecked
def safe_entropy(x: np.array) -> np.ndarray:
    y = np.multiply(x, np.log(x))
    y[~np.isfinite(y)] = 0.0
    return y


@typechecked
def moe_correct_ridge(
    Z_orig: np.ndarray,
    Z_cos: np.ndarray,
    Z_corr: np.ndarray,
    R: np.ndarray,
    W: np.ndarray,
    K: np.int64,
    Phi_Rk: np.ndarray,
    Phi_moe: np.ndarray,
    lamb: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Z_corr = Z_orig.copy()
    for i in range(K):
        Phi_Rk = np.multiply(Phi_moe, R[i, :])
        x = np.dot(Phi_Rk, Phi_moe.T) + lamb
        W = np.dot(np.dot(np.linalg.inv(x), Phi_Rk), Z_orig.T)
        W[0, :] = 0  # do not remove the intercept
        Z_corr -= np.dot(W.T, Phi_Rk)
    Z_cos = Z_corr / np.linalg.norm(Z_corr, ord=2, axis=0)
    return Z_cos, Z_corr, W, Phi_Rk
