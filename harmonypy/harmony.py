import pandas as pd
import numpy as np
from scipy.cluster.vq import kmeans

meta_data = pd.read_csv("meta.tsv.gz", sep = "\t")

data_mat = pd.read_csv("pcs.tsv.gz", sep = "\t")
data_mat = np.array(data_mat)

vars_use = ['dataset']

"""
The Harmony algorithm for integrating datasets.
"""

theta = 1
lamb = 0.1
sigma = 0.1
nclust = 50
tau = 0
block_size = 0.05
max_iter_harmony = 10
max_iter_cluster = 200
epsilon_cluster = 1e-5
epsilon_harmony = 1e-4
do_pca = False
return_object = True
verbose = True
reference_values = None
cluster_prior = None

N = meta_data.shape[0]
if data_mat.shape[1] != N:
    data_mat = data_mat.T

if nclust is None:
    nclust = np.min([np.round(N / 30.0), 100]).astype(int)

if theta is None:
    theta = np.repeat(2, len(vars_use))

if lamb is None:
    lamb = np.repeat(1, len(vars_use))

sigma = np.repeat(sigma, nclust)

categories = pd.Categorical(np.squeeze(meta_data[vars_use]))

phi = np.zeros((len(categories.categories), N))
for i in range(len(categories.categories)):
    ix = categories == categories.categories[i]
    phi[i,ix] = 1

N_b = phi.sum(axis = 1)
Pr_b = N_b / N

B_vec = np.array([len(categories.categories)])

theta = np.repeat(theta, B_vec)

lamb = np.repeat(lamb, B_vec)

lamb_mat = np.diag(np.insert(lamb, 0, 0))

phi_moe = np.vstack((np.repeat(1, N), phi))

def safe_entropy(x: np.array):
    y = np.multiply(x, np.log(x))
    y[~np.isfinite(y)] = 0.0
    return y

class Harmony(object):
    def __init__(
            self, __Z, __Phi, __Phi_moe, __Pr_b, __sigma,
            __theta, __max_iter_kmeans, __epsilon_kmeans,
            __epsilon_harmony, __K, tau, __block_size,
            __lambda, __verbose
    ):
        self.Z_corr = np.array(__Z)
        self.Z_orig = np.array(__Z)

        self.Z_cos = self.Z_orig / self.Z_orig.max(axis=0)
        self.Z_cos = self.Z_cos / np.linalg.norm(self.Z_cos, ord=2, axis=0)

        self.Phi             = __Phi;
        self.Phi_moe         = __Phi_moe
        self.N               = self.Z_corr.shape[1]
        self.Pr_b            = __Pr_b
        self.B               = self.Phi.shape[0]
        self.d               = self.Z_corr.shape[0]
        self.window_size     = 3
        self.epsilon_kmeans  = __epsilon_kmeans
        self.epsilon_harmony = __epsilon_harmony

        self.lamb            = __lambda
        self.sigma           = __sigma
        self.sigma_prior     = __sigma
        self.block_size      = __block_size
        self.K               = __K
        self.max_iter_kmeans = __max_iter_kmeans
        self.verbose         = __verbose
        self.theta           = __theta

        self.objective_harmony        = []
        self.objective_kmeans         = []
        self.objective_kmeans_dist    = []
        self.objective_kmeans_entropy = []
        self.objective_kmeans_cross   = []
        self.kmeans_rounds  = []

        self.allocate_buffers()
        self.init_cluster()
        self.harmonize(10, self.verbose)

    def result(self):
        return self.Z_corr

    def allocate_buffers(self):
        self._scale_dist = np.zeros((self.K, self.N))
        self.dist_mat    = np.zeros((self.K, self.N))
        self.O           = np.zeros((self.K, self.B))
        self.E           = np.zeros((self.K, self.B))
        self.W           = np.zeros((self.B + 1, self.d))
        self.Phi_Rk      = np.zeros((self.B + 1, self.N))

    def init_cluster(self):
        # Start with cluster centroids
        km = kmeans(self.Z_cos.T, self.K, iter=10)
        self.Y = km[0].T
        # (1) Normalize
        self.Y = self.Y / np.linalg.norm(self.Y, ord=2, axis=0)
        # (2) Assign cluster probabilities
        self.dist_mat = 2 * (1 - np.dot(self.Y.T, self.Z_cos))
        self.R = -self.dist_mat
        self.R = self.R / self.sigma[:,None]
        self.R -= np.max(self.R, axis = 0)
        self.R = np.exp(self.R)
        self.R = self.R / np.sum(self.R, axis = 0)
        # (3) Batch diversity statistics
        self.E = np.outer(np.sum(self.R, axis=1), self.Pr_b)
        self.O = np.inner(self.R , self.Phi)
        self.compute_objective()
        # Save results
        self.objective_harmony.append(self.objective_kmeans[-1])

    def compute_objective(self):
        kmeans_error = np.sum(np.multiply(self.R, self.dist_mat))
        # Entropy
        _entropy = np.sum(safe_entropy(ho.R) * sigma[:,np.newaxis])
        # Cross Entropy
        x = (ho.R * sigma[:,np.newaxis])
        y = np.tile(ho.theta[:,np.newaxis], ho.K).T
        z = np.log((ho.O + 1) / (ho.E + 1))
        w = np.dot(y * z, ho.Phi)
        _cross_entropy = np.sum(x * w)
        # Save results
        self.objective_kmeans.append(kmeans_error + _entropy + _cross_entropy)
        self.objective_kmeans_dist.append(kmeans_error)
        self.objective_kmeans_entropy.append(_entropy)
        self.objective_kmeans_cross.append(_cross_entropy)

    def harmonize(self, iter_harmony=10, verbose=True):
        for i in range(1, iter_harmony + 1):
            if verbose:
                print("Harmony {}/{}".format(i, iter_harmony))
            # STEP 1: Clustering
            self.cluster()
            # STEP 2: Regress out covariates
            self.moe_correct_ridge()
            # STEP 3: Check for convergence
            if self.check_convergence(1):
                if verbose:
                    print("Harmony converged after {} iteration{}".format(i, 's' if i > 1 else ''))
                break
        return 0

    def cluster(self):
        # Z_cos has changed
        # R is assumed to not have changed
        # Update Y to match new integrated data
        self.dist_mat = 2 * (1 - np.dot(self.Y.T, self.Z_cos))
        for i in range(self.max_iter_kmeans):
            print("kmeans {}".format(i))
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
        return 0

    def update_R(self):
        update_order = np.arange(self.N)
        np.random.shuffle(update_order)
        self._scale_dist = -self.dist_mat
        self._scale_dist = self._scale_dist / self.sigma[:,None]
        self._scale_dist -= np.max(self._scale_dist, axis=0)
        self._scale_dist = np.exp(self._scale_dist)
        # Update cells in blocks
        n_blocks = np.ceil(1 / self.block_size).astype(int)
        blocks = np.array_split(update_order, n_blocks)
        for b in blocks:
            # STEP 1: Remove cells
            self.E -= np.outer(np.sum(self.R[:,b], axis=1), self.Pr_b)
            self.O -= np.dot(self.R[:,b], self.Phi[:,b].T)
            # STEP 2: Recompute R for removed cells
            self.R[:,b] = self._scale_dist[:,b]
            self.R[:,b] = np.multiply(
                self.R[:,b],
                np.dot(
                    np.power((self.E + 1) / (self.O + 1), self.theta),
                    self.Phi[:,b]
                )
            )
            self.R[:,b] = self.R[:,b] / np.linalg.norm(self.R[:,b], ord=1, axis=0)
            # STEP 3: Put cells back
            self.E += np.outer(np.sum(self.R[:,b], axis=1), self.Pr_b)
            self.O += np.dot(self.R[:,b], self.Phi[:,b].T)
        return 0

    def moe_correct_ridge(self):
        self.Z_corr = self.Z_orig
        for i in range(self.K):
            self.Phi_Rk = np.dot(self.Phi_moe, np.diag(self.R[i,:]))
            x = np.dot(self.Phi_Rk, self.Phi_moe.T) + self.lamb
            self.W = np.dot(np.dot(np.linalg.inv(x), self.Phi_Rk), self.Z_orig.T)
            self.W[0,:] = 0 # do not remove the intercept
            self.Z_corr -= np.dot(self.W.T, self.Phi_Rk)
        self.Z_cos = self.Z_corr / np.linalg.norm(self.Z_corr, ord=2, axis=0)

    def check_convergence(self, i_type):
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
            if abs(obj_old - obj_new) / abs(obj_old) < self.epsilon_harmony:
                return True
            return False
        return True

ho = Harmony(
    data_mat, phi, phi_moe, Pr_b, sigma, theta, max_iter_cluster,
    epsilon_cluster, epsilon_harmony, nclust, tau, block_size, lamb_mat, verbose
)


# Tests

np.all(np.equal(ho.Y.shape, (ho.d, ho.K)))

np.all(np.equal(ho.Z_corr.shape, (ho.d, ho.N)))

np.all(np.equal(ho.Z_cos.shape, (ho.d, ho.N)))

np.all(np.equal(ho.R.shape, (ho.K, ho.N)))


import pandas as pd

res = pd.DataFrame(ho.Z_corr)
res.columns = ['X{}'.format(i + 1) for i in range(res.shape[1])]
res.to_csv("adj.tsv.gz", sep = "\t", index = False)

