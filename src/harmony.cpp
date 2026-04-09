// harmonypy - C++ backend matching R harmony2 package.
// Copyright (C) 2018  Ilya Korsunsky
//               2019  Kamil Slowikowski <kslowikowski@gmail.com>
//
// Follows the R harmony2 algorithm step-by-step, using sparse Phi
// matrices for BLAS-accelerated batch operations.

#include "harmony.hpp"
#include <numeric>
#include <set>

namespace harmony {

// K-means initialization (matches R harmony2: initialize_centroids + arma::kmeans)
//
// R approach:
//   1. Pick K random columns from X as initial centroids
//   2. For each centroid i, compute cosine distance to all cells,
//      then use Gumbel-max trick (-log(U)/dist) to sample a replacement
//   3. Refine with arma::kmeans for 10 iterations
//
// X is d x N (columns are L2-normalized cells), returns d x K centroids.
MATTYPE kmeans_init(const MATTYPE& X, int K, std::mt19937& rng) {
    int N = X.n_cols;

    // Step 1: Pick K random columns as initial centroids (matches R's randu seeds)
    std::uniform_real_distribution<float> uniform01(0.0f, 1.0f);
    MATTYPE Y(X.n_rows, K);
    for (int i = 0; i < K; ++i) {
        int idx = static_cast<int>(std::round(uniform01(rng) * N));
        if (idx >= N) idx = N - 1;
        Y.col(i) = X.col(idx);
    }

    // Step 2: Gumbel-max weighted sampling (matches R's initialize_centroids)
    std::set<unsigned> chosen;
    for (int i = 0; i < K; ++i) {
        // Cosine distance from centroid i to all cells
        VECTYPE distances = arma::abs((2.0f * (1.0f - Y.col(i).t() * X)).as_col());

        // Gumbel-max trick: prob = -log(U) / dist, pick argmin
        VECTYPE random_numbers(N, arma::fill::none);
        for (int j = 0; j < N; ++j) random_numbers(j) = uniform01(rng);
        VECTYPE prob = -arma::log(random_numbers) / (distances + 1e-10f);

        // Avoid re-selecting the same point
        for (auto idx : chosen) prob(idx) = prob.max();
        unsigned best = prob.index_min();
        while (chosen.count(best)) {
            prob(best) = prob.max();
            best = prob.index_min();
        }
        chosen.insert(best);
        Y.col(i) = X.col(best);
    }

    // Step 3: Refine with arma::kmeans for 10 iterations (matches R)
    for (int i = 0; i < 10; ++i) {
        arma::kmeans(Y, X, K, arma::keep_existing, 1, false);
    }

    return Y;
}

// Constructor
Harmony::Harmony(
    const arma::mat& Z,
    const arma::sp_mat& Phi_in,
    const arma::vec& Pr_b_in,
    const arma::vec& sigma_in,
    const arma::vec& theta_in,
    const arma::vec& lambda_in,
    double alpha_in,
    int max_iter_harmony,
    int max_iter_kmeans,
    double epsilon_kmeans,
    double epsilon_harmony,
    int K,
    double block_size,
    const std::vector<int>& B_vec_in,
    double batch_proportion_cutoff,
    bool verbose,
    int random_state
) : max_iter_harmony(max_iter_harmony),
    max_iter_kmeans(max_iter_kmeans),
    epsilon_kmeans(static_cast<float>(epsilon_kmeans)),
    epsilon_harmony(static_cast<float>(epsilon_harmony)),
    K(K),
    block_size(static_cast<float>(block_size)),
    verbose(verbose),
    window_size(3),
    alpha(static_cast<float>(alpha_in)),
    batch_proportion_cutoff(static_cast<float>(batch_proportion_cutoff)),
    B_vec(B_vec_in),
    rng(random_state)
{
    Z_orig = arma::conv_to<MATTYPE>::from(Z);
    Z_corr = arma::normalise(Z_orig, 2, 0);

    Pr_b = arma::conv_to<VECTYPE>::from(Pr_b_in);
    N = Z.n_cols;
    d = Z.n_rows;
    B = Phi_in.n_rows;

    sigma = arma::conv_to<VECTYPE>::from(sigma_in);
    theta = arma::conv_to<VECTYPE>::from(theta_in);

    if (lambda_in(0) < 0) {
        lambda_estimation = true;
        lambda.zeros(B + 1);
    } else {
        lambda_estimation = false;
        lambda = arma::conv_to<VECTYPE>::from(lambda_in);
    }

    // Covariate bounds (matches R: partial_sum)
    if (B_vec.size() > 1) {
        covariate_bounds.resize(B_vec.size() - 1);
        std::partial_sum(B_vec.begin(), B_vec.end(), covariate_bounds.begin());
    } else {
        covariate_bounds.push_back(B_vec.front());
    }

    build_batch_structures(Phi_in);
    allocate_buffers();

    if (verbose) std::cout << "Computing initial centroids..." << std::endl;
    init_cluster();
    if (verbose) std::cout << "Initialization complete." << std::endl;
    harmonize(max_iter_harmony, verbose);
}

void Harmony::build_batch_structures(const arma::sp_mat& Phi_in) {
    // Store sparse Phi (B x N) and its transpose for BLAS operations
    Phi = arma::conv_to<SPMAT>::from(Phi_in);
    Phi_t = Phi.t();

    // Build Phi_moe: (B+1) x N with intercept row of ones
    SPMAT intcpt = arma::zeros<SPMAT>(1, N);
    intcpt = intcpt + 1;
    Phi_moe = arma::join_cols(intcpt, Phi);
    Phi_moe_t = Phi_moe.t();

    // Build per-batch cell index lists (for ridge correction)
    arma::uvec batch_sizes_u = arma::conv_to<arma::uvec>::from(VECTYPE(arma::sum(Phi, 1)));
    std::vector<unsigned> counters(B, 0);
    batch_index.resize(B);
    for (int b = 0; b < B; ++b) {
        batch_index[b].zeros(batch_sizes_u(b));
    }
    typename arma::sp_mat::const_iterator it = Phi_in.begin();
    typename arma::sp_mat::const_iterator it_end = Phi_in.end();
    for (; it != it_end; ++it) {
        unsigned row_idx = it.row();
        unsigned col_idx = it.col();
        batch_index[row_idx](counters[row_idx]++) = col_idx;
    }
}

void Harmony::allocate_buffers() {
    dist_mat.zeros(K, N);
    O.zeros(K, B);
    E.zeros(K, B);
    W.zeros(B + 1, d);
    R.zeros(K, N);
    Y.zeros(d, K);
}

void Harmony::init_cluster() {
    // Matches R: Y = kmeans_centers(Z_corr, K)
    // kmeans_init works on d x N, returns d x K
    Y = kmeans_init(Z_corr, K, rng);
    Y = arma::normalise(Y, 2, 0);

    dist_mat = 2.0f * (1.0f - Y.t() * Z_corr);

    R = -dist_mat;
    R.each_col() /= sigma;
    R = arma::exp(R);
    R.each_row() /= arma::sum(R, 0);

    // Matches R: E = sum(R,1) * Pr_b.t(); O = R * Phi_t;
    E = arma::sum(R, 1) * Pr_b.t();
    O = R * Phi_t;

    compute_objective();
    objective_harmony.push_back(objective_kmeans.back());
}

void Harmony::compute_objective() {
    // Matches R: compute_objective
    // Uses O (K×B) instead of R's K×N Phi matmul to avoid ~327 MB temporary.
    // Mathematically identical: sum_j(R_kj * (theta_log * Phi)_kj) = sum_b(O_kb * theta_log_kb)
    const float norm_const = 2000.0f / static_cast<float>(N);

    // K-means error: sum(R % dist_mat)
    float kmeans_error = arma::accu(R % dist_mat);

    // Entropy: sum(xlogy(R, R) .each_col() % sigma)
    MATTYPE log_R = R;
    log_R.transform([](float val) { return val > 0 ? val * std::log(val) : 0.0f; });
    float _entropy = arma::as_scalar(arma::accu(log_R.each_col() % sigma));

    // Cross entropy via O (K×B) — equivalent to R's (R % sigma) % (theta_log * Phi)
    MATTYPE ratio = (O + E + 1) / (2 * E + 1);
    ratio.transform([](float val) { return std::log(val); });
    ratio.each_row() %= theta.t();
    ratio.each_col() %= sigma;
    ratio %= O;
    float _cross_entropy = arma::accu(ratio);

    objective_kmeans.push_back((kmeans_error + _entropy + _cross_entropy) * norm_const);
    objective_kmeans_dist.push_back(kmeans_error * norm_const);
    objective_kmeans_entropy.push_back(_entropy * norm_const);
    objective_kmeans_cross.push_back(_cross_entropy * norm_const);
}

void Harmony::harmonize(int iter_harmony, bool verbose_flag) {
    bool converged = false;
    for (int i = 1; i <= iter_harmony; ++i) {
        if (verbose_flag)
            std::cout << "Iteration " << i << " of " << iter_harmony << std::endl;

        cluster();
        moe_correct_ridge();

        converged = check_convergence(1);
        if (converged) {
            if (verbose_flag)
                std::cout << "Converged after " << i << " iteration"
                          << (i > 1 ? "s" : "") << std::endl;
            break;
        }
    }
    if (verbose_flag && !converged)
        std::cout << "Stopped before convergence" << std::endl;
}

void Harmony::cluster() {
    // Matches R: cluster_cpp
    if (objective_harmony.size() > 1) {
        Z_corr = arma::normalise(Z_corr, 2, 0);
        dist_mat = 2.0f * (1.0f - Y.t() * Z_corr);
        R = -dist_mat;
        R.each_col() /= sigma;
        R = arma::exp(R);
        R.each_row() /= arma::sum(R, 0);
        E = arma::sum(R, 1) * Pr_b.t();
        O = R * Phi_t;
    }

    int rounds = 0;
    for (int i = 0; i < max_iter_kmeans; ++i) {
        update_R();
        compute_objective();

        if (i > window_size) {
            if (check_convergence(0)) {
                rounds = i + 1;
                break;
            }
        }
        rounds = i + 1;
    }

    kmeans_rounds.push_back(rounds);
    objective_harmony.push_back(objective_kmeans.back());
}

void Harmony::update_R() {
    // Matches R: update_R — uses sparse Phi for O updates and diversity

    std::vector<unsigned> indices_vec(N);
    std::iota(indices_vec.begin(), indices_vec.end(), 0);
    std::shuffle(indices_vec.begin(), indices_vec.end(), rng);
    arma::uvec update_order(N);
    for (int i = 0; i < N; ++i) update_order(i) = indices_vec[i];

    arma::uvec indices = arma::linspace<arma::uvec>(0, N - 1, N);

    arma::uvec reverse_index(N, arma::fill::zeros);
    reverse_index.rows(update_order) = indices;

    unsigned n_blocks = static_cast<unsigned>(std::ceil(1.0 / block_size));
    unsigned cells_per_block = std::max(1u, static_cast<unsigned>(N * block_size));

    // Shuffle R, dist_mat, and Phi
    R = R.cols(update_order);
    dist_mat = dist_mat.cols(update_order);
    SPMAT Phi_rand(Phi.cols(update_order));
    SPMAT Phi_t_rand(Phi_rand.t());

    for (unsigned i = 0; i < n_blocks; ++i) {
        unsigned idx_min = i * cells_per_block;
        unsigned idx_max = ((i + 1) * cells_per_block) - 1;
        if (i == n_blocks - 1) idx_max = N - 1;
        if (idx_min >= static_cast<unsigned>(N)) break;

        auto Rcells = R.submat(0, idx_min, R.n_rows - 1, idx_max);
        auto Phicells = Phi_rand.submat(0, idx_min, Phi_rand.n_rows - 1, idx_max);
        auto Phi_tcells = Phi_t_rand.submat(idx_min, 0, idx_max, Phi_t_rand.n_cols - 1);
        auto dist_matcells = dist_mat.submat(0, idx_min, dist_mat.n_rows - 1, idx_max);

        // Step 1: remove cells (sparse matmul for O)
        E -= arma::sum(Rcells, 1) * Pr_b.t();
        O -= Rcells * Phi_tcells;

        // Step 2: recompute R for this block
        Rcells = -dist_matcells;
        Rcells.each_col() /= sigma;
        Rcells = arma::exp(Rcells);
        Rcells = arma::normalise(Rcells, 1, 0);

        // Apply diversity penalty (sparse matmul gathers per-cell batch values)
        Rcells = Rcells % (harmony_pow(((2*E) + 1) / (O + E + 1), theta) * Phicells);
        Rcells = arma::normalise(Rcells, 1, 0);

        // Step 3: put cells back (sparse matmul for O)
        E += arma::sum(Rcells, 1) * Pr_b.t();
        O += Rcells * Phi_tcells;
    }

    // Unshuffle
    R = R.cols(reverse_index);
    dist_mat = dist_mat.cols(reverse_index);
}

bool Harmony::check_convergence(int i_type) {
    // Matches R: check_convergence
    if (i_type == 0) {
        if (objective_kmeans.size() <= static_cast<size_t>(window_size + 1))
            return false;

        float obj_old = 0.0f, obj_new = 0.0f;
        size_t n = objective_kmeans.size();
        for (int i = 0; i < window_size; ++i) {
            obj_old += objective_kmeans[n - 2 - i];
            obj_new += objective_kmeans[n - 1 - i];
        }
        return std::abs(obj_old - obj_new) / std::abs(obj_old) < epsilon_kmeans;
    }

    if (i_type == 1) {
        if (objective_harmony.size() < 2) return false;
        float obj_old = objective_harmony[objective_harmony.size() - 2];
        float obj_new = objective_harmony[objective_harmony.size() - 1];
        return (obj_old - obj_new) / std::abs(obj_old) < epsilon_harmony;
    }
    return true;
}

void Harmony::moe_correct_ridge() {
    // Matches R: moe_correct_ridge_cpp
    Z_corr = Z_orig;

    VECTYPE sizes(arma::sum(Phi, 1));

    for (int k = 0; k < K; ++k) {
        VECTYPE avg_R = O.row(k).t() / sizes;

        // Determine which batches qualify (matches R covariate_bounds logic)
        std::vector<unsigned> keep;
        std::vector<unsigned> cov_levels(B_vec.size(), 0);

        for (unsigned b = 0, current_cov = 0; b < static_cast<unsigned>(B); ++b) {
            if (current_cov < covariate_bounds.size() && !(b < covariate_bounds[current_cov]))
                current_cov++;
            if (arma::as_scalar(avg_R.row(b)) > batch_proportion_cutoff)
                cov_levels[current_cov]++;
        }

        unsigned active_covariates = 0;
        for (auto const& l : cov_levels) {
            if (l > 1) active_covariates++;
        }

        for (unsigned b = 0, current_cov = 0; b < static_cast<unsigned>(B); ++b) {
            if (current_cov < covariate_bounds.size() && !(b < covariate_bounds[current_cov]))
                current_cov++;
            if (arma::as_scalar(avg_R.row(b)) > batch_proportion_cutoff && cov_levels[current_cov] > 1)
                keep.push_back(b);
        }

        if (active_covariates == 0) continue;

        arma::uvec keep_batch = arma::conv_to<arma::uvec>::from(keep);

        // Pointers for either full or subset path (matches R's pointer approach)
        MATTYPE *_Z_corr, *_Z_tmp;
        SPMAT *_Phi_moe, *_Phi_moe_t, *_lambda_mat, *_Rk;
        std::vector<arma::uvec>* _index;
        bool subset_data = false;

        if (keep.size() == static_cast<size_t>(B)) {
            // All batches qualify — use full data
            _Z_corr = &(this->Z_corr);
            _Z_tmp = new MATTYPE(Z_orig);
            _Phi_moe = &(this->Phi_moe);
            _Phi_moe_t = &(this->Phi_moe_t);
            _Rk = new SPMAT(N, N);
            _Rk->diag() = R.row(k);
            _index = &(this->batch_index);
            _lambda_mat = new SPMAT(B + 1, B + 1);

            if (lambda_estimation)
                _lambda_mat->diag() = find_lambda(alpha, VECTYPE(E.row(k).t()));
            else
                _lambda_mat->diag() = lambda;
        } else {
            // Subset to qualifying batches
            subset_data = true;

            // Collect qualifying cell indices
            std::vector<unsigned> keep_cols_scratch;
            keep_cols_scratch.reserve(N);
            for (auto b : keep) {
                keep_cols_scratch.insert(keep_cols_scratch.end(),
                    batch_index[b].memptr(), batch_index[b].memptr() + batch_index[b].n_rows);
            }

            std::set<unsigned> keep_cols_set(keep_cols_scratch.begin(), keep_cols_scratch.end());
            arma::uvec keep_cols = arma::conv_to<arma::uvec>::from(
                std::vector<unsigned>(keep_cols_set.begin(), keep_cols_set.end()));

            // Map old cell indices to new contiguous indices
            std::vector<int> cell_map(N, -1);
            unsigned idx = 0;
            for (auto c : keep_cols_set) cell_map[c] = idx++;

            unsigned n_keep = keep.size();
            unsigned n_cells = keep_cols.n_elem;
            unsigned PhiNonZero = n_cells + keep_cols_scratch.size();

            // Build new sparse Phi_moe_t
            arma::uvec rowind_new(PhiNonZero);
            arma::uvec indptr_new(n_keep + 2);
            rowind_new.subvec(0, n_cells - 1) = arma::linspace<arma::uvec>(0, n_cells - 1, n_cells);
            indptr_new[0] = 0;
            indptr_new[1] = n_cells;

            const arma::uword* rowind_old = Phi_moe_t.row_indices;
            const arma::uword* indptr_old = Phi_moe_t.col_ptrs;

            _index = new std::vector<arma::uvec>();
            for (unsigned i = 0; i < n_keep; ++i) {
                unsigned batch_id = keep[i];
                unsigned cell_offset = 0;
                unsigned max_idx = indptr_old[batch_id + 2], min_idx = indptr_old[batch_id + 1];
                unsigned base_range = indptr_new(i + 1);

                for (unsigned j = min_idx; j < max_idx; ++j) {
                    int new_index = cell_map[rowind_old[j]];
                    if (new_index >= 0) {
                        rowind_new(base_range + cell_offset++) = new_index;
                    }
                }
                indptr_new(i + 2) = base_range + cell_offset;
                _index->push_back(rowind_new.subvec(base_range, indptr_new(i + 2) - 1));
            }

            _Z_corr = new MATTYPE(this->Z_corr.cols(keep_cols));
            _Z_tmp = new MATTYPE(this->Z_orig.cols(keep_cols));

            _Phi_moe_t = new SPMAT(rowind_new, indptr_new,
                VECTYPE(rowind_new.n_elem, arma::fill::ones),
                n_cells, n_keep + 1);
            _Phi_moe = new SPMAT(_Phi_moe_t->t());

            _Rk = new SPMAT(n_cells, n_cells);
            VECTYPE _Rvec(R.row(k).as_col());
            _Rk->diag() = _Rvec.rows(keep_cols);

            _lambda_mat = new SPMAT(n_keep + 1, n_keep + 1);
            if (lambda_estimation) {
                VECTYPE Esub = VECTYPE(E.row(k).t());
                Esub = Esub.rows(keep_batch);
                _lambda_mat->diag() = find_lambda(alpha, Esub);
            } else {
                VECTYPE ltmp(n_keep + 1);
                ltmp(0) = 0;
                ltmp.subvec(1, n_keep) = lambda.rows(keep_batch + 1);
                _lambda_mat->diag() = ltmp;
            }
        }

        // References for cleaner code
        MATTYPE& Zc = *_Z_corr;
        SPMAT& Pm = *_Phi_moe;
        SPMAT& Pmt = *_Phi_moe_t;
        SPMAT& lam = *_lambda_mat;
        SPMAT& Rk = *_Rk;
        MATTYPE& Zt = *_Z_tmp;
        std::vector<arma::uvec>& idx = *_index;

        // Phi_Rk = Phi_moe * Rk  (sparse matmul)
        SPMAT Phi_Rk = Pm * Rk;

        // Phi_cov = Phi_Rk * Phi_moe_t + lambda
        MATTYPE Phi_cov = MATTYPE(Phi_Rk * Pmt) + MATTYPE(lam);

        // Invert covariance
        MATTYPE inv_cov;
        if (B_vec.size() > 1) {
            inv_cov = arma::inv(Phi_cov);
        } else {
            // Arrowhead inverse (matches R)
            VECTYPE ac = -Phi_cov.row(0).as_col();
            ac(0) = 1;
            float b0 = Phi_cov(0, 0);
            VECTYPE b = 1.0f / Phi_cov.diag();
            b(0) = 0;
            float u = b0 - arma::accu(arma::square(ac) % b);
            VECTYPE ac_b = ac % b;
            ac_b(0) = 1;
            inv_cov = (1.0f / u) * (ac_b * ac_b.t());
            inv_cov.diag() += b;
        }

        // Pre-scale Z_tmp by Rk (matches R: Z_tmp.each_row() % Rk.diag())
        Zt = Zt.each_row() % VECTYPE(Rk.diag()).as_row();

        // W = inv_cov[:,0] * sum(Z_tmp)' (intercept contribution)
        W = inv_cov.unsafe_col(0) * arma::sum(Zt, 1).t();

        // Per-batch contribution
        for (unsigned b = 0; b < idx.size(); ++b) {
            W += inv_cov.unsafe_col(b + 1) * arma::sum(Zt.cols(idx[b]), 1).t();
        }

        Y.col(k) = W.row(0).t();
        W.row(0).zeros();

        // Apply correction: Z_corr -= W' * Phi_Rk (single sparse matmul)
        Zc -= W.t() * Phi_Rk;

        if (subset_data) {
            // Write corrected subset back to full Z_corr
            std::set<unsigned> keep_cols_set;
            for (auto b : keep) {
                keep_cols_set.insert(batch_index[b].memptr(),
                    batch_index[b].memptr() + batch_index[b].n_rows);
            }
            arma::uvec keep_cols = arma::conv_to<arma::uvec>::from(
                std::vector<unsigned>(keep_cols_set.begin(), keep_cols_set.end()));
            this->Z_corr.cols(keep_cols) = Zc;

            delete _Z_corr;
            delete _Phi_moe;
            delete _Phi_moe_t;
            delete _index;
        }

        delete _lambda_mat;
        delete _Rk;
        delete _Z_tmp;
    }

    Y = arma::normalise(Y, 2, 0);
}

} // namespace harmony
