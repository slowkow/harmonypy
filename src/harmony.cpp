// harmonypy - C++ backend matching R harmony2 package.
// Copyright (C) 2018  Ilya Korsunsky
//               2019  Kamil Slowikowski <kslowikowski@gmail.com>
//
// Uses float precision and minimal memory layout to match R package.
// No separate Z_cos (Z_corr is normalized in-place).
// No separate _scale_dist (computed inline per block in update_R).

#include "harmony.hpp"
#include <numeric>
#include <set>

#ifdef HARMONY_USE_OPENMP
#include <omp.h>
#endif

namespace harmony {

constexpr int PARALLEL_THRESHOLD = 50000;

// K-means++ initialization
MATTYPE kmeans_plusplus(const MATTYPE& data, int K, std::mt19937& rng) {
    int N = data.n_rows;
    int d_dim = data.n_cols;

    MATTYPE centroids(K, d_dim);
    std::vector<bool> chosen(N, false);

    std::uniform_int_distribution<int> uniform(0, N - 1);
    int first = uniform(rng);
    centroids.row(0) = data.row(first);
    chosen[first] = true;

    VECTYPE min_distances(N, arma::fill::value(std::numeric_limits<float>::max()));

    for (int k = 1; k < K; ++k) {
        #ifdef HARMONY_USE_OPENMP
        #pragma omp parallel for schedule(static) if(N > PARALLEL_THRESHOLD)
        #endif
        for (int i = 0; i < N; ++i) {
            if (!chosen[i]) {
                auto diff = data.row(i) - centroids.row(k-1);
                float dist = arma::dot(diff, diff);
                min_distances(i) = std::min(min_distances(i), dist);
            }
        }

        // discrete_distribution needs double iterators
        arma::vec min_dist_d = arma::conv_to<arma::vec>::from(min_distances);
        std::discrete_distribution<int> weighted_dist(
            min_dist_d.begin(), min_dist_d.end()
        );

        int next = weighted_dist(rng);
        while (chosen[next]) next = uniform(rng);

        centroids.row(k) = data.row(next);
        chosen[next] = true;
        min_distances(next) = 0;
    }

    // Refine with 25 iterations of k-means
    for (int iter = 0; iter < 25; ++iter) {
        std::vector<int> assignments(N);
        #ifdef HARMONY_USE_OPENMP
        #pragma omp parallel for schedule(static) if(N > PARALLEL_THRESHOLD)
        #endif
        for (int i = 0; i < N; ++i) {
            float min_dist = std::numeric_limits<float>::max();
            int best_k = 0;
            for (int kk = 0; kk < K; ++kk) {
                auto diff = data.row(i) - centroids.row(kk);
                float dist = arma::dot(diff, diff);
                if (dist < min_dist) { min_dist = dist; best_k = kk; }
            }
            assignments[i] = best_k;
        }

        MATTYPE new_centroids(K, d_dim, arma::fill::zeros);
        VECTYPE counts(K, arma::fill::zeros);
        for (int i = 0; i < N; ++i) {
            new_centroids.row(assignments[i]) += data.row(i);
            counts(assignments[i]) += 1;
        }
        for (int kk = 0; kk < K; ++kk) {
            if (counts(kk) > 0) new_centroids.row(kk) /= counts(kk);
            else new_centroids.row(kk) = data.row(uniform(rng));
        }

        float change = arma::norm(new_centroids - centroids, "fro");
        centroids = new_centroids;
        if (change < 1e-6f) break;
    }

    return centroids;
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
    // Convert double inputs to float
    Z_orig = arma::conv_to<MATTYPE>::from(Z);

    // Z_corr = L2-normalized Z_orig (no separate Z_cos — matches R)
    Z_corr = arma::normalise(Z_orig, 2, 0);

    // Sparse batch indicators (convert to float sparse)
    Phi = arma::conv_to<SPMAT>::from(Phi_in);
    Phi_t = Phi.t();

    // Phi_moe: intercept row + Phi
    SPMAT intercept(1, Phi.n_cols);
    for (unsigned i = 0; i < Phi.n_cols; ++i) intercept(0, i) = 1.0f;
    Phi_moe = arma::join_cols(intercept, Phi);
    Phi_moe_t = Phi_moe.t();

    Pr_b = arma::conv_to<VECTYPE>::from(Pr_b_in);
    N = Z.n_cols;
    d = Z.n_rows;
    B = Phi.n_rows;

    sigma = arma::conv_to<VECTYPE>::from(sigma_in);
    theta = arma::conv_to<VECTYPE>::from(theta_in);

    if (lambda_in(0) < 0) {
        lambda_estimation = true;
        lambda.zeros(B + 1);
    } else {
        lambda_estimation = false;
        lambda = arma::conv_to<VECTYPE>::from(lambda_in);
    }

    // Covariate bounds
    covariate_bounds.resize(B_vec.size());
    unsigned cumsum = 0;
    for (unsigned i = 0; i < B_vec.size(); ++i) {
        cumsum += B_vec[i];
        covariate_bounds[i] = cumsum;
    }

    build_batch_index();
    allocate_buffers();

    if (verbose) std::cout << "Computing initial centroids..." << std::endl;
    init_cluster();
    if (verbose) std::cout << "Initialization complete." << std::endl;
    harmonize(max_iter_harmony, verbose);
}

void Harmony::build_batch_index() {
    VECTYPE sizes = VECTYPE(arma::sum(Phi, 1));
    std::vector<unsigned> counters(B, 0);
    batch_index.resize(B);
    for (int b = 0; b < B; ++b) {
        batch_index[b].zeros(static_cast<unsigned>(sizes(b)));
    }
    typename SPMAT::const_iterator it = Phi.begin();
    typename SPMAT::const_iterator it_end = Phi.end();
    for (; it != it_end; ++it) {
        unsigned row_idx = it.row();
        unsigned col_idx = it.col();
        batch_index[row_idx](counters[row_idx]++) = col_idx;
    }
}

void Harmony::allocate_buffers() {
    // No _scale_dist — computed inline per block (matches R)
    dist_mat.zeros(K, N);
    O.zeros(K, B);
    E.zeros(K, B);
    W.zeros(B + 1, d);
    R.zeros(K, N);
    Y.zeros(d, K);
}

void Harmony::init_cluster() {
    // Z_corr is already L2-normalized from constructor
    MATTYPE centroids = kmeans_plusplus(Z_corr.t(), K, rng);
    Y = centroids.t();
    Y = arma::normalise(Y, 2, 0);

    dist_mat = 2.0f * (1.0f - Y.t() * Z_corr);

    R = -dist_mat;
    R.each_col() /= sigma;
    R.transform([](float val) { return std::exp(val); });
    R.each_row() /= arma::sum(R, 0);

    E = VECTYPE(arma::sum(R, 1)) * Pr_b.t();
    O = R * Phi_t;

    compute_objective();
    objective_harmony.push_back(objective_kmeans.back());
}

void Harmony::compute_objective() {
    const float norm_const = 2000.0f / static_cast<float>(N);

    float kmeans_error = arma::accu(R % dist_mat);

    // Entropy: sum(xlogy(R, R) * sigma)
    MATTYPE log_R = R;
    log_R.transform([](float val) { return val > 0 ? std::log(val) : 0.0f; });
    MATTYPE entropy_mat = R % log_R;
    entropy_mat.each_col() %= sigma;
    float _entropy = arma::accu(entropy_mat);

    // Cross entropy (harmony2 formula)
    MATTYPE R_sigma = R;
    R_sigma.each_col() %= sigma;
    MATTYPE ratio = (O + E + 1) / (2 * E + 1);
    MATTYPE log_ratio = ratio;
    log_ratio.transform([](float val) { return std::log(val); });
    MATTYPE theta_log = arma::repmat(theta.t(), K, 1) % log_ratio;
    float _cross_entropy = arma::accu(R_sigma % (theta_log * Phi));

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
    // harmony2: cold-start R re-estimation after correction
    // Z_corr was updated by moe_correct_ridge, re-normalize in-place
    if (objective_harmony.size() > 1) {
        Z_corr = arma::normalise(Z_corr, 2, 0);
        dist_mat = 2.0f * (1.0f - Y.t() * Z_corr);
        R = -dist_mat;
        R.each_col() /= sigma;
        R.transform([](float val) { return std::exp(val); });
        R.each_row() /= arma::sum(R, 0);
        E = VECTYPE(arma::sum(R, 1)) * Pr_b.t();
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
    // Matches R package: shuffle R and dist_mat as references (in-place),
    // compute softmax from dist_mat per block (no separate _scale_dist buffer)

    // Shuffle order
    std::vector<unsigned> indices_vec(N);
    std::iota(indices_vec.begin(), indices_vec.end(), 0);
    std::shuffle(indices_vec.begin(), indices_vec.end(), rng);
    arma::uvec update_order(N);
    for (int i = 0; i < N; ++i) update_order(i) = indices_vec[i];

    arma::uvec indices = arma::linspace<arma::uvec>(0, N - 1, N);
    arma::uvec reverse_index(N, arma::fill::zeros);
    reverse_index.elem(update_order) = indices;

    unsigned n_blocks = static_cast<unsigned>(std::ceil(1.0 / block_size));
    unsigned cells_per_block = static_cast<unsigned>(N * block_size);

    // Shuffle R and dist_mat in-place (matches R: "Reference matrices")
    R = R.cols(update_order);
    dist_mat = dist_mat.cols(update_order);

    // Phi must be copied (sparse, can't shuffle in-place efficiently)
    SPMAT Phi_randomized(Phi.cols(update_order));
    SPMAT Phi_t_randomized(Phi_randomized.t());

    for (unsigned i = 0; i < n_blocks; ++i) {
        unsigned idx_min = i * cells_per_block;
        unsigned idx_max = ((i + 1) * cells_per_block) - 1;
        if (i == n_blocks - 1) idx_max = N - 1;

        auto Rcells = R.submat(0, idx_min, R.n_rows - 1, idx_max);
        auto Phicells = Phi_randomized.submat(0, idx_min, Phi_randomized.n_rows - 1, idx_max);
        auto Phi_tcells = Phi_t_randomized.submat(idx_min, 0, idx_max, Phi_t_randomized.n_cols - 1);
        auto dist_matcells = dist_mat.submat(0, idx_min, dist_mat.n_rows - 1, idx_max);

        // Step 1: remove cells
        E -= VECTYPE(arma::sum(Rcells, 1)) * Pr_b.t();
        O -= Rcells * Phi_tcells;

        // Step 2: recompute R from dist_mat (no _scale_dist buffer)
        Rcells = -dist_matcells;
        Rcells.each_col() /= sigma;
        Rcells.transform([](float val) { return std::exp(val); });
        Rcells = arma::normalise(Rcells, 1, 0);
        Rcells = Rcells % (harmony_pow((2*E + 1) / (O + E + 1), theta) * Phicells);
        Rcells = arma::normalise(Rcells, 1, 0);

        // Step 3: put cells back
        E += VECTYPE(arma::sum(Rcells, 1)) * Pr_b.t();
        O += Rcells * Phi_tcells;
    }

    // Unshuffle (in-place, matches R)
    R = R.cols(reverse_index);
    dist_mat = dist_mat.cols(reverse_index);
}

bool Harmony::check_convergence(int i_type) {
    if (i_type == 0) {
        if (objective_kmeans.size() <= static_cast<size_t>(window_size + 1))
            return false;

        float obj_old = 0.0f, obj_new = 0.0f;
        size_t n = objective_kmeans.size();
        for (int i = 0; i < window_size; ++i) {
            obj_old += objective_kmeans[n - window_size - 1 + i];
            obj_new += objective_kmeans[n - window_size + i];
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
    // Reset Z_corr to Z_orig (matches R)
    Z_corr = Z_orig;

    VECTYPE sizes = VECTYPE(arma::sum(Phi, 1));

    for (int k = 0; k < K; ++k) {
        VECTYPE avg_R = O.row(k).t() / sizes;

        // Determine which batches qualify
        std::vector<unsigned> keep;
        std::vector<unsigned> cov_levels(B_vec.size(), 0);

        for (unsigned b = 0, current_cov = 0; b < static_cast<unsigned>(B); ++b) {
            if (current_cov < covariate_bounds.size() - 1 &&
                b >= covariate_bounds[current_cov])
                current_cov++;
            if (avg_R(b) > batch_proportion_cutoff)
                cov_levels[current_cov]++;
        }

        unsigned active_covariates = 0;
        for (auto const& l : cov_levels) {
            if (l > 1) active_covariates++;
        }

        for (unsigned b = 0, current_cov = 0; b < static_cast<unsigned>(B); ++b) {
            if (current_cov < covariate_bounds.size() - 1 &&
                b >= covariate_bounds[current_cov])
                current_cov++;
            if (avg_R(b) > batch_proportion_cutoff && cov_levels[current_cov] > 1)
                keep.push_back(b);
        }

        if (active_covariates == 0) continue;

        if (keep.size() == static_cast<size_t>(B)) {
            // All batches qualify
            SPMAT lambda_mat(B + 1, B + 1);
            if (lambda_estimation)
                lambda_mat.diag() = find_lambda(alpha, VECTYPE(E.row(k).t()));
            else
                lambda_mat.diag() = lambda;

            // Build Phi_cov directly from O[k,:] — no N×N sparse Rk needed
            VECTYPE Ok = VECTYPE(O.row(k).t());
            MATTYPE Phi_cov(B + 1, B + 1, arma::fill::zeros);
            float Ok_sum = arma::accu(Ok);
            Phi_cov(0, 0) = Ok_sum;
            for (int b = 0; b < B; ++b) {
                Phi_cov(0, b + 1) = Ok(b);
                Phi_cov(b + 1, 0) = Ok(b);
                Phi_cov(b + 1, b + 1) = Ok(b);
            }
            Phi_cov += MATTYPE(lambda_mat);

            MATTYPE inv_cov;
            if (B_vec.size() > 1) {
                inv_cov = arma::inv(Phi_cov);
            } else {
                // Arrowhead inverse
                VECTYPE ac = -VECTYPE(Phi_cov.row(0).t());
                ac(0) = 1;
                float b0 = Phi_cov(0, 0);
                VECTYPE b = 1.0f / Phi_cov.diag();
                b(0) = 0;
                float u = b0 - arma::accu(arma::square(ac) % b);
                VECTYPE ac_b = ac % b;
                ac_b(0) = 1;
                inv_cov = (1.0f/u) * (ac_b * ac_b.t());
                inv_cov.diag() += b;
            }

            // Compute W via per-batch matrix-vector products (no d×N Z_tmp)
            VECTYPE Rk = VECTYPE(R.row(k).t());
            std::vector<VECTYPE> z_sums(B);
            VECTYPE z_sum_all(d, arma::fill::zeros);
            for (int b = 0; b < B; ++b) {
                z_sums[b] = Z_orig.cols(batch_index[b]) * Rk.rows(batch_index[b]);
                z_sum_all += z_sums[b];
            }

            W = inv_cov.col(0) * z_sum_all.t();
            for (int b = 0; b < B; ++b) {
                W += inv_cov.col(b + 1) * z_sums[b].t();
            }

            Y.col(k) = W.row(0).t();
            W.row(0).zeros();

            // Batch-wise correction (no d×N product, no N×N Rk)
            for (int b = 0; b < B; ++b) {
                Z_corr.cols(batch_index[b]) -= W.row(b + 1).t() * Rk.rows(batch_index[b]).t();
            }

        } else {
            // Subset to qualifying batches
            unsigned n_keep = keep.size();

            // Lambda for subsetted batches
            SPMAT lambda_mat(n_keep + 1, n_keep + 1);
            if (lambda_estimation) {
                arma::uvec keep_batch = arma::conv_to<arma::uvec>::from(keep);
                VECTYPE E_sub = VECTYPE(E.row(k).t());
                E_sub = E_sub.rows(keep_batch);
                lambda_mat.diag() = find_lambda(alpha, E_sub);
            } else {
                VECTYPE ltmp(n_keep + 1);
                ltmp(0) = 0;
                arma::uvec keep_batch = arma::conv_to<arma::uvec>::from(keep);
                ltmp.subvec(1, n_keep) = lambda.rows(keep_batch + 1);
                lambda_mat.diag() = ltmp;
            }

            // Build cov_mat directly from O[k, keep]
            VECTYPE Ok_sub(n_keep);
            for (unsigned i = 0; i < n_keep; ++i) Ok_sub(i) = O(k, keep[i]);
            MATTYPE cov_mat(n_keep + 1, n_keep + 1, arma::fill::zeros);
            float Ok_sub_sum = arma::accu(Ok_sub);
            cov_mat(0, 0) = Ok_sub_sum;
            for (unsigned i = 0; i < n_keep; ++i) {
                cov_mat(0, i + 1) = Ok_sub(i);
                cov_mat(i + 1, 0) = Ok_sub(i);
                cov_mat(i + 1, i + 1) = Ok_sub(i);
            }
            cov_mat += MATTYPE(lambda_mat);

            MATTYPE inv_cov;
            if (B_vec.size() > 1)
                inv_cov = arma::inv(cov_mat);
            else {
                VECTYPE ac = -VECTYPE(cov_mat.row(0).t());
                ac(0) = 1;
                float b0 = cov_mat(0, 0);
                VECTYPE b = 1.0f / cov_mat.diag();
                b(0) = 0;
                float u = b0 - arma::accu(arma::square(ac) % b);
                VECTYPE ac_b = ac % b;
                ac_b(0) = 1;
                inv_cov = (1.0f/u) * (ac_b * ac_b.t());
                inv_cov.diag() += b;
            }

            // Compute W via per-batch matrix-vector products
            VECTYPE Rk = VECTYPE(R.row(k).t());
            std::vector<VECTYPE> z_sums(n_keep);
            VECTYPE z_sum_all(d, arma::fill::zeros);
            for (unsigned i = 0; i < n_keep; ++i) {
                unsigned b = keep[i];
                z_sums[i] = Z_orig.cols(batch_index[b]) * Rk.rows(batch_index[b]);
                z_sum_all += z_sums[i];
            }

            MATTYPE W_sub = inv_cov.col(0) * z_sum_all.t();
            for (unsigned i = 0; i < n_keep; ++i) {
                W_sub += inv_cov.col(i + 1) * z_sums[i].t();
            }

            Y.col(k) = W_sub.row(0).t();
            W_sub.row(0).zeros();

            // Batch-wise correction
            for (unsigned i = 0; i < n_keep; ++i) {
                unsigned b = keep[i];
                Z_corr.cols(batch_index[b]) -= W_sub.row(i + 1).t() * Rk.rows(batch_index[b]).t();
            }
        }
    }

    Y = arma::normalise(Y, 2, 0);
    // Z_corr will be re-normalized at the start of the next cluster() call
}

} // namespace harmony
