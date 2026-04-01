// harmonypy - C++ backend matching R harmony2 package.
// Copyright (C) 2018  Ilya Korsunsky
//               2019  Kamil Slowikowski <kslowikowski@gmail.com>

#include "harmony.hpp"
#include <numeric>
#include <set>

#ifdef HARMONY_USE_OPENMP
#include <omp.h>
#endif

namespace harmony {

constexpr int PARALLEL_THRESHOLD = 50000;

// K-means++ initialization
arma::mat kmeans_plusplus(const arma::mat& data, int K, std::mt19937& rng) {
    int N = data.n_rows;
    int d_dim = data.n_cols;

    arma::mat centroids(K, d_dim);
    std::vector<bool> chosen(N, false);

    std::uniform_int_distribution<int> uniform(0, N - 1);
    int first = uniform(rng);
    centroids.row(0) = data.row(first);
    chosen[first] = true;

    arma::vec min_distances(N, arma::fill::value(std::numeric_limits<double>::max()));

    for (int k = 1; k < K; ++k) {
        #ifdef HARMONY_USE_OPENMP
        #pragma omp parallel for schedule(static) if(N > PARALLEL_THRESHOLD)
        #endif
        for (int i = 0; i < N; ++i) {
            if (!chosen[i]) {
                arma::rowvec diff = data.row(i) - centroids.row(k-1);
                double dist = arma::dot(diff, diff);
                min_distances(i) = std::min(min_distances(i), dist);
            }
        }

        std::discrete_distribution<int> weighted_dist(
            min_distances.begin(), min_distances.end()
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
            double min_dist = std::numeric_limits<double>::max();
            int best_k = 0;
            for (int kk = 0; kk < K; ++kk) {
                arma::rowvec diff = data.row(i) - centroids.row(kk);
                double dist = arma::dot(diff, diff);
                if (dist < min_dist) { min_dist = dist; best_k = kk; }
            }
            assignments[i] = best_k;
        }

        arma::mat new_centroids(K, d_dim, arma::fill::zeros);
        arma::vec counts(K, arma::fill::zeros);
        for (int i = 0; i < N; ++i) {
            new_centroids.row(assignments[i]) += data.row(i);
            counts(assignments[i]) += 1;
        }
        for (int kk = 0; kk < K; ++kk) {
            if (counts(kk) > 0) new_centroids.row(kk) /= counts(kk);
            else new_centroids.row(kk) = data.row(uniform(rng));
        }

        double change = arma::norm(new_centroids - centroids, "fro");
        centroids = new_centroids;
        if (change < 1e-6) break;
    }

    return centroids;
}

// Constructor — matches R harmony2
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
    epsilon_kmeans(epsilon_kmeans),
    epsilon_harmony(epsilon_harmony),
    K(K),
    block_size(block_size),
    verbose(verbose),
    window_size(3),
    alpha(alpha_in),
    batch_proportion_cutoff(batch_proportion_cutoff),
    B_vec(B_vec_in),
    rng(random_state)
{
    Z_orig = Z;

    // L2 normalize each column
    Z_cos = arma::normalise(Z, 2, 0);

    // Sparse batch indicators
    Phi = Phi_in;
    Phi_t = Phi.t();

    // Phi_moe: intercept row + Phi
    arma::sp_mat intercept(1, Phi.n_cols);
    for (unsigned i = 0; i < Phi.n_cols; ++i) intercept(0, i) = 1.0;
    Phi_moe = arma::join_cols(intercept, Phi);
    Phi_moe_t = Phi_moe.t();

    Pr_b = Pr_b_in;
    N = Z.n_cols;
    d = Z.n_rows;
    B = Phi.n_rows;

    sigma = sigma_in;
    theta = theta_in;

    if (lambda_in(0) < 0) {
        lambda_estimation = true;
        lambda.zeros(B + 1);
    } else {
        lambda_estimation = false;
        lambda = lambda_in;
    }

    // Covariate bounds (cumulative sum of B_vec)
    covariate_bounds.resize(B_vec.size());
    unsigned cumsum = 0;
    for (unsigned i = 0; i < B_vec.size(); ++i) {
        cumsum += B_vec[i];
        covariate_bounds[i] = cumsum;
    }

    build_batch_index();
    allocate_buffers();

    if (verbose) std::cout << "Computing initial centroids..." << std::endl;
    std::cout.flush();
    init_cluster();
    if (verbose) std::cout << "Initialization complete." << std::endl;
    std::cout.flush();
    harmonize(max_iter_harmony, verbose);
}

void Harmony::build_batch_index() {
    arma::vec sizes = arma::vec(arma::sum(Phi, 1));
    std::vector<unsigned> counters(B, 0);
    batch_index.resize(B);
    for (int b = 0; b < B; ++b) {
        batch_index[b].zeros(static_cast<unsigned>(sizes(b)));
    }
    arma::sp_mat::const_iterator it = Phi.begin();
    arma::sp_mat::const_iterator it_end = Phi.end();
    for (; it != it_end; ++it) {
        unsigned row_idx = it.row();
        unsigned col_idx = it.col();
        batch_index[row_idx](counters[row_idx]++) = col_idx;
    }
}

void Harmony::allocate_buffers() {
    _scale_dist.zeros(K, N);
    dist_mat.zeros(K, N);
    O.zeros(K, B);
    E.zeros(K, B);
    W.zeros(B + 1, d);
    R.zeros(K, N);
    Y.zeros(d, K);
}

void Harmony::init_cluster() {
    arma::mat centroids = kmeans_plusplus(Z_cos.t(), K, rng);
    Y = centroids.t();
    Y = arma::normalise(Y, 2, 0);

    dist_mat = 2.0 * (1.0 - Y.t() * Z_cos);

    R = -dist_mat;
    R.each_col() /= sigma;
    R.transform([](double val) { return std::exp(val); });
    R.each_row() /= arma::sum(R, 0);

    E = arma::sum(R, 1) * Pr_b.t();
    O = R * Phi_t;

    compute_objective();
    objective_harmony.push_back(objective_kmeans.back());
}

void Harmony::compute_objective() {
    const double norm_const = 2000.0 / static_cast<double>(N);

    double kmeans_error = arma::accu(R % dist_mat);

    // Entropy: sum(xlogy(R, R) * sigma)
    arma::mat log_R = R;
    log_R.transform([](double val) { return val > 0 ? std::log(val) : 0.0; });
    arma::mat entropy_mat = R % log_R;
    entropy_mat.each_col() %= sigma;
    double _entropy = arma::accu(entropy_mat);

    // Cross entropy (harmony2 formula)
    arma::mat R_sigma = R;
    R_sigma.each_col() %= sigma;
    arma::mat ratio = (O + E + 1) / (2 * E + 1);
    arma::mat log_ratio = ratio;
    log_ratio.transform([](double val) { return std::log(val); });
    arma::mat theta_log = arma::repmat(theta.t(), K, 1) % log_ratio;
    double _cross_entropy = arma::accu(R_sigma % (theta_log * Phi));

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
    if (objective_harmony.size() > 1) {
        Z_cos = arma::normalise(Z_corr, 2, 0);
        dist_mat = 2.0 * (1.0 - Y.t() * Z_cos);
        R = -dist_mat;
        R.each_col() /= sigma;
        R.transform([](double val) { return std::exp(val); });
        R.each_row() /= arma::sum(R, 0);
        E = arma::sum(R, 1) * Pr_b.t();
        O = R * Phi_t;
    }

    int rounds = 0;
    for (int i = 0; i < max_iter_kmeans; ++i) {
        // harmony2: no Y/dist_mat update inside k-means loop
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
    // Compute scaled distances
    _scale_dist = -dist_mat;
    _scale_dist.each_col() /= sigma;
    _scale_dist.transform([](double val) { return std::exp(val); });
    _scale_dist = arma::normalise(_scale_dist, 1, 0);

    // Shuffle
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

    // Shuffle matrices
    arma::mat R_randomized = R.cols(update_order);
    arma::sp_mat Phi_randomized(Phi.cols(update_order));
    arma::sp_mat Phi_t_randomized(Phi_randomized.t());
    arma::mat scale_dist_randomized = _scale_dist.cols(update_order);

    for (unsigned i = 0; i < n_blocks; ++i) {
        unsigned idx_min = i * cells_per_block;
        unsigned idx_max = ((i + 1) * cells_per_block) - 1;
        if (i == n_blocks - 1) idx_max = N - 1;

        auto R_block = R_randomized.cols(idx_min, idx_max);
        auto Phi_block = Phi_randomized.cols(idx_min, idx_max);
        auto Phi_t_block = Phi_t_randomized.rows(idx_min, idx_max);
        auto scale_block = scale_dist_randomized.cols(idx_min, idx_max);

        // Remove cells
        E -= arma::sum(R_block, 1) * Pr_b.t();
        O -= R_block * Phi_t_block;

        // harmony2 formula: (2*E+1) / (O+E+1)
        R_block = scale_block % (harmony_pow((2*E + 1) / (O + E + 1), theta) * Phi_block);
        R_block = arma::normalise(R_block, 1, 0);

        // Add cells back
        E += arma::sum(R_block, 1) * Pr_b.t();
        O += R_block * Phi_t_block;

        R_randomized.cols(idx_min, idx_max) = R_block;
    }

    R = R_randomized.cols(reverse_index);
}

bool Harmony::check_convergence(int i_type) {
    if (i_type == 0) {
        if (objective_kmeans.size() <= static_cast<size_t>(window_size + 1))
            return false;

        double obj_old = 0.0, obj_new = 0.0;
        size_t n = objective_kmeans.size();
        for (int i = 0; i < window_size; ++i) {
            obj_old += objective_kmeans[n - window_size - 1 + i];
            obj_new += objective_kmeans[n - window_size + i];
        }
        return std::abs(obj_old - obj_new) / std::abs(obj_old) < epsilon_kmeans;
    }

    if (i_type == 1) {
        if (objective_harmony.size() < 2) return false;
        double obj_old = objective_harmony[objective_harmony.size() - 2];
        double obj_new = objective_harmony[objective_harmony.size() - 1];
        return (obj_old - obj_new) / std::abs(obj_old) < epsilon_harmony;
    }
    return true;
}

void Harmony::moe_correct_ridge() {
    // Reset Z_corr (harmony2: start from Z_orig each time)
    Z_corr = Z_orig;

    arma::vec sizes = arma::vec(arma::sum(Phi, 1));

    for (int k = 0; k < K; ++k) {
        arma::vec avg_R = O.row(k).t() / sizes;

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
            // All batches qualify — use full data
            arma::sp_mat lambda_mat(B + 1, B + 1);
            if (lambda_estimation)
                lambda_mat.diag() = find_lambda(alpha, E.row(k).t());
            else
                lambda_mat.diag() = lambda;

            arma::sp_mat _Rk(N, N);
            _Rk.diag() = R.row(k).t();

            arma::sp_mat Phi_Rk = Phi_moe * _Rk;
            arma::mat Phi_cov = arma::mat(Phi_Rk * Phi_moe_t + lambda_mat);

            arma::mat inv_cov;
            if (B_vec.size() > 1) {
                inv_cov = arma::inv(Phi_cov);
            } else {
                // Arrowhead inverse
                arma::vec ac = -Phi_cov.row(0).t();
                ac(0) = 1;
                double b0 = Phi_cov(0, 0);
                arma::vec b = 1.0 / Phi_cov.diag();
                b(0) = 0;
                double u = b0 - arma::accu(arma::square(ac) % b);
                arma::vec ac_b = ac % b;
                ac_b(0) = 1;
                inv_cov = (1.0/u) * (ac_b * ac_b.t());
                inv_cov.diag() += b;
            }

            arma::mat Z_tmp = Z_orig.each_row() % R.row(k);

            W = inv_cov.col(0) * arma::sum(Z_tmp, 1).t();
            for (int b = 0; b < B; ++b) {
                W += inv_cov.col(b + 1) * arma::sum(Z_tmp.cols(batch_index[b]), 1).t();
            }

            Y.col(k) = W.row(0).t();
            W.row(0).zeros();
            Z_corr -= W.t() * Phi_Rk;

        } else {
            // Subset to qualifying batches and their cells
            // Collect cells
            std::set<unsigned> keep_cells_set;
            for (auto b : keep) {
                for (unsigned i = 0; i < batch_index[b].n_elem; ++i) {
                    keep_cells_set.insert(batch_index[b](i));
                }
            }

            arma::uvec keep_cols = arma::conv_to<arma::uvec>::from(
                std::vector<unsigned>(keep_cells_set.begin(), keep_cells_set.end()));

            unsigned n_keep = keep.size();
            unsigned n_cells = keep_cols.n_elem;

            // Build cell map for fast lookup
            std::vector<int> cell_map(N, -1);
            for (unsigned i = 0; i < n_cells; ++i) {
                cell_map[keep_cols(i)] = i;
            }

            // Build subsetted batch index
            std::vector<arma::uvec> sub_batch_index(n_keep);
            for (unsigned i = 0; i < n_keep; ++i) {
                unsigned b = keep[i];
                std::vector<unsigned> mapped;
                mapped.reserve(batch_index[b].n_elem);
                for (unsigned j = 0; j < batch_index[b].n_elem; ++j) {
                    int mi = cell_map[batch_index[b](j)];
                    if (mi >= 0) mapped.push_back(static_cast<unsigned>(mi));
                }
                sub_batch_index[i] = arma::conv_to<arma::uvec>::from(mapped);
            }

            // Build subsetted Phi_moe (n_keep+1 x n_cells, sparse)
            // Row 0 = intercept (all ones)
            // Rows 1..n_keep = one-hot for qualifying batches
            arma::umat locations(2, n_cells + n_cells); // worst case
            arma::vec values(n_cells + n_cells);
            unsigned nnz = 0;
            // Intercept row
            for (unsigned j = 0; j < n_cells; ++j) {
                locations(0, nnz) = 0;
                locations(1, nnz) = j;
                values(nnz) = 1.0;
                nnz++;
            }
            // Batch rows
            for (unsigned i = 0; i < n_keep; ++i) {
                for (unsigned j = 0; j < sub_batch_index[i].n_elem; ++j) {
                    locations(0, nnz) = i + 1;
                    locations(1, nnz) = sub_batch_index[i](j);
                    values(nnz) = 1.0;
                    nnz++;
                }
            }
            locations = locations.cols(0, nnz - 1);
            values = values.subvec(0, nnz - 1);

            arma::sp_mat sub_Phi_moe(locations, values, n_keep + 1, n_cells);
            arma::sp_mat sub_Phi_moe_t = sub_Phi_moe.t();

            // Lambda
            arma::sp_mat lambda_mat(n_keep + 1, n_keep + 1);
            if (lambda_estimation) {
                arma::uvec keep_batch = arma::conv_to<arma::uvec>::from(keep);
                arma::vec E_sub = E.row(k).t();
                E_sub = E_sub.rows(keep_batch);
                lambda_mat.diag() = find_lambda(alpha, E_sub);
            } else {
                arma::vec ltmp(n_keep + 1);
                ltmp(0) = 0;
                arma::uvec keep_batch = arma::conv_to<arma::uvec>::from(keep);
                ltmp.subvec(1, n_keep) = lambda.rows(keep_batch + 1);
                lambda_mat.diag() = ltmp;
            }

            // Rk for subset
            arma::vec Rk_full = R.row(k).t();
            arma::sp_mat sub_Rk(n_cells, n_cells);
            sub_Rk.diag() = Rk_full.rows(keep_cols);

            arma::sp_mat sub_Phi_Rk = sub_Phi_moe * sub_Rk;
            arma::mat sub_Phi_cov = arma::mat(sub_Phi_Rk * sub_Phi_moe_t + lambda_mat);

            arma::mat inv_cov;
            if (B_vec.size() > 1)
                inv_cov = arma::inv(sub_Phi_cov);
            else {
                arma::vec ac = -sub_Phi_cov.row(0).t();
                ac(0) = 1;
                double b0 = sub_Phi_cov(0, 0);
                arma::vec b = 1.0 / sub_Phi_cov.diag();
                b(0) = 0;
                double u = b0 - arma::accu(arma::square(ac) % b);
                arma::vec ac_b = ac % b;
                ac_b(0) = 1;
                inv_cov = (1.0/u) * (ac_b * ac_b.t());
                inv_cov.diag() += b;
            }

            arma::mat Z_tmp = Z_orig.cols(keep_cols);
            Z_tmp = Z_tmp.each_row() % Rk_full.rows(keep_cols).t();

            arma::mat W_sub = inv_cov.col(0) * arma::sum(Z_tmp, 1).t();
            for (unsigned i = 0; i < n_keep; ++i) {
                W_sub += inv_cov.col(i + 1) * arma::sum(Z_tmp.cols(sub_batch_index[i]), 1).t();
            }

            Y.col(k) = W_sub.row(0).t();
            W_sub.row(0).zeros();
            arma::mat correction = W_sub.t() * sub_Phi_Rk;
            Z_corr.cols(keep_cols) -= correction;
        }
    }

    Y = arma::normalise(Y, 2, 0);
    Z_cos = arma::normalise(Z_corr, 2, 0);
}

} // namespace harmony
