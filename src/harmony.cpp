// harmonypy - C++ backend matching R harmony2 package.
// Copyright (C) 2018  Ilya Korsunsky
//               2019  Kamil Slowikowski <kslowikowski@gmail.com>
//
// Uses custom scatter/gather kernels on a batch_id vector instead of
// sparse Phi matrices. BLAS threading (Accelerate/OpenBLAS) is controlled
// by the ncores parameter via environment variables in the Python layer.

#include "harmony.hpp"
#include <numeric>
#include <set>

namespace harmony {

// =========================================================================
// Custom kernels
// =========================================================================

void Harmony::scatter_add_O(const MATTYPE& Rsub, const arma::uvec& ids, float sign) {
    const int n = Rsub.n_cols;
    const int k = Rsub.n_rows;
    for (int j = 0; j < n; ++j) {
        unsigned b = ids(j);
        const float* col = Rsub.colptr(j);
        float* dst = O.colptr(b);
        if (sign > 0) {
            for (int i = 0; i < k; ++i) dst[i] += col[i];
        } else {
            for (int i = 0; i < k; ++i) dst[i] -= col[i];
        }
    }
}

// =========================================================================
// K-means initialization (matches R harmony2)
// =========================================================================

MATTYPE kmeans_init(const MATTYPE& X, int K, std::mt19937& rng) {
    int N = X.n_cols;

    std::uniform_real_distribution<float> uniform01(0.0f, 1.0f);
    MATTYPE Y(X.n_rows, K);
    for (int i = 0; i < K; ++i) {
        int idx = static_cast<int>(std::round(uniform01(rng) * N));
        if (idx >= N) idx = N - 1;
        Y.col(i) = X.col(idx);
    }

    std::set<unsigned> chosen;
    for (int i = 0; i < K; ++i) {
        VECTYPE distances = arma::abs((2.0f * (1.0f - Y.col(i).t() * X)).as_col());
        VECTYPE random_numbers(N, arma::fill::none);
        for (int j = 0; j < N; ++j) random_numbers(j) = uniform01(rng);
        VECTYPE prob = -arma::log(random_numbers) / (distances + 1e-10f);

        for (auto idx : chosen) prob(idx) = prob.max();
        unsigned best = prob.index_min();
        while (chosen.count(best)) {
            prob(best) = prob.max();
            best = prob.index_min();
        }
        chosen.insert(best);
        Y.col(i) = X.col(best);
    }

    for (int i = 0; i < 10; ++i) {
        arma::kmeans(Y, X, K, arma::keep_existing, 1, false);
    }

    return Y;
}

// =========================================================================
// Constructor
// =========================================================================

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
    batch_id.set_size(N);
    arma::sp_mat::const_iterator it = Phi_in.begin();
    arma::sp_mat::const_iterator it_end = Phi_in.end();
    for (; it != it_end; ++it) {
        batch_id(it.col()) = it.row();
    }

    batch_sizes = arma::conv_to<VECTYPE>::from(arma::vec(arma::sum(Phi_in, 1)));
    batch_index.resize(B);
    std::vector<unsigned> counters(B, 0);
    for (int b = 0; b < B; ++b) {
        batch_index[b].set_size(static_cast<unsigned>(batch_sizes(b)));
    }
    for (int j = 0; j < N; ++j) {
        unsigned b = batch_id(j);
        batch_index[b](counters[b]++) = j;
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

// =========================================================================
// init_cluster
// =========================================================================

void Harmony::init_cluster() {
    Y = kmeans_init(Z_corr, K, rng);
    Y = arma::normalise(Y, 2, 0);

    dist_mat = 2.0f * (1.0f - Y.t() * Z_corr);

    R = -dist_mat;
    R.each_col() /= sigma;
    R = arma::exp(R);
    R.each_row() /= arma::sum(R, 0);

    E = arma::sum(R, 1) * Pr_b.t();
    O.zeros();
    scatter_add_O(R, batch_id, 1.0f);

    compute_objective();
    objective_harmony.push_back(objective_kmeans.back());
}

// =========================================================================
// compute_objective
// =========================================================================

void Harmony::compute_objective() {
    const float norm_const = 2000.0f / static_cast<float>(N);

    float kmeans_error = arma::accu(R % dist_mat);

    MATTYPE log_R = R;
    log_R.transform([](float val) { return val > 0 ? val * std::log(val) : 0.0f; });
    float _entropy = arma::as_scalar(arma::accu(log_R.each_col() % sigma));

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

// =========================================================================
// harmonize / cluster
// =========================================================================

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
    if (objective_harmony.size() > 1) {
        Z_corr = arma::normalise(Z_corr, 2, 0);
        dist_mat = 2.0f * (1.0f - Y.t() * Z_corr);
        R = -dist_mat;
        R.each_col() /= sigma;
        R = arma::exp(R);
        R.each_row() /= arma::sum(R, 0);
        E = arma::sum(R, 1) * Pr_b.t();
        O.zeros();
        scatter_add_O(R, batch_id, 1.0f);
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

// =========================================================================
// update_R
// =========================================================================

void Harmony::update_R() {
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

    R = R.cols(update_order);
    dist_mat = dist_mat.cols(update_order);
    arma::uvec batch_id_shuf = batch_id.rows(update_order);

    for (unsigned i = 0; i < n_blocks; ++i) {
        unsigned idx_min = i * cells_per_block;
        unsigned idx_max = ((i + 1) * cells_per_block) - 1;
        if (i == n_blocks - 1) idx_max = N - 1;
        if (idx_min >= static_cast<unsigned>(N)) break;
        unsigned block_n = idx_max - idx_min + 1;

        auto Rcells = R.submat(0, idx_min, R.n_rows - 1, idx_max);
        auto dist_matcells = dist_mat.submat(0, idx_min, dist_mat.n_rows - 1, idx_max);
        arma::uvec block_ids = batch_id_shuf.subvec(idx_min, idx_max);

        E -= arma::sum(Rcells, 1) * Pr_b.t();
        scatter_add_O(Rcells, block_ids, -1.0f);

        Rcells = -dist_matcells;
        Rcells.each_col() /= sigma;
        Rcells = arma::exp(Rcells);
        Rcells = arma::normalise(Rcells, 1, 0);

        MATTYPE div_ratio = harmony_pow(((2*E) + 1) / (O + E + 1), theta);
        for (unsigned j = 0; j < block_n; ++j) {
            unsigned b = block_ids(j);
            float* col = Rcells.colptr(j);
            const float* src = div_ratio.colptr(b);
            for (int ki = 0; ki < K; ++ki) col[ki] *= src[ki];
        }
        Rcells = arma::normalise(Rcells, 1, 0);

        E += arma::sum(Rcells, 1) * Pr_b.t();
        scatter_add_O(Rcells, block_ids, 1.0f);
    }

    R = R.cols(reverse_index);
    dist_mat = dist_mat.cols(reverse_index);
}

// =========================================================================
// check_convergence
// =========================================================================

bool Harmony::check_convergence(int i_type) {
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

// =========================================================================
// moe_correct_ridge
// =========================================================================

void Harmony::moe_correct_ridge() {
    Z_corr = Z_orig;

    for (int k = 0; k < K; ++k) {
        VECTYPE avg_R = O.row(k).t() / batch_sizes;

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

        unsigned n_keep = keep.size();
        bool all_qualify = (n_keep == static_cast<unsigned>(B));

        VECTYPE lamb_vec;
        if (all_qualify) {
            lamb_vec = lambda_estimation ? find_lambda(alpha, VECTYPE(E.row(k).t())) : lambda;
        } else {
            arma::uvec keep_batch = arma::conv_to<arma::uvec>::from(keep);
            if (lambda_estimation) {
                VECTYPE Esub = VECTYPE(E.row(k).t());
                Esub = Esub.rows(keep_batch);
                lamb_vec = find_lambda(alpha, Esub);
            } else {
                VECTYPE ltmp(n_keep + 1);
                ltmp(0) = 0;
                ltmp.subvec(1, n_keep) = lambda.rows(keep_batch + 1);
                lamb_vec = ltmp;
            }
        }

        unsigned mat_size = (all_qualify ? B : n_keep) + 1;
        VECTYPE Ok(all_qualify ? B : n_keep);
        if (all_qualify) {
            Ok = VECTYPE(O.row(k).t());
        } else {
            for (unsigned i = 0; i < n_keep; ++i) Ok(i) = O(k, keep[i]);
        }

        MATTYPE cov_mat(mat_size, mat_size, arma::fill::zeros);
        float Ok_sum = arma::accu(Ok);
        cov_mat(0, 0) = Ok_sum;
        for (unsigned i = 0; i < Ok.n_elem; ++i) {
            cov_mat(0, i + 1) = Ok(i);
            cov_mat(i + 1, 0) = Ok(i);
            cov_mat(i + 1, i + 1) = Ok(i);
        }
        cov_mat += arma::diagmat(lamb_vec);

        MATTYPE inv_cov;
        if (B_vec.size() > 1) {
            inv_cov = arma::inv(cov_mat);
        } else {
            VECTYPE ac = -cov_mat.row(0).as_col();
            ac(0) = 1;
            float b0 = cov_mat(0, 0);
            VECTYPE b = 1.0f / cov_mat.diag();
            b(0) = 0;
            float u = b0 - arma::accu(arma::square(ac) % b);
            VECTYPE ac_b = ac % b;
            ac_b(0) = 1;
            inv_cov = (1.0f / u) * (ac_b * ac_b.t());
            inv_cov.diag() += b;
        }

        ROWTYPE Rk = R.row(k);
        unsigned n_batches = all_qualify ? B : n_keep;

        std::vector<VECTYPE> z_sums(n_batches);
        VECTYPE z_sum_all(d, arma::fill::zeros);

        for (unsigned i = 0; i < n_batches; ++i) {
            unsigned b = all_qualify ? i : keep[i];
            const arma::uvec& idx = batch_index[b];
            z_sums[i] = Z_orig.cols(idx) * arma::conv_to<VECTYPE>::from(Rk.cols(idx).t());
            z_sum_all += z_sums[i];
        }

        W = inv_cov.unsafe_col(0) * z_sum_all.t();
        for (unsigned i = 0; i < n_batches; ++i) {
            W += inv_cov.unsafe_col(i + 1) * z_sums[i].t();
        }

        Y.col(k) = W.row(0).t();
        W.row(0).zeros();

        if (all_qualify) {
            for (int b = 0; b < B; ++b) {
                const arma::uvec& idx = batch_index[b];
                Z_corr.cols(idx) -= W.row(b + 1).t() * Rk.cols(idx);
            }
        } else {
            for (unsigned i = 0; i < n_keep; ++i) {
                unsigned b = keep[i];
                const arma::uvec& idx = batch_index[b];
                Z_corr.cols(idx) -= W.row(i + 1).t() * Rk.cols(idx);
            }
        }
    }

    Y = arma::normalise(Y, 2, 0);
}

} // namespace harmony
