// harmonypy - A data alignment algorithm.
// Copyright (C) 2018  Ilya Korsunsky
//               2019  Kamil Slowikowski <kslowikowski@gmail.com>
//
// harmony2 C++ backend — matches the R harmony2 package algorithm.
// Uses custom scatter/gather kernels on a batch_id vector instead of
// sparse Phi matrices, eliminating all sparse matrix overhead.

#ifndef HARMONY_HPP
#define HARMONY_HPP

#include <armadillo>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace harmony {

typedef arma::Mat<float> MATTYPE;
typedef arma::Col<float> VECTYPE;
typedef arma::Row<float> ROWTYPE;

inline VECTYPE find_lambda(float alpha, const VECTYPE& cluster_E) {
    VECTYPE lambda_vec(cluster_E.n_elem + 1, arma::fill::zeros);
    lambda_vec.subvec(1, lambda_vec.n_elem - 1) = cluster_E * alpha;
    return lambda_vec;
}

inline MATTYPE harmony_pow(MATTYPE A, const VECTYPE& T) {
    for (unsigned c = 0; c < A.n_cols; c++) {
        A.col(c) = arma::pow(A.col(c), T(c));
    }
    return A;
}

MATTYPE kmeans_init(const MATTYPE& X, int K, std::mt19937& rng);

class Harmony {
public:
    MATTYPE Z_orig;
    MATTYPE Z_corr;

    arma::uvec batch_id;
    VECTYPE Pr_b;
    VECTYPE batch_sizes;
    std::vector<arma::uvec> batch_index;

    MATTYPE Y;
    MATTYPE R;
    MATTYPE dist_mat;

    MATTYPE O;
    MATTYPE E;
    MATTYPE W;

    VECTYPE sigma;
    VECTYPE theta;
    VECTYPE lambda;

    float alpha;
    bool lambda_estimation;

    int N, d, K, B;
    int max_iter_harmony, max_iter_kmeans;
    float epsilon_kmeans, epsilon_harmony;
    float block_size;
    int window_size;
    bool verbose;

    std::vector<int> B_vec;
    std::vector<unsigned> covariate_bounds;
    float batch_proportion_cutoff;

    std::vector<float> objective_harmony;
    std::vector<float> objective_kmeans;
    std::vector<float> objective_kmeans_dist;
    std::vector<float> objective_kmeans_entropy;
    std::vector<float> objective_kmeans_cross;
    std::vector<int> kmeans_rounds;

    std::mt19937 rng;

    Harmony(
        const arma::mat& Z,
        const arma::sp_mat& Phi,
        const arma::vec& Pr_b,
        const arma::vec& sigma,
        const arma::vec& theta,
        const arma::vec& lambda,
        double alpha,
        int max_iter_harmony,
        int max_iter_kmeans,
        double epsilon_kmeans,
        double epsilon_harmony,
        int K,
        double block_size,
        const std::vector<int>& B_vec,
        double batch_proportion_cutoff,
        bool verbose,
        int random_state
    );

    arma::mat result() const { return arma::conv_to<arma::mat>::from(Z_corr); }
    arma::mat get_Z_corr() const { return arma::conv_to<arma::mat>::from(Z_corr); }
    arma::mat get_Z_orig() const { return arma::conv_to<arma::mat>::from(Z_orig); }
    arma::mat get_Z_cos() const { return arma::conv_to<arma::mat>::from(Z_corr); }
    arma::mat get_R() const { return arma::conv_to<arma::mat>::from(R); }
    arma::mat get_Y() const { return arma::conv_to<arma::mat>::from(Y); }

    void init_cluster();
    void harmonize(int iter_harmony, bool verbose);
    void cluster();
    void update_R();
    void compute_objective();
    bool check_convergence(int i_type);
    void moe_correct_ridge();

private:
    void allocate_buffers();
    void build_batch_structures(const arma::sp_mat& Phi_in);
    void scatter_add_O(const MATTYPE& Rsub, const arma::uvec& ids, float sign);
};

} // namespace harmony

#endif // HARMONY_HPP
