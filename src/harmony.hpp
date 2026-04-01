// harmonypy - A data alignment algorithm.
// Copyright (C) 2018  Ilya Korsunsky
//               2019  Kamil Slowikowski <kslowikowski@gmail.com>
//
// harmony2 C++ backend — matches the R harmony2 package algorithm.

#ifndef HARMONY_HPP
#define HARMONY_HPP

#include <armadillo>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace harmony {

inline arma::vec find_lambda(double alpha, const arma::vec& cluster_E) {
    arma::vec lambda_vec(cluster_E.n_elem + 1, arma::fill::zeros);
    lambda_vec.subvec(1, lambda_vec.n_elem - 1) = cluster_E * alpha;
    return lambda_vec;
}

inline arma::mat harmony_pow(arma::mat A, const arma::vec& T) {
    for (unsigned c = 0; c < A.n_cols; c++) {
        A.col(c) = arma::pow(A.col(c), T(c));
    }
    return A;
}

arma::mat kmeans_plusplus(const arma::mat& data, int K, std::mt19937& rng);

class Harmony {
public:
    // Data matrices (d x N)
    arma::mat Z_orig;
    arma::mat Z_corr;
    arma::mat Z_cos;

    // Sparse batch indicators
    arma::sp_mat Phi;        // B x N
    arma::sp_mat Phi_t;      // N x B
    arma::sp_mat Phi_moe;    // (B+1) x N
    arma::sp_mat Phi_moe_t;  // N x (B+1)

    arma::vec Pr_b;
    std::vector<arma::uvec> batch_index;  // cell indices per batch

    arma::mat Y;           // d x K centroids
    arma::mat R;           // K x N soft assignments
    arma::mat dist_mat;    // K x N distances

    arma::mat O;           // K x B observed
    arma::mat E;           // K x B expected
    arma::mat W;           // (B+1) x d ridge weights

    arma::vec sigma;       // K
    arma::vec theta;       // B
    arma::vec lambda;      // B+1

    double alpha;
    bool lambda_estimation;

    int N, d, K, B;
    int max_iter_harmony, max_iter_kmeans;
    double epsilon_kmeans, epsilon_harmony;
    double block_size;
    int window_size;
    bool verbose;

    // Covariate structure (for multi-covariate support)
    std::vector<int> B_vec;
    std::vector<unsigned> covariate_bounds;
    double batch_proportion_cutoff;

    // Tracking
    std::vector<double> objective_harmony;
    std::vector<double> objective_kmeans;
    std::vector<double> objective_kmeans_dist;
    std::vector<double> objective_kmeans_entropy;
    std::vector<double> objective_kmeans_cross;
    std::vector<int> kmeans_rounds;

    std::mt19937 rng;
    arma::mat _scale_dist;

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

    arma::mat result() const { return Z_corr; }

    void init_cluster();
    void harmonize(int iter_harmony, bool verbose);
    void cluster();
    void update_R();
    void compute_objective();
    bool check_convergence(int i_type);
    void moe_correct_ridge();

private:
    void allocate_buffers();
    void build_batch_index();
};

} // namespace harmony

#endif // HARMONY_HPP
