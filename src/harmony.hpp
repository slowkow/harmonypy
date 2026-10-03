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
#include <functional>
#include <memory>
#include "thread_pool.hpp"

namespace harmony {

typedef arma::Mat<float> MATTYPE;
typedef arma::Col<float> VECTYPE;
typedef arma::Row<float> ROWTYPE;

inline VECTYPE find_lambda(float alpha, const VECTYPE& cluster_E) {
    VECTYPE lambda_vec(cluster_E.n_elem + 1, arma::fill::zeros);
    lambda_vec.subvec(1, lambda_vec.n_elem - 1) = cluster_E * alpha;
    return lambda_vec;
}

bool objective_converged(float obj_old, float obj_new, float epsilon);
MATTYPE assignment_logits(
    const MATTYPE& distances,
    const VECTYPE& sigma,
    const MATTYPE& E,
    const MATTYPE& O,
    const VECTYPE& theta,
    const arma::Mat<arma::uword>& batch_ids
);
ROWTYPE exponentiate_shifted_logits(MATTYPE& logits);

// Positions [begin, end) of a cell list, all from one group (a batch, or a
// pair of batches). Work is split into these runs, never by thread.
struct CellRun {
    unsigned group;
    unsigned begin;
    unsigned end;
};

// Cells grouped by a key: the cells with key g are
// cells[offsets[g] .. offsets[g + 1]), in increasing order, split into runs.
struct CellGroups {
    std::vector<unsigned> cells;
    std::vector<unsigned> offsets;
    std::vector<CellRun> runs;
};

// The cells that share a batch from covariate a and a batch from covariate b.
struct CovariatePair {
    std::vector<unsigned> batch_a;   // batch from the first covariate, per pair
    std::vector<unsigned> batch_b;   // batch from the second covariate, per pair
    CellGroups groups;               // cells grouped by pair
};

class Harmony {
public:
    MATTYPE Z_orig;
    MATTYPE Z_corr;

    arma::Mat<arma::uword> batch_ids;   // n_cov x N: batch index per covariate
    int n_covariates;
    VECTYPE Pr_b;
    VECTYPE batch_sizes;

    MATTYPE Y;
    MATTYPE R;
    MATTYPE dist_mat;

    MATTYPE O;
    MATTYPE E;

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
    unsigned n_threads;

    std::vector<int> B_vec;
    std::vector<unsigned> covariate_bounds;
    float batch_proportion_cutoff;

    std::vector<float> objective_harmony;
    std::vector<float> objective_kmeans;
    std::vector<float> objective_kmeans_dist;
    std::vector<float> objective_kmeans_entropy;
    std::vector<float> objective_kmeans_cross;
    std::vector<int> kmeans_rounds;

    // Callback for progress messages (set from Python to go through logging)
    std::function<void(const std::string&)> log_fn;

    std::mt19937 rng;

    Harmony(
        const arma::mat& Z,
        const arma::Mat<int64_t>& batch_of_cell,  // n_cov x N
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
        int random_state,
        int ncores = 1,
        std::function<void(const std::string&)> log_fn = nullptr
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
    std::unique_ptr<ThreadPool> pool;

    // batch_groups: the cells of each batch (all covariates).
    // covariate_pairs: one entry per pair of covariates, in (a, b) order.
    CellGroups batch_groups;
    std::vector<CovariatePair> covariate_pairs;
    // Runs of batch_groups for the ridge sums, and each run's partial sum
    // (-1 when its batch has a single run).
    std::vector<CellRun> ridge_runs;
    std::vector<int> ridge_partial;
    unsigned n_ridge_partials = 0;
    // Batch of each cell per covariate, cell-major: cell_batches[j * n_cov + c].
    std::vector<unsigned> cell_batches;
    // Covariate of each batch.
    std::vector<unsigned> batch_covariate;
    // Cells sorted by their combination of batches.
    std::vector<unsigned> cells_by_combination;

    // Scratch space for update_R, reused across blocks.
    std::vector<unsigned> block_cells;
    std::vector<unsigned> block_batches;
    std::vector<unsigned> block_counts;
    std::vector<CellRun> block_runs;
    MATTYPE run_sums;
    std::vector<double> run_error;
    std::vector<double> run_entropy;
    std::vector<unsigned char> run_flags;
    MATTYPE log_div;

    // Objective terms and invariant checks from the last update_R.
    double update_error = 0.0;
    double update_entropy = 0.0;
    unsigned char update_flags = 0;

    void allocate_buffers();
    void build_batch_structures(const arma::Mat<int64_t>& batch_of_cell);
    void build_covariate_pairs();
    template <class F> void run_tasks(size_t n_tasks, size_t work, F&& fn) const;
    void normalise_columns(MATTYPE& X);
    void assign_without_diversity(const char* stage);
    void sum_runs(const MATTYPE& X, const std::vector<unsigned>& cells,
                  const std::vector<CellRun>& runs, size_t first, size_t last, MATTYPE& sums);
    void rebuild_O_E();
    size_t group_block(const unsigned* cells, unsigned n_cells);
    void compute_log_diversity();
    void reassign_block_runs(size_t n_lead_runs);
    void record_objective(double kmeans_error, double entropy);
    MATTYPE kmeans_init(const MATTYPE& X);
    bool kmeans_lloyd_step(MATTYPE& means, const MATTYPE& X,
                           std::vector<unsigned>& assignment, CellGroups& members);
    void batch_coordinate_sums(std::vector<MATTYPE>& RZ, const std::vector<char>& used);
    void apply_corrections(const std::vector<MATTYPE>& M);
    void multi_covariate_totals(const arma::Mat<unsigned char>& kept, MATTYPE& z_all, VECTYPE& cov_sum);
    void check_state(const char* stage) const;
    void check_assignment_update() const;
    void check_objectives(const char* stage) const;
    [[noreturn]] void numerical_error(const char* stage, const char* invariant) const;
};

} // namespace harmony

#endif // HARMONY_HPP
