// harmonypy - C++ backend matching R harmony2 package.
// Copyright (C) 2018  Ilya Korsunsky
//               2019  Kamil Slowikowski <kslowikowski@gmail.com>
//
// Uses custom scatter/gather kernels on a batch_id vector instead of
// sparse Phi matrices. Per-cell work runs on a std::thread pool
// (thread_pool.hpp); matrix products go to BLAS (Accelerate/OpenBLAS).
//
// Work is split into the same tasks for any number of threads (runs of at
// most kCellsPerTask cells from one batch, or one cluster), and per-task
// results are combined in task order, so results do not depend on ncores.

#include "harmony.hpp"
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace harmony {

namespace {

// Cells per task: enough work to amortize scheduling, small enough to balance load.
constexpr unsigned kCellsPerTask = 1024;

// Below this much work (roughly, element operations), run tasks on the calling
// thread: waking the pool would cost more than it saves.
constexpr size_t kMinParallelWork = size_t(1) << 16;

// Invariant failures recorded by the assignment kernel.
constexpr unsigned char kBadNormalizer = 1;
constexpr unsigned char kBadColumnSum = 2;

inline size_t n_chunks(size_t n) { return (n + kCellsPerTask - 1) / kCellsPerTask; }

// Add runs of at most kCellsPerTask positions covering [begin, end).
void append_runs(std::vector<CellRun>& runs, unsigned group, unsigned begin, unsigned end) {
    for (unsigned start = begin; start < end; start += kCellsPerTask)
        runs.push_back({group, start, std::min(end, start + kCellsPerTask)});
}

// Group cells 0..n_cells-1 by key (key[j] < n_keys), keeping cell order within a group.
void group_by_key(CellGroups& groups, const unsigned* key, unsigned n_cells, unsigned n_keys) {
    groups.offsets.assign(n_keys + 1, 0);
    for (unsigned j = 0; j < n_cells; ++j) groups.offsets[key[j] + 1]++;
    std::partial_sum(groups.offsets.begin(), groups.offsets.end(), groups.offsets.begin());
    groups.cells.resize(n_cells);
    std::vector<unsigned> next(groups.offsets.begin(), groups.offsets.end() - 1);
    for (unsigned j = 0; j < n_cells; ++j) groups.cells[next[key[j]]++] = j;
    groups.runs.clear();
    for (unsigned g = 0; g < n_keys; ++g)
        append_runs(groups.runs, g, groups.offsets[g], groups.offsets[g + 1]);
}

// Sum of n floats in the same order as arma::sum (two interleaved partial sums).
inline float accumulate(const float* x, unsigned n) {
    float acc1 = 0.0f, acc2 = 0.0f;
    unsigned j;
    for (j = 1; j < n; j += 2) {
        acc1 += x[j - 1];
        acc2 += x[j];
    }
    if (j - 1 < n) acc1 += x[j - 1];
    return acc1 + acc2;
}

// Softmax of one column of logits, in place, as normalize_log_assignments did:
// shift by the max, exponentiate, divide by the sum. Returns false (leaving the
// column unnormalized) if the sum is not finite and positive.
inline bool softmax_column(float* x, unsigned n) {
    float max_logit = -std::numeric_limits<float>::infinity();
    for (unsigned i = 0; i < n; ++i)
        if (x[i] > max_logit) max_logit = x[i];
    for (unsigned i = 0; i < n; ++i) x[i] = std::exp(x[i] - max_logit);
    const float total = accumulate(x, n);
    if (!(std::isfinite(total) && total > 0.0f)) return false;
    for (unsigned i = 0; i < n; ++i) x[i] = x[i] / total;
    return true;
}

} // namespace

bool objective_converged(float obj_old, float obj_new, float epsilon) {
    if (!std::isfinite(obj_old) || !std::isfinite(obj_new) || !std::isfinite(epsilon))
        return false;
    if (obj_old == 0.0f) return obj_new == 0.0f;

    float delta = (obj_old - obj_new) / std::abs(obj_old);
    return delta >= 0.0f && delta < epsilon;
}

MATTYPE assignment_logits(
    const MATTYPE& distances,
    const VECTYPE& sigma,
    const MATTYPE& E,
    const MATTYPE& O,
    const VECTYPE& theta,
    const arma::Mat<arma::uword>& batch_ids
) {
    MATTYPE logits = -distances;
    logits.each_col() /= sigma;

    MATTYPE log_diversity = arma::log((2 * E) + 1) - arma::log(O + E + 1);
    log_diversity.each_row() %= theta.t();
    for (arma::uword c = 0; c < batch_ids.n_rows; ++c) {
        for (arma::uword j = 0; j < batch_ids.n_cols; ++j) {
            logits.col(j) += log_diversity.col(batch_ids(c, j));
        }
    }
    return logits;
}

ROWTYPE exponentiate_shifted_logits(MATTYPE& logits) {
    logits.each_row() -= arma::max(logits, 0);
    logits = arma::exp(logits);
    return arma::sum(logits, 0);
}

[[noreturn]] void Harmony::numerical_error(const char* stage, const char* invariant) const {
    std::ostringstream oss;
    oss << "Harmony numerical error during " << stage << ": " << invariant
        << " (sigma=[";
    for (arma::uword i = 0; i < sigma.n_elem; ++i) {
        if (i > 0) oss << ", ";
        oss << sigma(i);
    }
    oss << "], theta=[";
    for (arma::uword i = 0; i < theta.n_elem; ++i) {
        if (i > 0) oss << ", ";
        oss << theta(i);
    }
    oss << "], block_size=" << block_size << ", K=" << K << ", N=" << N << ")";
    throw std::runtime_error(oss.str());
}

// =========================================================================
// Task helpers
// =========================================================================

// Run fn(t) for t in [0, n_tasks): on the pool, or inline when the work is small.
template <class F>
void Harmony::run_tasks(size_t n_tasks, size_t work, F&& fn) const {
    if (!pool || pool->size() == 1 || n_tasks <= 1 || work < kMinParallelWork) {
        for (size_t t = 0; t < n_tasks; ++t) fn(t);
        return;
    }
    pool->parallel_for(n_tasks, fn);
}

// sums(:, r) = sum of X(:, cells[i]) over the positions i of run r, for runs [first, last).
void Harmony::sum_runs(const MATTYPE& X, const std::vector<unsigned>& cells,
                       const std::vector<CellRun>& runs, size_t first, size_t last, MATTYPE& sums) {
    const unsigned n_rows = X.n_rows;
    if (sums.n_rows != n_rows || sums.n_cols < runs.size()) sums.set_size(n_rows, runs.size());
    size_t work = 0;
    for (size_t r = first; r < last; ++r) work += runs[r].end - runs[r].begin;
    run_tasks(last - first, work * n_rows, [&](size_t t) {
        const CellRun& run = runs[first + t];
        float* acc = sums.colptr(first + t);
        std::fill(acc, acc + n_rows, 0.0f);
        for (unsigned i = run.begin; i < run.end; ++i) {
            const float* x = X.colptr(cells[i]);
            for (unsigned k = 0; k < n_rows; ++k) acc[k] += x[k];
        }
    });
}

// dst = src(:, cells of group g), copied in parallel.
void Harmony::gather_cells(const MATTYPE& src, const CellGroups& groups, unsigned g, MATTYPE& dst) {
    const unsigned begin = groups.offsets[g];
    const unsigned n = groups.offsets[g + 1] - begin;
    const unsigned n_rows = src.n_rows;
    const unsigned* cells = groups.cells.data() + begin;
    dst.set_size(n_rows, n);
    run_tasks(n_chunks(n), size_t(n) * n_rows, [&](size_t t) {
        const unsigned i0 = t * kCellsPerTask;
        const unsigned i1 = std::min(n, i0 + kCellsPerTask);
        for (unsigned i = i0; i < i1; ++i)
            std::memcpy(dst.colptr(i), src.colptr(cells[i]), n_rows * sizeof(float));
    });
}

// Scale each column to unit L2 norm, exactly as arma::normalise(X, 2, 0).
void Harmony::normalise_columns(MATTYPE& X) {
    const unsigned n_rows = X.n_rows;
    const unsigned n_cols = X.n_cols;
    run_tasks(n_chunks(n_cols), size_t(n_rows) * n_cols, [&](size_t t) {
        const unsigned j0 = t * kCellsPerTask;
        const unsigned j1 = std::min(n_cols, j0 + kCellsPerTask);
        for (unsigned j = j0; j < j1; ++j) {
            float* x = X.colptr(j);
            const VECTYPE column(x, n_rows, false, true);
            const float norm_a = arma::norm(column, 2);
            const float norm_b = (norm_a != 0.0f) ? norm_a : 1.0f;
            for (unsigned i = 0; i < n_rows; ++i) x[i] = x[i] / norm_b;
        }
    });
}

// =========================================================================
// Invariant checks
// =========================================================================

void Harmony::check_objectives(const char* stage) const {
    auto all_finite = [](const std::vector<float>& values) {
        return std::all_of(values.begin(), values.end(), [](float value) {
            return std::isfinite(value);
        });
    };
    if (!all_finite(objective_harmony) || !all_finite(objective_kmeans) ||
        !all_finite(objective_kmeans_dist) || !all_finite(objective_kmeans_entropy) ||
        !all_finite(objective_kmeans_cross))
        numerical_error(stage, "objectives must be finite");
}

void Harmony::check_state(const char* stage) const {
    constexpr unsigned char kNonfinite = 1, kNegative = 2, kColumnSum = 4, kCoordinates = 8;
    const size_t n_tasks = n_chunks(N);
    std::vector<unsigned char> flags(n_tasks, 0);
    run_tasks(n_tasks, size_t(N) * (K + d), [&](size_t t) {
        const unsigned j0 = t * kCellsPerTask;
        const unsigned j1 = std::min<unsigned>(N, j0 + kCellsPerTask);
        unsigned char found = 0;
        for (unsigned j = j0; j < j1; ++j) {
            const float* r = R.colptr(j);
            for (int k = 0; k < K; ++k) {
                if (!std::isfinite(r[k])) found |= kNonfinite;
                else if (r[k] < 0.0f) found |= kNegative;
            }
            const float total = accumulate(r, K);
            if (!std::isfinite(total) || std::abs(total - 1.0f) > 1e-4f) found |= kColumnSum;
            const float* z = Z_corr.colptr(j);
            for (int i = 0; i < d; ++i)
                if (!std::isfinite(z[i])) found |= kCoordinates;
        }
        flags[t] = found;
    });
    unsigned char found = 0;
    for (unsigned char f : flags) found |= f;

    if (found & kNonfinite) numerical_error(stage, "assignments must be finite");
    if (found & kNegative) numerical_error(stage, "assignments must be nonnegative");
    if (found & kColumnSum) numerical_error(stage, "assignment columns must sum to one");
    check_objectives(stage);
    if (found & kCoordinates) numerical_error(stage, "corrected coordinates must be finite");
}

// After update_R, the assignment kernel has checked every column of R, and
// Z_corr is unchanged since the last full check.
void Harmony::check_assignment_update() const {
    const char* stage = "assignment update";
    if (update_flags & kBadColumnSum) numerical_error(stage, "assignment columns must sum to one");
    check_objectives(stage);
}

// =========================================================================
// K-means initialization (matches R harmony2)
// =========================================================================

MATTYPE Harmony::kmeans_init(const MATTYPE& X) {
    std::uniform_real_distribution<float> uniform01(0.0f, 1.0f);
    MATTYPE Y(X.n_rows, K);
    for (int i = 0; i < K; ++i) {
        int idx = static_cast<int>(std::round(uniform01(rng) * N));
        if (idx >= N) idx = N - 1;
        Y.col(i) = X.col(idx);
    }

    std::set<unsigned> chosen;
    VECTYPE random_numbers(N, arma::fill::none);
    VECTYPE prob(N, arma::fill::none);
    float* u = random_numbers.memptr();
    float* p = prob.memptr();
    const size_t n_tasks = n_chunks(N);
    std::vector<float> task_best(n_tasks);
    std::vector<unsigned> task_index(n_tasks);
    for (int i = 0; i < K; ++i) {
        const ROWTYPE similarity = Y.col(i).t() * X;
        const float* sim = similarity.memptr();
        // A local copy keeps the generator state in registers.
        std::mt19937 draws = rng;
        for (int j = 0; j < N; ++j) u[j] = uniform01(draws);
        rng = draws;

        // prob = -log(u) / distance, and its maximum.
        run_tasks(n_tasks, size_t(N) * 16, [&](size_t t) {
            const unsigned j0 = t * kCellsPerTask;
            const unsigned j1 = std::min<unsigned>(N, j0 + kCellsPerTask);
            float max_prob = -std::numeric_limits<float>::infinity();
            for (unsigned j = j0; j < j1; ++j) {
                const float distance = std::abs((1.0f - sim[j]) * 2.0f);
                p[j] = (-std::log(u[j])) / (distance + 1e-10f);
                if (p[j] > max_prob) max_prob = p[j];
            }
            task_best[t] = max_prob;
        });
        float max_prob = -std::numeric_limits<float>::infinity();
        for (float value : task_best)
            if (value > max_prob) max_prob = value;
        for (auto idx : chosen) prob(idx) = max_prob;

        // The first cell with the smallest prob, as prob.index_min().
        run_tasks(n_tasks, size_t(N), [&](size_t t) {
            const unsigned j0 = t * kCellsPerTask;
            const unsigned j1 = std::min<unsigned>(N, j0 + kCellsPerTask);
            float min_prob = std::numeric_limits<float>::infinity();
            unsigned min_index = 0;
            for (unsigned j = j0; j < j1; ++j) {
                if (p[j] < min_prob) {
                    min_prob = p[j];
                    min_index = j;
                }
            }
            task_best[t] = min_prob;
            task_index[t] = min_index;
        });
        float min_prob = std::numeric_limits<float>::infinity();
        unsigned best = 0;
        for (size_t t = 0; t < n_tasks; ++t) {
            if (task_best[t] < min_prob) {
                min_prob = task_best[t];
                best = task_index[t];
            }
        }
        while (chosen.count(best)) {
            prob(best) = prob.max();
            best = prob.index_min();
        }
        chosen.insert(best);
        Y.col(i) = X.col(best);
    }

    // arma::kmeans fails on non-finite data and leaves the means empty.
    if (!X.is_finite()) {
        Y.reset();
        return Y;
    }
    std::vector<unsigned> assignment(N);
    CellGroups members;
    for (int i = 0; i < 10; ++i) {
        if (!kmeans_lloyd_step(Y, X, assignment, members)) {
            Y.reset();
            break;
        }
    }

    return Y;
}

// One iteration of arma::kmeans(means, X, k, arma::keep_existing, 1, false),
// with the same arithmetic and tie-breaking, split across threads: each cell
// goes to its nearest mean, and each mean's cells are summed in cell order.
// Returns false where Armadillo fails (the caller then empties the means).
bool Harmony::kmeans_lloyd_step(MATTYPE& means, const MATTYPE& X,
                                std::vector<unsigned>& assignment, CellGroups& members) {
    if (means.is_empty()) return false;
    const unsigned n_dims = X.n_rows;
    const unsigned n_means = means.n_cols;
    const unsigned n_cells = X.n_cols;

    // Squared Euclidean distance as arma's distance<eT, 1>::eval, for up to
    // kMeansPerPass means at once: (x - mean)^2 summed in two interleaved
    // partial sums. means_t(g, i) puts dimension i of every mean side by side.
    constexpr unsigned kMeansPerPass = 16;
    const MATTYPE means_t = means.t();
    run_tasks(n_chunks(n_cells), size_t(n_cells) * n_means * n_dims, [&](size_t t) {
        const unsigned j0 = t * kCellsPerTask;
        const unsigned j1 = std::min(n_cells, j0 + kCellsPerTask);
        float acc1[kMeansPerPass];
        float acc2[kMeansPerPass];
        for (unsigned j = j0; j < j1; ++j) {
            const float* x = X.colptr(j);
            float min_dist = std::numeric_limits<float>::infinity();
            unsigned best = 0;
            for (unsigned g0 = 0; g0 < n_means; g0 += kMeansPerPass) {
                const unsigned n_pass = std::min(kMeansPerPass, n_means - g0);
                for (unsigned g = 0; g < n_pass; ++g) {
                    acc1[g] = 0.0f;
                    acc2[g] = 0.0f;
                }
                unsigned a, b;
                for (a = 0, b = 1; b < n_dims; a += 2, b += 2) {
                    const float x_a = x[a];
                    const float x_b = x[b];
                    const float* mean_a = means_t.colptr(a) + g0;
                    const float* mean_b = means_t.colptr(b) + g0;
                    for (unsigned g = 0; g < n_pass; ++g) {
                        float tmp_a = x_a;
                        float tmp_b = x_b;
                        tmp_a -= mean_a[g];
                        tmp_b -= mean_b[g];
                        acc1[g] += tmp_a * tmp_a;
                        acc2[g] += tmp_b * tmp_b;
                    }
                }
                if (a < n_dims) {
                    const float x_a = x[a];
                    const float* mean_a = means_t.colptr(a) + g0;
                    for (unsigned g = 0; g < n_pass; ++g) {
                        const float tmp_a = x_a - mean_a[g];
                        acc1[g] += tmp_a * tmp_a;
                    }
                }
                for (unsigned g = 0; g < n_pass; ++g) {
                    const float dist = acc1[g] + acc2[g];
                    if (dist < min_dist) {
                        min_dist = dist;
                        best = g0 + g;
                    }
                }
            }
            assignment[j] = best;
        }
    });

    group_by_key(members, assignment.data(), n_cells, n_means);
    MATTYPE sums(n_dims, n_means);
    run_tasks(n_means, size_t(n_cells) * n_dims, [&](size_t g) {
        float* sum = sums.colptr(g);
        std::fill(sum, sum + n_dims, 0.0f);
        for (unsigned i = members.offsets[g]; i < members.offsets[g + 1]; ++i) {
            const float* x = X.colptr(members.cells[i]);
            for (unsigned k = 0; k < n_dims; ++k) sum[k] += x[k];
        }
    });

    MATTYPE new_means(n_dims, n_means);
    std::vector<unsigned> counts(n_means), last_cell(n_means, 0);
    for (unsigned g = 0; g < n_means; ++g) {
        counts[g] = members.offsets[g + 1] - members.offsets[g];
        if (counts[g] > 0) last_cell[g] = members.cells[members.offsets[g + 1] - 1];
        for (unsigned k = 0; k < n_dims; ++k)
            new_means(k, g) = (counts[g] >= 1) ? (sums(k, g) / float(counts[g])) : 0.0f;
    }

    // Armadillo's recovery for means with no cells: take the last cell of a
    // mean with at least two cells, in decreasing order of mean index.
    std::vector<unsigned> dead, live;
    for (unsigned g = 0; g < n_means; ++g)
        if (counts[g] == 0) dead.push_back(g);
    if (!dead.empty()) {
        for (unsigned g = n_means; g-- > 0;)
            if (counts[g] >= 2) live.push_back(g);
        if (live.empty()) return false;
        size_t n_used = 0;
        for (unsigned dead_g : dead) {
            unsigned proposed = 0;
            if (n_used < live.size()) {
                const unsigned live_g = live[n_used++];
                if (live_g == dead_g) return false;
                proposed = last_cell[live_g];
            } else {
                proposed = arma::as_scalar(
                    arma::randi<arma::uvec>(1, arma::distr_param(0, static_cast<int>(n_cells) - 1)));
            }
            if (proposed >= n_cells) return false;
            new_means.col(dead_g) = X.col(proposed);
        }
    }

    if (!new_means.is_finite()) return false;
    means = new_means;
    return true;
}

// =========================================================================
// Constructor
// =========================================================================

Harmony::Harmony(
    const arma::mat& Z,
    const arma::Mat<int64_t>& batch_of_cell,
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
    int random_state,
    int ncores,
    std::function<void(const std::string&)> log_fn_in
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
    log_fn(std::move(log_fn_in)),
    rng(random_state)
{
    // ncores <= 0 means one thread per available core.
    unsigned threads = ncores > 0 ? static_cast<unsigned>(ncores) : std::thread::hardware_concurrency();
    n_threads = std::max(1u, threads);
    pool = std::make_unique<ThreadPool>(n_threads);

    Z_orig = arma::conv_to<MATTYPE>::from(Z);

    Pr_b = arma::conv_to<VECTYPE>::from(Pr_b_in);
    N = Z.n_cols;
    d = Z.n_rows;
    B = 0;
    for (auto v : B_vec) B += v;

    Z_corr = Z_orig;
    normalise_columns(Z_corr);

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
        covariate_bounds.resize(B_vec.size());
        std::partial_sum(B_vec.begin(), B_vec.end(), covariate_bounds.begin());
    } else {
        covariate_bounds.push_back(B_vec.front());
    }

    build_batch_structures(batch_of_cell);
    allocate_buffers();

    if (verbose && log_fn) log_fn("Computing initial centroids...");
    init_cluster();
    check_state("initialization");
    if (verbose && log_fn) log_fn("Initialization complete.");
    harmonize(max_iter_harmony, verbose);
    check_state("return");

    // The worker threads are not needed once the result is ready.
    pool.reset();
}

void Harmony::build_batch_structures(const arma::Mat<int64_t>& batch_of_cell) {
    // batch_of_cell is n_cov x N (int64). Each row c contains the batch
    // index for covariate c, with values in [offset_c, offset_c + n_levels_c).
    n_covariates = batch_of_cell.n_rows;
    if (n_covariates != static_cast<int>(B_vec.size()) || static_cast<int>(batch_of_cell.n_cols) != N)
        throw std::invalid_argument("batch_of_cell must be n_covariates x N");
    batch_ids.set_size(n_covariates, N);
    cell_batches.resize(static_cast<size_t>(N) * n_covariates);
    for (int j = 0; j < N; ++j) {
        for (int c = 0; c < n_covariates; ++c) {
            const int64_t first = c == 0 ? 0 : covariate_bounds[c - 1];
            const int64_t b = batch_of_cell(c, j);
            if (b < first || b >= static_cast<int64_t>(covariate_bounds[c]))
                throw std::invalid_argument("batch index out of range for its covariate");
            batch_ids(c, j) = static_cast<arma::uword>(b);
            cell_batches[static_cast<size_t>(j) * n_covariates + c] = static_cast<unsigned>(b);
        }
    }

    // Group the cells of each batch. Batch indices are disjoint across
    // covariates, so every cell appears once per covariate.
    batch_groups.offsets.assign(B + 1, 0);
    for (unsigned b : cell_batches) batch_groups.offsets[b + 1]++;
    std::partial_sum(batch_groups.offsets.begin(), batch_groups.offsets.end(),
                     batch_groups.offsets.begin());
    batch_groups.cells.resize(cell_batches.size());
    std::vector<unsigned> next(batch_groups.offsets.begin(), batch_groups.offsets.end() - 1);
    for (int j = 0; j < N; ++j)
        for (int c = 0; c < n_covariates; ++c)
            batch_groups.cells[next[cell_batches[static_cast<size_t>(j) * n_covariates + c]]++] = j;
    batch_groups.runs.clear();
    for (int b = 0; b < B; ++b)
        append_runs(batch_groups.runs, b, batch_groups.offsets[b], batch_groups.offsets[b + 1]);

    batch_covariate.resize(B);
    for (int c = 0; c < n_covariates; ++c)
        for (unsigned b = c == 0 ? 0 : covariate_bounds[c - 1]; b < covariate_bounds[c]; ++b)
            batch_covariate[b] = c;

    // Count cells in each covariate level.
    batch_sizes.set_size(B);
    for (int b = 0; b < B; ++b)
        batch_sizes(b) = static_cast<float>(batch_groups.offsets[b + 1] - batch_groups.offsets[b]);

    if (n_covariates > 1) build_covariate_pairs();
}

// For each pair of covariates, group cells by their pair of batches. The ridge
// correction needs these totals for every pair of kept batches.
void Harmony::build_covariate_pairs() {
    covariate_pairs.clear();
    std::vector<unsigned> pair_of_cell(N);
    for (int a = 0; a < n_covariates; ++a) {
        for (int b = a + 1; b < n_covariates; ++b) {
            CovariatePair pair;
            std::unordered_map<uint64_t, unsigned> pair_index;
            for (int j = 0; j < N; ++j) {
                const unsigned batch_a = cell_batches[static_cast<size_t>(j) * n_covariates + a];
                const unsigned batch_b = cell_batches[static_cast<size_t>(j) * n_covariates + b];
                const uint64_t key = (static_cast<uint64_t>(batch_a) << 32) | batch_b;
                auto inserted = pair_index.emplace(key, static_cast<unsigned>(pair.batch_a.size()));
                if (inserted.second) {
                    pair.batch_a.push_back(batch_a);
                    pair.batch_b.push_back(batch_b);
                }
                pair_of_cell[j] = inserted.first->second;
            }
            group_by_key(pair.groups, pair_of_cell.data(), N, pair.batch_a.size());
            covariate_pairs.push_back(std::move(pair));
        }
    }
}

void Harmony::allocate_buffers() {
    dist_mat.zeros(K, N);
    O.zeros(K, B);
    E.zeros(K, B);
    R.zeros(K, N);
    Y.zeros(d, K);
    log_div.zeros(K, B);
}

// =========================================================================
// init_cluster
// =========================================================================

// dist_mat holds Y' * Z_corr. Turn it into distances 2 * (1 - cosine) and set
// R = softmax(-dist / sigma) for every cell, without the diversity penalty.
void Harmony::assign_without_diversity(const char* stage) {
    const size_t n_tasks = n_chunks(N);
    std::vector<unsigned char> flags(n_tasks, 0);
    const float* s = sigma.memptr();
    run_tasks(n_tasks, size_t(N) * K * 8, [&](size_t t) {
        const unsigned j0 = t * kCellsPerTask;
        const unsigned j1 = std::min<unsigned>(N, j0 + kCellsPerTask);
        unsigned char found = 0;
        for (unsigned j = j0; j < j1; ++j) {
            float* dist = dist_mat.colptr(j);
            float* logits = R.colptr(j);
            for (int k = 0; k < K; ++k) {
                const float distance = 2.0f * (1.0f - dist[k]);
                dist[k] = distance;
                logits[k] = (-distance) / s[k];
            }
            if (!softmax_column(logits, K)) found |= kBadNormalizer;
        }
        flags[t] = found;
    });
    for (unsigned char f : flags)
        if (f & kBadNormalizer)
            numerical_error(stage, "assignment normalizers must be finite and positive");
}

// E = rowsum(R) * Pr_b' and O(:, b) = the sum of R over the cells of batch b.
void Harmony::rebuild_O_E() {
    const size_t n_tasks = n_chunks(N);
    MATTYPE task_sums(K, n_tasks);
    run_tasks(n_tasks, size_t(N) * K, [&](size_t t) {
        const unsigned j0 = t * kCellsPerTask;
        const unsigned j1 = std::min<unsigned>(N, j0 + kCellsPerTask);
        float* acc = task_sums.colptr(t);
        std::fill(acc, acc + K, 0.0f);
        for (unsigned j = j0; j < j1; ++j) {
            const float* r = R.colptr(j);
            for (int k = 0; k < K; ++k) acc[k] += r[k];
        }
    });
    VECTYPE totals(K, arma::fill::zeros);
    for (size_t t = 0; t < n_tasks; ++t) totals += task_sums.col(t);
    E = totals * Pr_b.t();

    MATTYPE sums;
    sum_runs(R, batch_groups.cells, batch_groups.runs, 0, batch_groups.runs.size(), sums);
    O.zeros();
    for (size_t r = 0; r < batch_groups.runs.size(); ++r)
        O.col(batch_groups.runs[r].group) += sums.col(r);
}

void Harmony::init_cluster() {
    Y = kmeans_init(Z_corr);
    Y = arma::normalise(Y, 2, 0);

    dist_mat = Y.t() * Z_corr;
    assign_without_diversity("initialization");
    rebuild_O_E();

    compute_objective();
    objective_harmony.push_back(objective_kmeans.back());
}

// =========================================================================
// compute_objective
// =========================================================================

// Add the cross-entropy term to kmeans_error and entropy (sums over cells)
// and record the objective.
void Harmony::record_objective(double kmeans_error, double entropy) {
    const float norm_const = 2000.0f / static_cast<float>(N);

    double cross_entropy = 0.0;
    for (int b = 0; b < B; ++b) {
        for (int k = 0; k < K; ++k) {
            const float e = E(k, b);
            const float o = O(k, b);
            const float twice = 2 * e;
            const float ratio = (o + e + 1) / (twice + 1);
            float term = std::log(ratio);
            term = term * theta(b);
            term = term * sigma(k);
            term = term * o;
            cross_entropy += term;
        }
    }

    const float kmeans_error_f = static_cast<float>(kmeans_error);
    const float entropy_f = static_cast<float>(entropy);
    const float cross_entropy_f = static_cast<float>(cross_entropy);
    objective_kmeans.push_back((kmeans_error_f + entropy_f + cross_entropy_f) * norm_const);
    objective_kmeans_dist.push_back(kmeans_error_f * norm_const);
    objective_kmeans_entropy.push_back(entropy_f * norm_const);
    objective_kmeans_cross.push_back(cross_entropy_f * norm_const);
}

void Harmony::compute_objective() {
    const size_t n_tasks = n_chunks(N);
    std::vector<double> errors(n_tasks), entropies(n_tasks);
    const float* s = sigma.memptr();
    run_tasks(n_tasks, size_t(N) * K * 8, [&](size_t t) {
        const unsigned j0 = t * kCellsPerTask;
        const unsigned j1 = std::min<unsigned>(N, j0 + kCellsPerTask);
        double error = 0.0, entropy = 0.0;
        for (unsigned j = j0; j < j1; ++j) {
            const float* r = R.colptr(j);
            const float* dist = dist_mat.colptr(j);
            for (int k = 0; k < K; ++k) {
                error += double(r[k]) * dist[k];
                if (r[k] > 0) entropy += double(s[k]) * (r[k] * std::log(r[k]));
            }
        }
        errors[t] = error;
        entropies[t] = entropy;
    });
    double kmeans_error = 0.0, entropy = 0.0;
    for (size_t t = 0; t < n_tasks; ++t) {
        kmeans_error += errors[t];
        entropy += entropies[t];
    }
    record_objective(kmeans_error, entropy);
}

// =========================================================================
// harmonize / cluster
// =========================================================================

void Harmony::harmonize(int iter_harmony, bool verbose_flag) {
    bool converged = false;
    for (int i = 1; i <= iter_harmony; ++i) {
        if (verbose_flag && log_fn) {
            std::ostringstream oss;
            oss << "Iteration " << i << " of " << iter_harmony;
            log_fn(oss.str());
        }

        cluster();
        moe_correct_ridge();

        converged = check_convergence(1);
        if (converged) {
            if (verbose_flag && log_fn) {
                std::ostringstream oss;
                oss << "Converged after " << i << " iteration"
                    << (i > 1 ? "s" : "");
                log_fn(oss.str());
            }
            break;
        }
    }
    if (verbose_flag && !converged && log_fn)
        log_fn("Stopped before convergence");
}

void Harmony::cluster() {
    if (objective_harmony.size() > 1) {
        normalise_columns(Z_corr);
        dist_mat = Y.t() * Z_corr;
        assign_without_diversity("cluster initialization");
        rebuild_O_E();
    }

    int rounds = 0;
    for (int i = 0; i < max_iter_kmeans; ++i) {
        update_R();
        record_objective(update_error, update_entropy);
        check_assignment_update();

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

// Group the block's cells by batch, once per covariate: positions
// [c * n_cells, (c + 1) * n_cells) of block_cells hold the cells sorted
// (stably) by their batch for covariate c, and block_runs covers covariate 0
// first. Returns the number of covariate-0 runs.
size_t Harmony::group_block(const unsigned* cells, unsigned n_cells) {
    const unsigned n_cov = n_covariates;
    block_cells.resize(static_cast<size_t>(n_cells) * n_cov);
    block_batches.resize(static_cast<size_t>(n_cells) * n_cov);
    for (unsigned i = 0; i < n_cells; ++i) {
        const unsigned* batches = &cell_batches[static_cast<size_t>(cells[i]) * n_cov];
        for (unsigned c = 0; c < n_cov; ++c) block_batches[static_cast<size_t>(c) * n_cells + i] = batches[c];
    }

    block_runs.clear();
    block_counts.assign(B, 0);
    size_t n_lead_runs = 0;
    for (unsigned c = 0; c < n_cov; ++c) {
        const unsigned first = c == 0 ? 0 : covariate_bounds[c - 1];
        const unsigned last = covariate_bounds[c];
        const unsigned* batches = &block_batches[static_cast<size_t>(c) * n_cells];
        for (unsigned i = 0; i < n_cells; ++i) block_counts[batches[i]]++;
        unsigned position = c * n_cells;
        for (unsigned b = first; b < last; ++b) {
            const unsigned count = block_counts[b];
            block_counts[b] = position;
            append_runs(block_runs, b, position, position + count);
            position += count;
        }
        for (unsigned i = 0; i < n_cells; ++i) block_cells[block_counts[batches[i]]++] = cells[i];
        if (c == 0) n_lead_runs = block_runs.size();
    }
    return n_lead_runs;
}

// log_div(k, b) = theta(b) * (log(2 E + 1) - log(O + E + 1)), the log of the
// diversity penalty, as in assignment_logits.
void Harmony::compute_log_diversity() {
    run_tasks(B, size_t(K) * B * 16, [&](size_t b) {
        const float* e = E.colptr(b);
        const float* o = O.colptr(b);
        float* out = log_div.colptr(b);
        const float weight = theta(b);
        for (int k = 0; k < K; ++k) {
            const float twice = 2 * e[k];
            const float log_expected = std::log(twice + 1);
            const float total = o[k] + e[k];
            const float log_observed = std::log(total + 1);
            const float diff = log_expected - log_observed;
            out[k] = diff * weight;
        }
    });
}

// Reassign the cells of the block's covariate-0 runs (one batch per run)
// with E and O frozen: logits from the distances plus the diversity penalty
// of each of the cell's batches, then a softmax. For each run, also record
// the sum of the new assignments and the cells' objective terms.
void Harmony::reassign_block_runs(size_t n_lead_runs) {
    const float* s = sigma.memptr();
    const unsigned n_cov = n_covariates;
    size_t n_cells = 0;
    for (size_t r = 0; r < n_lead_runs; ++r) n_cells += block_runs[r].end - block_runs[r].begin;
    run_tasks(n_lead_runs, n_cells * K * 8, [&](size_t r) {
        const CellRun& run = block_runs[r];
        const float* lead_div = log_div.colptr(run.group);
        float* sums = run_sums.colptr(r);
        std::fill(sums, sums + K, 0.0f);
        double error = 0.0, entropy = 0.0;
        unsigned char found = 0;
        for (unsigned i = run.begin; i < run.end; ++i) {
            const unsigned j = block_cells[i];
            const float* dist = dist_mat.colptr(j);
            float* logits = R.colptr(j);
            for (int k = 0; k < K; ++k) {
                const float scaled = (-dist[k]) / s[k];
                logits[k] = scaled + lead_div[k];
            }
            for (unsigned c = 1; c < n_cov; ++c) {
                const float* div = log_div.colptr(cell_batches[static_cast<size_t>(j) * n_cov + c]);
                for (int k = 0; k < K; ++k) logits[k] = logits[k] + div[k];
            }

            // Softmax. With p = e / total and log p = shifted - log(total), the
            // entropy term sum(sigma p log p) needs only these two sums.
            float max_logit = -std::numeric_limits<float>::infinity();
            for (int k = 0; k < K; ++k)
                if (logits[k] > max_logit) max_logit = logits[k];
            double weighted = 0.0, weighted_shifted = 0.0;
            for (int k = 0; k < K; ++k) {
                const float shifted = logits[k] - max_logit;
                const float e = std::exp(shifted);
                logits[k] = e;
                const double w = double(s[k]) * e;
                weighted += w;
                weighted_shifted += w * shifted;
            }
            const float total = accumulate(logits, K);
            if (!(std::isfinite(total) && total > 0.0f)) {
                found |= kBadNormalizer;
                continue;
            }
            double cell_error = 0.0;
            for (int k = 0; k < K; ++k) {
                const float p = logits[k] / total;
                logits[k] = p;
                sums[k] += p;
                cell_error += double(p) * dist[k];
            }
            const float column_sum = accumulate(logits, K);
            if (!std::isfinite(column_sum) || std::abs(column_sum - 1.0f) > 1e-4f)
                found |= kBadColumnSum;
            error += cell_error;
            entropy += (weighted_shifted - std::log(double(total)) * weighted) / total;
        }
        run_error[r] = error;
        run_entropy[r] = entropy;
        run_flags[r] = found;
    });
}

void Harmony::update_R() {
    std::vector<unsigned> indices_vec(N);
    std::iota(indices_vec.begin(), indices_vec.end(), 0);
    std::shuffle(indices_vec.begin(), indices_vec.end(), rng);

    unsigned n_blocks = static_cast<unsigned>(std::ceil(1.0 / block_size));
    unsigned cells_per_block = std::max(1u, static_cast<unsigned>(N * block_size));

    update_error = 0.0;
    update_entropy = 0.0;
    update_flags = 0;
    VECTYPE totals(K);

    for (unsigned i = 0; i < n_blocks; ++i) {
        unsigned idx_min = i * cells_per_block;
        unsigned idx_max = ((i + 1) * cells_per_block) - 1;
        if (i == n_blocks - 1) idx_max = N - 1;
        if (idx_min >= static_cast<unsigned>(N)) break;

        const size_t n_lead_runs = group_block(indices_vec.data() + idx_min, idx_max - idx_min + 1);
        const size_t n_runs = block_runs.size();
        if (run_sums.n_rows != static_cast<unsigned>(K) || run_sums.n_cols < n_runs)
            run_sums.set_size(K, n_runs);
        run_error.resize(n_lead_runs);
        run_entropy.resize(n_lead_runs);
        run_flags.resize(n_lead_runs);

        // Take the block's cells out of O and E.
        sum_runs(R, block_cells, block_runs, 0, n_runs, run_sums);
        totals.zeros();
        for (size_t r = 0; r < n_runs; ++r) {
            const float* sums = run_sums.colptr(r);
            float* o = O.colptr(block_runs[r].group);
            for (int k = 0; k < K; ++k) o[k] -= sums[k];
            if (r < n_lead_runs)
                for (int k = 0; k < K; ++k) totals[k] += sums[k];
        }
        for (int b = 0; b < B; ++b) {
            float* e = E.colptr(b);
            for (int k = 0; k < K; ++k) {
                const float removed = totals[k] * Pr_b[b];
                e[k] -= removed;
            }
        }

        // E and O stay frozen while every cell in this block is reassigned.
        compute_log_diversity();
        reassign_block_runs(n_lead_runs);
        for (size_t r = 0; r < n_lead_runs; ++r)
            if (run_flags[r] & kBadNormalizer)
                numerical_error("assignment update", "assignment normalizers must be finite and positive");

        // Put the block's cells back.
        sum_runs(R, block_cells, block_runs, n_lead_runs, n_runs, run_sums);
        totals.zeros();
        for (size_t r = 0; r < n_runs; ++r) {
            const float* sums = run_sums.colptr(r);
            float* o = O.colptr(block_runs[r].group);
            for (int k = 0; k < K; ++k) o[k] += sums[k];
            if (r < n_lead_runs)
                for (int k = 0; k < K; ++k) totals[k] += sums[k];
        }
        for (int b = 0; b < B; ++b) {
            float* e = E.colptr(b);
            for (int k = 0; k < K; ++k) {
                const float added = totals[k] * Pr_b[b];
                e[k] += added;
            }
        }

        for (size_t r = 0; r < n_lead_runs; ++r) {
            update_error += run_error[r];
            update_entropy += run_entropy[r];
            update_flags |= run_flags[r];
        }
    }
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
        return objective_converged(obj_old, obj_new, epsilon_harmony);
    }
    return true;
}

// =========================================================================
// moe_correct_ridge
// =========================================================================

/**
 * Totals for groups that share cells, such as lab A and Monday: for cluster k,
 * a cell counts in the intercept terms if any of its groups is kept
 * (kept(k, b) != 0), and counts once even if several are. cov_sum(k) is the
 * sum of R(k, :) over those cells and z_all(:, k) the matching weighted
 * coordinate sum, computed as one masked GEMM per chunk of cells.
 */
void Harmony::multi_covariate_totals(const arma::Mat<unsigned char>& kept, MATTYPE& z_all, VECTYPE& cov_sum) {
    constexpr unsigned kCellsPerProduct = 16 * kCellsPerTask;
    const unsigned n_cov = n_covariates;
    z_all.zeros(d, K);
    MATTYPE masked(K, std::min<unsigned>(kCellsPerProduct, N));
    MATTYPE task_sums(K, n_chunks(N));
    const unsigned char* kept_mem = kept.memptr();
    for (unsigned j0 = 0; j0 < static_cast<unsigned>(N); j0 += kCellsPerProduct) {
        const unsigned n = std::min<unsigned>(kCellsPerProduct, N - j0);
        run_tasks(n_chunks(n), size_t(n) * K * n_cov, [&](size_t t) {
            const unsigned i0 = t * kCellsPerTask;
            const unsigned i1 = std::min(n, i0 + kCellsPerTask);
            float* sums = task_sums.colptr(j0 / kCellsPerTask + t);
            std::fill(sums, sums + K, 0.0f);
            for (unsigned i = i0; i < i1; ++i) {
                const unsigned j = j0 + i;
                const unsigned* batches = &cell_batches[static_cast<size_t>(j) * n_cov];
                const float* r = R.colptr(j);
                float* m = masked.colptr(i);
                for (int k = 0; k < K; ++k) {
                    unsigned char any = 0;
                    for (unsigned c = 0; c < n_cov; ++c) any |= kept_mem[static_cast<size_t>(batches[c]) * K + k];
                    m[k] = any ? r[k] : 0.0f;
                    sums[k] += m[k];
                }
            }
        });
        const MATTYPE Z_chunk(const_cast<float*>(Z_orig.colptr(j0)), d, n, false, true);
        if (n == masked.n_cols) {
            z_all += Z_chunk * masked.t();
        } else {
            z_all += Z_chunk * masked.cols(0, n - 1).t();
        }
    }
    cov_sum.zeros(K);
    for (arma::uword t = 0; t < task_sums.n_cols; ++t) cov_sum += task_sums.col(t);
}

void Harmony::moe_correct_ridge() {
    const bool multiple_covariates = B_vec.size() > 1;

    // Choose the batches each cluster corrects.
    std::vector<std::vector<unsigned>> keeps(K);
    std::vector<char> active(K, 0);
    for (int k = 0; k < K; ++k) {
        VECTYPE avg_R = O.row(k).t() / batch_sizes;

        std::vector<unsigned>& keep = keeps[k];
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

        active[k] = active_covariates > 0;
    }

    // ZR[b] = Z_orig(:, cells of b) * R(:, cells of b)' (d x K) holds the
    // R-weighted coordinate sums of batch b for every cluster: one GEMM per
    // batch instead of one pass over Z_orig per cluster.
    std::vector<char> used(B, 0);
    for (int k = 0; k < K; ++k)
        if (active[k])
            for (unsigned b : keeps[k]) used[b] = 1;
    std::vector<MATTYPE> ZR(B);
    MATTYPE Z_batch, R_batch;
    for (int b = 0; b < B; ++b) {
        if (!used[b]) continue;
        gather_cells(Z_orig, batch_groups, b, Z_batch);
        gather_cells(R, batch_groups, b, R_batch);
        ZR[b] = Z_batch * R_batch.t();
    }

    // With several covariates, a cell belongs to one batch per covariate, so
    // the intercept terms and the overlap of kept batches need extra totals.
    MATTYPE z_all;
    VECTYPE cov_sum;
    std::vector<MATTYPE> pair_sums(covariate_pairs.size());
    if (multiple_covariates) {
        arma::Mat<unsigned char> kept(K, B, arma::fill::zeros);
        for (int k = 0; k < K; ++k)
            if (active[k])
                for (unsigned b : keeps[k]) kept(k, b) = 1;
        multi_covariate_totals(kept, z_all, cov_sum);

        MATTYPE sums;
        for (size_t p = 0; p < covariate_pairs.size(); ++p) {
            const CellGroups& groups = covariate_pairs[p].groups;
            sum_runs(R, groups.cells, groups.runs, 0, groups.runs.size(), sums);
            pair_sums[p].zeros(K, covariate_pairs[p].batch_a.size());
            for (size_t r = 0; r < groups.runs.size(); ++r)
                pair_sums[p].col(groups.runs[r].group) += sums.col(r);
        }
    }

    // M[b](:, k) is the correction direction of batch b in cluster k.
    std::vector<MATTYPE> M(B);
    // Position of each kept batch in the dense block (>= 1; 0 is the
    // intercept) or in the diagonal block (>= 0); -1 when not kept.
    std::vector<int> dense_slot(B, -1), diag_slot(B, -1);
    for (int k = 0; k < K; ++k) {
        if (!active[k]) continue;

        const std::vector<unsigned>& keep = keeps[k];
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

        // The ridge system has one row for the intercept and one per kept batch:
        //
        //   cov = [ total  O_k'              ] + diag(lamb_vec)
        //         [ O_k    diag(O_k) + overlap ]
        //
        // where overlap holds the cells shared by kept batches of different
        // covariates. Batches of one covariate share no cells, so the rows of
        // the covariate with the most kept batches form a diagonal block D.
        // Eliminating D (a Schur complement) leaves a small dense system: a
        // single equation when there is one covariate, which is the arrowhead
        // inverse R harmony uses.
        std::vector<unsigned> kept_per_covariate(n_covariates, 0);
        for (unsigned b : keep) kept_per_covariate[batch_covariate[b]]++;
        const unsigned diag_covariate = std::max_element(kept_per_covariate.begin(), kept_per_covariate.end())
                                        - kept_per_covariate.begin();
        unsigned n_dense = 1, n_diag = 0;
        for (unsigned b : keep) {
            if (batch_covariate[b] == diag_covariate) diag_slot[b] = n_diag++;
            else dense_slot[b] = n_dense++;
        }

        MATTYPE A(n_dense, n_dense, arma::fill::zeros);   // intercept and other covariates
        MATTYPE C(n_dense, n_diag, arma::fill::zeros);    // coupling to the diagonal block
        VECTYPE D(n_diag);
        MATTYPE rhs_dense(n_dense, d), rhs_diag(n_diag, d);
        VECTYPE Ok(n_keep);
        for (unsigned i = 0; i < n_keep; ++i) {
            const unsigned b = keep[i];
            const float o = O(k, b);
            Ok(i) = o;
            if (diag_slot[b] >= 0) {
                const unsigned j = diag_slot[b];
                C(0, j) = o;
                D(j) = o + lamb_vec(i + 1);
                rhs_diag.row(j) = ZR[b].col(k).t();
            } else {
                const unsigned j = dense_slot[b];
                A(0, j) = o;
                A(j, 0) = o;
                A(j, j) = o + lamb_vec(i + 1);
                rhs_dense.row(j) = ZR[b].col(k).t();
            }
        }

        if (multiple_covariates) {
            // Count each cell once in the intercept, even if it belongs to
            // several kept batches, and add the overlap of kept batches.
            A(0, 0) = cov_sum(k);
            rhs_dense.row(0) = z_all.col(k).t();
            for (size_t p = 0; p < covariate_pairs.size(); ++p) {
                const CovariatePair& pair = covariate_pairs[p];
                for (size_t q = 0; q < pair.batch_a.size(); ++q) {
                    const unsigned a = pair.batch_a[q], b = pair.batch_b[q];
                    const bool a_dense = dense_slot[a] > 0, b_dense = dense_slot[b] > 0;
                    if (!(a_dense || diag_slot[a] >= 0) || !(b_dense || diag_slot[b] >= 0)) continue;
                    const float overlap = pair_sums[p](k, q);
                    if (a_dense && b_dense) {
                        A(dense_slot[a], dense_slot[b]) += overlap;
                        A(dense_slot[b], dense_slot[a]) += overlap;
                    } else if (a_dense) {
                        C(dense_slot[a], diag_slot[b]) += overlap;
                    } else if (b_dense) {
                        C(dense_slot[b], diag_slot[a]) += overlap;
                    }
                }
            }
        } else {
            A(0, 0) = accumulate(Ok.memptr(), n_keep);
            VECTYPE z_sum_all(d, arma::fill::zeros);
            for (unsigned b : keep) z_sum_all += ZR[b].col(k);
            rhs_dense.row(0) = z_sum_all.t();
        }
        A(0, 0) += lamb_vec(0);

        // Solve [A C; C' diag(D)] [W_dense; W_diag] = [rhs_dense; rhs_diag].
        const MATTYPE C_over_D = C.each_row() / D.t();
        const MATTYPE schur = A - C_over_D * C.t();
        const MATTYPE rhs_reduced = rhs_dense - C_over_D * rhs_diag;
        const MATTYPE W_dense = n_dense == 1 ? MATTYPE(rhs_reduced / schur(0, 0))
                                             : MATTYPE(arma::inv(schur) * rhs_reduced);
        MATTYPE W_diag = rhs_diag - C.t() * W_dense;
        W_diag.each_col() /= D;

        Y.col(k) = W_dense.row(0).t();

        for (unsigned b : keep) {
            MATTYPE& correction = M[b];
            if (correction.is_empty()) correction.zeros(d, K);
            correction.col(k) = diag_slot[b] >= 0 ? W_diag.row(diag_slot[b]).t()
                                                  : W_dense.row(dense_slot[b]).t();
            dense_slot[b] = -1;
            diag_slot[b] = -1;
        }
    }

    // Z_corr(:, cells of b) = Z_orig(:, cells of b) - M[b] * R(:, cells of b),
    // summed over the batches of each covariate.
    Z_corr = Z_orig;
    MATTYPE correction;
    for (int b = 0; b < B; ++b) {
        if (M[b].is_empty()) continue;
        gather_cells(R, batch_groups, b, R_batch);
        correction = M[b] * R_batch;
        const unsigned begin = batch_groups.offsets[b];
        const unsigned n = batch_groups.offsets[b + 1] - begin;
        const unsigned* cells = batch_groups.cells.data() + begin;
        run_tasks(n_chunks(n), size_t(n) * d, [&](size_t t) {
            const unsigned i0 = t * kCellsPerTask;
            const unsigned i1 = std::min(n, i0 + kCellsPerTask);
            for (unsigned i = i0; i < i1; ++i) {
                float* z = Z_corr.colptr(cells[i]);
                const float* delta = correction.colptr(i);
                for (int r = 0; r < d; ++r) z[r] -= delta[r];
            }
        });
    }

    Y = arma::normalise(Y, 2, 0);
}

} // namespace harmony
