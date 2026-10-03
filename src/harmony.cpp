// harmonypy - C++ backend matching R harmony2 package.
// Copyright (C) 2018  Ilya Korsunsky
//               2019  Kamil Slowikowski <kslowikowski@gmail.com>
//
// Uses custom scatter/gather kernels on a batch_id vector instead of
// sparse Phi matrices. Per-cell work, including the large matrix products,
// and the per-cluster ridge systems run on a std::thread pool
// (thread_pool.hpp). Armadillo is built without BLAS and LAPACK
// (ARMA_DONT_USE_BLAS, ARMA_DONT_USE_LAPACK): it only stores matrices and
// does element-wise work, so the module needs no BLAS or LAPACK library.
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

// One scratch buffer of n values per thread, each starting on its own
// 128-byte line, so threads never share a cache line and workers never
// allocate memory.
template <class T = float>
class ThreadScratch {
public:
    ThreadScratch(unsigned n_threads, size_t n) : stride_((n + 31) / 32 * 32), buffer_(stride_ * n_threads + 32) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(buffer_.data());
        offset_ = ((128 - address % 128) % 128) / sizeof(T);
    }
    T* get(unsigned thread) { return buffer_.data() + offset_ + stride_ * thread; }

private:
    size_t stride_;
    std::vector<T> buffer_;
    size_t offset_ = 0;
};

// Cells added per pass over a K x d sum of outer products.
constexpr unsigned kCellsPerPass = 4;

// sums(k, i) += sum over cells c of r[c * K + k] * z[c][i], for kCellsPerPass
// cells (a missing cell has r = 0). Adding several cells per pass divides the
// loads and stores of sums, which bound this loop, by kCellsPerPass.
inline void add_outer_products(double* __restrict__ sums, unsigned K, unsigned d,
                               const double* __restrict__ r, const float* const* z) {
    static_assert(kCellsPerPass == 4, "the loop below adds four cells");
    const double* __restrict__ r0 = r;
    const double* __restrict__ r1 = r + K;
    const double* __restrict__ r2 = r + 2 * static_cast<size_t>(K);
    const double* __restrict__ r3 = r + 3 * static_cast<size_t>(K);
    for (unsigned i = 0; i < d; ++i) {
        const double z0 = z[0][i], z1 = z[1][i], z2 = z[2][i], z3 = z[3][i];
        double* __restrict__ row = sums + static_cast<size_t>(i) * K;
        for (unsigned k = 0; k < K; ++k) row[k] += z0 * r0[k] + z1 * r1[k] + z2 * r2[k] + z3 * r3[k];
    }
}

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

// Dot product of n floats in eight partial sums combined in a fixed order,
// so the loop vectorizes and every platform sums in the same order.
inline float dot(const float* a, const float* b, unsigned n) {
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    unsigned i = 0;
    for (; i + 8 <= n; i += 8)
        for (unsigned l = 0; l < 8; ++l) acc[l] += a[i + l] * b[i + l];
    for (unsigned l = 0; i < n; ++i, ++l) acc[l] += a[i] * b[i];
    return ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
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

// Run fn(t, thread) for t in [0, n_tasks): on the pool, or inline when the
// work is small. thread (0 for the calling thread, below pool_threads())
// selects the running thread's scratch space.
template <class F>
void Harmony::run_tasks_on_threads(size_t n_tasks, size_t work, F&& fn) const {
    if (!pool || pool->size() == 1 || n_tasks <= 1 || work < kMinParallelWork) {
        for (size_t t = 0; t < n_tasks; ++t) fn(t, 0u);
        return;
    }
    pool->parallel_for(n_tasks, fn);
}

// Run fn(t) for t in [0, n_tasks), as run_tasks_on_threads.
template <class F>
void Harmony::run_tasks(size_t n_tasks, size_t work, F&& fn) const {
    run_tasks_on_threads(n_tasks, work, [&](size_t t, unsigned) { fn(t); });
}

// sums(:, r) = sum of X(:, cells[i]) over the positions i of run r, for runs
// [first, last), accumulated in the precision of sums (float or double).
template <class T>
void Harmony::sum_runs(const MATTYPE& X, const std::vector<unsigned>& cells,
                       const std::vector<CellRun>& runs, size_t first, size_t last, arma::Mat<T>& sums) {
    const unsigned n_rows = X.n_rows;
    if (sums.n_rows != n_rows || sums.n_cols < runs.size()) sums.set_size(n_rows, runs.size());
    size_t work = 0;
    for (size_t r = first; r < last; ++r) work += runs[r].end - runs[r].begin;
    ThreadScratch<T> scratch(pool_threads(), n_rows);
    run_tasks_on_threads(last - first, work * n_rows, [&](size_t t, unsigned thread) {
        const CellRun& run = runs[first + t];
        T* acc = scratch.get(thread);
        std::fill(acc, acc + n_rows, T(0));
        for (unsigned i = run.begin; i < run.end; ++i) {
            const float* x = X.colptr(cells[i]);
            for (unsigned k = 0; k < n_rows; ++k) acc[k] += x[k];
        }
        std::copy(acc, acc + n_rows, sums.colptr(first + t));
    });
}

// Scale each column to unit L2 norm.
void Harmony::normalise_columns(MATTYPE& X) {
    const unsigned n_rows = X.n_rows;
    const unsigned n_cols = X.n_cols;
    run_tasks(n_chunks(n_cols), size_t(n_rows) * n_cols, [&](size_t t) {
        const unsigned j0 = t * kCellsPerTask;
        const unsigned j1 = std::min(n_cols, j0 + kCellsPerTask);
        for (unsigned j = j0; j < j1; ++j) scale_to_unit_length(X.colptr(j), n_rows);
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
    VECTYPE draws_now(N, arma::fill::none), draws_next(N, arma::fill::none);
    VECTYPE prob(N, arma::fill::none);
    float* p = prob.memptr();
    const size_t n_tasks = n_chunks(N);
    std::vector<float> task_best(n_tasks);
    std::vector<unsigned> task_index(n_tasks);

    // Each centroid needs N uniform draws, made in order from rng. With
    // several threads, the next centroid's draws are made on another thread
    // while this one is chosen; the draws are the same either way. A local
    // generator keeps its state in registers.
    std::mt19937 generator = rng;
    auto draw = [&](VECTYPE& out) {
        float* u = out.memptr();
        for (int j = 0; j < N; ++j) u[j] = uniform01(generator);
    };
    BackgroundJob next_draws;
    draw(draws_now);
    for (int i = 0; i < K; ++i) {
        const bool more = i + 1 < K;
        const bool drawing_ahead = more && n_threads > 1 && next_draws.try_start([&] { draw(draws_next); });
        const float* y = Y.colptr(i);
        const unsigned n_dims = X.n_rows;
        const float* u = draws_now.memptr();

        // prob = -log(u) / distance with distance = |2 (1 - y'x)|, and its maximum.
        run_tasks(n_tasks, size_t(N) * (n_dims + 16), [&](size_t t) {
            const unsigned j0 = t * kCellsPerTask;
            const unsigned j1 = std::min<unsigned>(N, j0 + kCellsPerTask);
            float max_prob = -std::numeric_limits<float>::infinity();
            for (unsigned j = j0; j < j1; ++j) {
                const float distance = std::abs((1.0f - dot(y, X.colptr(j), n_dims)) * 2.0f);
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

        if (more) {
            if (drawing_ahead) next_draws.wait();
            else draw(draws_next);
            draws_now.swap(draws_next);
        }
    }
    rng = generator;

    // The constructor rejects non-finite input, so X is finite here.
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

    // Assign each cell to its nearest mean as arma::kmeans does: the smallest
    // squared_distance, the first mean on ties. The distances to
    // kMeansPerPass means at once are summed side by side (means_t(g, i)
    // holds dimension i of every mean), for two cells per pass, so each
    // loaded mean coordinate serves both cells; the arithmetic for every
    // (cell, mean) pair is unchanged.
    constexpr unsigned kMeansPerPass = 16;
    const MATTYPE means_t = means.t();
    run_tasks(n_chunks(n_cells), size_t(n_cells) * n_means * n_dims, [&](size_t t) {
        const unsigned j0 = t * kCellsPerTask;
        const unsigned j1 = std::min(n_cells, j0 + kCellsPerTask);
        float acc1[2][kMeansPerPass];
        float acc2[2][kMeansPerPass];
        for (unsigned j = j0; j < j1; j += 2) {
            const unsigned n_pair = std::min(2u, j1 - j);
            const float* x0 = X.colptr(j);
            const float* x1 = X.colptr(j + n_pair - 1);
            float min_dist[2] = {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()};
            unsigned best[2] = {0, 0};
            for (unsigned g0 = 0; g0 < n_means; g0 += kMeansPerPass) {
                const unsigned n_pass = std::min(kMeansPerPass, n_means - g0);
                for (unsigned g = 0; g < n_pass; ++g) {
                    acc1[0][g] = 0.0f;
                    acc2[0][g] = 0.0f;
                    acc1[1][g] = 0.0f;
                    acc2[1][g] = 0.0f;
                }
                unsigned a, b;
                for (a = 0, b = 1; b < n_dims; a += 2, b += 2) {
                    const float* mean_a = means_t.colptr(a) + g0;
                    const float* mean_b = means_t.colptr(b) + g0;
                    const float x0_a = x0[a], x0_b = x0[b];
                    const float x1_a = x1[a], x1_b = x1[b];
                    for (unsigned g = 0; g < n_pass; ++g) {
                        float tmp0_a = x0_a, tmp0_b = x0_b;
                        float tmp1_a = x1_a, tmp1_b = x1_b;
                        tmp0_a -= mean_a[g];
                        tmp0_b -= mean_b[g];
                        tmp1_a -= mean_a[g];
                        tmp1_b -= mean_b[g];
                        acc1[0][g] += tmp0_a * tmp0_a;
                        acc2[0][g] += tmp0_b * tmp0_b;
                        acc1[1][g] += tmp1_a * tmp1_a;
                        acc2[1][g] += tmp1_b * tmp1_b;
                    }
                }
                if (a < n_dims) {
                    const float* mean_a = means_t.colptr(a) + g0;
                    const float x0_a = x0[a], x1_a = x1[a];
                    for (unsigned g = 0; g < n_pass; ++g) {
                        const float tmp0_a = x0_a - mean_a[g];
                        const float tmp1_a = x1_a - mean_a[g];
                        acc1[0][g] += tmp0_a * tmp0_a;
                        acc1[1][g] += tmp1_a * tmp1_a;
                    }
                }
                for (unsigned c = 0; c < 2; ++c) {
                    for (unsigned g = 0; g < n_pass; ++g) {
                        const float dist = acc1[c][g] + acc2[c][g];
                        if (dist < min_dist[c]) {
                            min_dist[c] = dist;
                            best[c] = g0 + g;
                        }
                    }
                }
            }
            assignment[j] = best[0];
            if (n_pair == 2) assignment[j + 1] = best[1];
        }
    });

    group_by_key(members, assignment.data(), n_cells, n_means);
    MATTYPE sums(n_dims, n_means);
    ThreadScratch scratch(pool_threads(), n_dims);
    run_tasks_on_threads(n_means, size_t(n_cells) * n_dims, [&](size_t g, unsigned thread) {
        float* sum = scratch.get(thread);
        std::fill(sum, sum + n_dims, 0.0f);
        for (unsigned i = members.offsets[g]; i < members.offsets[g + 1]; ++i) {
            const float* x = X.colptr(members.cells[i]);
            for (unsigned k = 0; k < n_dims; ++k) sum[k] += x[k];
        }
        std::copy(sum, sum + n_dims, sums.colptr(g));
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
    N = Z.n_cols;
    d = Z.n_rows;
    B = 0;
    for (auto v : B_vec) B += v;

    // The kernels index these without bounds checks.
    if (K < 1) throw std::invalid_argument("nclust must be at least 1");
    if (!Z.is_finite()) throw std::invalid_argument("data_mat must not contain NaN or infinite values");
    if (sigma_in.n_elem != static_cast<arma::uword>(K))
        throw std::invalid_argument("sigma must have one value per cluster (nclust)");
    if (theta_in.n_elem != static_cast<arma::uword>(B) || Pr_b_in.n_elem != static_cast<arma::uword>(B))
        throw std::invalid_argument("theta must have one value per batch");
    if (lambda_in.n_elem == 0 || (lambda_in(0) >= 0 && lambda_in.n_elem != static_cast<arma::uword>(B) + 1))
        throw std::invalid_argument("lamb must have one value per batch");

    // ncores <= 0 means one thread per available core. The pool may start
    // fewer threads than requested if the system refuses more.
    const unsigned requested = ncores > 0 ? static_cast<unsigned>(ncores) : std::thread::hardware_concurrency();
    pool = std::make_unique<ThreadPool>(std::max(1u, requested));
    n_threads = pool->size();

    Z_orig = arma::conv_to<MATTYPE>::from(Z);
    Pr_b = arma::conv_to<VECTYPE>::from(Pr_b_in);

    Z_corr = Z_orig;
    normalise_columns(Z_corr);

    sigma = arma::conv_to<VECTYPE>::from(sigma_in);
    theta = arma::conv_to<VECTYPE>::from(theta_in);

    if (lambda_in(0) < 0) {
        lambda_estimation = true;
        lambda.zeros(B + 1);
        if (!(std::isfinite(alpha_in) && alpha_in >= 0))
            throw std::invalid_argument("alpha must be finite and not negative");
    } else {
        lambda_estimation = false;
        lambda = arma::conv_to<VECTYPE>::from(lambda_in);
        if (!lambda_in.is_finite() || lambda_in.min() < 0)
            throw std::invalid_argument("lamb must be finite and not negative");
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
    next_shuffle.wait();
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
    cell_batches.resize(static_cast<size_t>(N) * n_covariates);
    for (int j = 0; j < N; ++j) {
        for (int c = 0; c < n_covariates; ++c) {
            const int64_t first = c == 0 ? 0 : covariate_bounds[c - 1];
            const int64_t b = batch_of_cell(c, j);
            if (b < first || b >= static_cast<int64_t>(covariate_bounds[c]))
                throw std::invalid_argument("batch index out of range for its covariate");
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

    // Runs for the ridge sums, at most about 1024 in total plus one per
    // batch, so that the partial sums of batches with several runs stay small.
    const unsigned run_size = std::max<unsigned>(kCellsPerTask, (batch_groups.cells.size() + 1023) / 1024);
    ridge_runs.clear();
    ridge_partial.clear();
    n_ridge_partials = 0;
    for (int b = 0; b < B; ++b) {
        const unsigned begin = batch_groups.offsets[b], end = batch_groups.offsets[b + 1];
        const bool split = end - begin > run_size;
        for (unsigned start = begin; start < end; start += run_size) {
            ridge_runs.push_back({static_cast<unsigned>(b), start, std::min(end, start + run_size)});
            ridge_partial.push_back(split ? static_cast<int>(n_ridge_partials++) : -1);
        }
    }

    batch_covariate.resize(B);
    for (int c = 0; c < n_covariates; ++c)
        for (unsigned b = c == 0 ? 0 : covariate_bounds[c - 1]; b < covariate_bounds[c]; ++b)
            batch_covariate[b] = c;

    // Count cells in each covariate level.
    batch_sizes.set_size(B);
    for (int b = 0; b < B; ++b)
        batch_sizes(b) = static_cast<float>(batch_groups.offsets[b + 1] - batch_groups.offsets[b]);

    // Cells sorted by their combination of batches (stable).
    cells_by_combination.resize(N);
    std::iota(cells_by_combination.begin(), cells_by_combination.end(), 0);
    std::stable_sort(cells_by_combination.begin(), cells_by_combination.end(), [&](unsigned a, unsigned b) {
        const unsigned* batches_a = &cell_batches[static_cast<size_t>(a) * n_covariates];
        const unsigned* batches_b = &cell_batches[static_cast<size_t>(b) * n_covariates];
        return std::lexicographical_compare(batches_a, batches_a + n_covariates, batches_b, batches_b + n_covariates);
    });

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
    // init_cluster overwrites every element of dist_mat and R before use.
    dist_mat.set_size(K, N);
    R.set_size(K, N);
    O.zeros(K, B);
    E.zeros(K, B);
    Y.zeros(d, K);
    log_div.zeros(K, B);
}

// =========================================================================
// init_cluster
// =========================================================================

// Set dist_mat to the distances 2 * (1 - cosine) between each cell and each
// centroid, and R = softmax(-dist / sigma) for every cell, without the
// diversity penalty. The similarities Y' Z_corr(:, j) are summed over
// dimensions in order, for all clusters side by side, so the loop vectorizes.
void Harmony::assign_without_diversity(const char* stage) {
    const size_t n_tasks = n_chunks(N);
    std::vector<unsigned char> flags(n_tasks, 0);
    const float* s = sigma.memptr();
    const MATTYPE Y_t = Y.t();
    run_tasks(n_tasks, size_t(N) * K * (d + 8), [&](size_t t) {
        const unsigned j0 = t * kCellsPerTask;
        const unsigned j1 = std::min<unsigned>(N, j0 + kCellsPerTask);
        unsigned char found = 0;
        for (unsigned j = j0; j < j1; ++j) {
            const float* x = Z_corr.colptr(j);
            float* __restrict__ dist = dist_mat.colptr(j);
            std::fill(dist, dist + K, 0.0f);
            for (int a = 0; a < d; ++a) {
                const float x_a = x[a];
                const float* __restrict__ y_a = Y_t.colptr(a);
                for (int k = 0; k < K; ++k) dist[k] += x_a * y_a[k];
            }
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
    ThreadScratch scratch(pool_threads(), K);
    run_tasks_on_threads(n_tasks, size_t(N) * K, [&](size_t t, unsigned thread) {
        const unsigned j0 = t * kCellsPerTask;
        const unsigned j1 = std::min<unsigned>(N, j0 + kCellsPerTask);
        float* acc = scratch.get(thread);
        std::fill(acc, acc + K, 0.0f);
        for (unsigned j = j0; j < j1; ++j) {
            const float* r = R.colptr(j);
            for (int k = 0; k < K; ++k) acc[k] += r[k];
        }
        std::copy(acc, acc + K, task_sums.colptr(t));
    });
    VECTYPE totals(K, arma::fill::zeros);
    for (size_t t = 0; t < n_tasks; ++t) totals += task_sums.col(t);
    for (int b = 0; b < B; ++b)
        for (int k = 0; k < K; ++k) E(k, b) = totals(k) * Pr_b(b);

    MATTYPE sums;
    sum_runs(R, batch_groups.cells, batch_groups.runs, 0, batch_groups.runs.size(), sums);
    O.zeros();
    for (size_t r = 0; r < batch_groups.runs.size(); ++r)
        O.col(batch_groups.runs[r].group) += sums.col(r);
}

void Harmony::init_cluster() {
    Y = kmeans_init(Z_corr);
    normalise_columns(Y);

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
    ThreadScratch scratch(pool_threads(), K);
    run_tasks_on_threads(n_lead_runs, n_cells * K * 8, [&](size_t r, unsigned thread) {
        const CellRun& run = block_runs[r];
        const float* lead_div = log_div.colptr(run.group);
        float* sums = scratch.get(thread);
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
        std::copy(sums, sums + K, run_sums.colptr(r));
        run_error[r] = error;
        run_entropy[r] = entropy;
        run_flags[r] = found;
    });
}

void Harmony::update_R() {
    // With several threads, the next pass's order is shuffled on another
    // thread during this pass. rng is used only here once clustering starts,
    // so the shuffles come out in the same order either way.
    std::vector<unsigned> indices_vec;
    if (next_order_pending) {
        next_shuffle.wait();
        indices_vec.swap(next_order);
        next_order_pending = false;
    } else {
        indices_vec.resize(N);
        std::iota(indices_vec.begin(), indices_vec.end(), 0);
        std::shuffle(indices_vec.begin(), indices_vec.end(), rng);
    }
    if (n_threads > 1) {
        next_order.resize(N);
        next_order_pending = next_shuffle.try_start([this] {
            std::iota(next_order.begin(), next_order.end(), 0);
            std::shuffle(next_order.begin(), next_order.end(), rng);
        });
    }

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
 * sum of R(k, :) over those cells and z_all(k, :) the matching weighted
 * coordinate sum, accumulated over fixed chunks of cells and added in order.
 */
void Harmony::multi_covariate_totals(const arma::Mat<unsigned char>& kept, arma::mat& z_all, arma::vec& cov_sum) {
    const unsigned n_cov = n_covariates;
    const size_t sum_size = static_cast<size_t>(K) * d;
    // At most about 1024 tasks, so the per-task sums stay small.
    const unsigned task_cells = std::max<unsigned>(kCellsPerTask, (static_cast<unsigned>(N) + 1023) / 1024);
    const size_t n_tasks = (static_cast<size_t>(N) + task_cells - 1) / task_cells;
    arma::mat partials(sum_size + K, n_tasks);
    ThreadScratch<double> scratch(pool_threads(), sum_size + (1 + kCellsPerPass) * static_cast<size_t>(K));
    const unsigned char* kept_mem = kept.memptr();
    run_tasks_on_threads(n_tasks, size_t(N) * K * (d + n_cov), [&](size_t t, unsigned thread) {
        const unsigned j0 = t * task_cells;
        const unsigned j1 = std::min<unsigned>(N, j0 + task_cells);
        // acc(k, i) holds the K x d coordinate sums, then K totals, then the
        // masked assignments of kCellsPerPass cells.
        double* acc = scratch.get(thread);
        double* totals = acc + sum_size;
        double* masked = totals + K;
        std::fill(acc, acc + sum_size + K, 0.0);
        const float* z[kCellsPerPass];
        for (unsigned j = j0; j < j1; j += kCellsPerPass) {
            for (unsigned c = 0; c < kCellsPerPass; ++c) {
                double* m = masked + static_cast<size_t>(c) * K;
                z[c] = Z_orig.colptr(j);
                if (j + c >= j1) {
                    std::fill(m, m + K, 0.0);
                    continue;
                }
                const unsigned* batches = &cell_batches[static_cast<size_t>(j + c) * n_cov];
                const float* r = R.colptr(j + c);
                z[c] = Z_orig.colptr(j + c);
                for (int k = 0; k < K; ++k) {
                    unsigned char any = 0;
                    for (unsigned b = 0; b < n_cov; ++b) any |= kept_mem[static_cast<size_t>(batches[b]) * K + k];
                    m[k] = any ? r[k] : 0.0f;
                    totals[k] += m[k];
                }
            }
            add_outer_products(acc, K, d, masked, z);
        }
        std::copy(acc, acc + sum_size + K, partials.colptr(t));
    });
    z_all.zeros(K, d);
    cov_sum.zeros(K);
    for (size_t t = 0; t < n_tasks; ++t) {
        const double* part = partials.colptr(t);
        double* sums = z_all.memptr();
        for (size_t i = 0; i < sum_size; ++i) sums[i] += part[i];
        for (int k = 0; k < K; ++k) cov_sum[k] += part[sum_size + k];
    }
}

// RZ[b](k, i) = sum of R(k, j) * Z_orig(i, j) over the cells j of batch b,
// and RZ[b](k, d) = sum of R(k, j), for each batch in use: every cluster's
// R-weighted coordinate sums and totals. Each run of a batch's cells adds
// rank-one updates to a K x (d + 1) sum; a batch split over several runs adds
// its runs' sums in run order. The sums are accumulated in double (see
// moe_correct_ridge).
void Harmony::batch_coordinate_sums(std::vector<arma::mat>& RZ, const std::vector<char>& used) {
    const size_t sum_size = static_cast<size_t>(K) * (d + 1);
    arma::mat partials(sum_size, n_ridge_partials);
    for (int b = 0; b < B; ++b)
        if (used[b]) RZ[b].set_size(K, d + 1);
    ThreadScratch<double> scratch(pool_threads(), sum_size + kCellsPerPass * static_cast<size_t>(K));
    run_tasks_on_threads(ridge_runs.size(), size_t(N) * n_covariates * sum_size, [&](size_t r, unsigned thread) {
        const CellRun& run = ridge_runs[r];
        if (!used[run.group]) return;
        // acc(k, i) holds the K x d coordinate sums, then K totals, then the
        // assignments of kCellsPerPass cells.
        double* acc = scratch.get(thread);
        double* totals = acc + static_cast<size_t>(K) * d;
        double* r_cells = acc + sum_size;
        std::fill(acc, acc + sum_size, 0.0);
        const float* z[kCellsPerPass];
        for (unsigned t = run.begin; t < run.end; t += kCellsPerPass) {
            for (unsigned c = 0; c < kCellsPerPass; ++c) {
                double* r_c = r_cells + static_cast<size_t>(c) * K;
                if (t + c >= run.end) {
                    z[c] = z[0];
                    std::fill(r_c, r_c + K, 0.0);
                    continue;
                }
                const unsigned j = batch_groups.cells[t + c];
                const float* r_j = R.colptr(j);
                z[c] = Z_orig.colptr(j);
                for (int k = 0; k < K; ++k) {
                    r_c[k] = r_j[k];
                    totals[k] += r_c[k];
                }
            }
            add_outer_products(acc, K, d, r_cells, z);
        }
        double* out = ridge_partial[r] < 0 ? RZ[run.group].memptr() : partials.colptr(ridge_partial[r]);
        std::copy(acc, acc + sum_size, out);
    });
    std::vector<char> started(B, 0);
    for (size_t r = 0; r < ridge_runs.size(); ++r) {
        const unsigned b = ridge_runs[r].group;
        if (ridge_partial[r] < 0 || !used[b]) continue;
        const double* part = partials.colptr(ridge_partial[r]);
        double* sum = RZ[b].memptr();
        if (!started[b]) std::copy(part, part + sum_size, sum);
        else for (size_t i = 0; i < sum_size; ++i) sum[i] += part[i];
        started[b] = 1;
    }
}

// Z_corr(:, j) = Z_orig(:, j) - sum over the cell's batches b of M[b] * R(:, j).
// M[b] is empty for batches no cluster corrects. Cells that share all their
// batches are visited together, so their combined M is built once. Columns
// are padded to a multiple of 16 floats so the inner loop has no remainder.
void Harmony::apply_corrections(const std::vector<MATTYPE>& M) {
    const unsigned n_cov = n_covariates;
    const unsigned d_pad = (static_cast<unsigned>(d) + 15) & ~15u;
    const size_t combined_size = static_cast<size_t>(d_pad) * K;
    ThreadScratch scratch(pool_threads(), combined_size + d_pad);
    run_tasks_on_threads(n_chunks(N), size_t(N) * K * d, [&](size_t t, unsigned thread) {
        const unsigned i0 = t * kCellsPerTask;
        const unsigned i1 = std::min<unsigned>(N, i0 + kCellsPerTask);
        float* combined = scratch.get(thread);
        float* total = combined + combined_size;
        const unsigned* combination = nullptr;
        bool corrected = false;
        for (unsigned i = i0; i < i1; ++i) {
            const unsigned j = cells_by_combination[i];
            const unsigned* batches = &cell_batches[static_cast<size_t>(j) * n_cov];
            if (!combination || !std::equal(batches, batches + n_cov, combination)) {
                combination = batches;
                corrected = false;
                std::fill(combined, combined + combined_size, 0.0f);
                for (unsigned c = 0; c < n_cov; ++c) {
                    const MATTYPE& m = M[batches[c]];
                    if (m.is_empty()) continue;
                    corrected = true;
                    for (int k = 0; k < K; ++k) {
                        const float* __restrict__ from = m.colptr(k);
                        float* __restrict__ to = &combined[static_cast<size_t>(k) * d_pad];
                        for (int r = 0; r < d; ++r) to[r] += from[r];
                    }
                }
            }
            const float* z = Z_orig.colptr(j);
            float* out = Z_corr.colptr(j);
            if (!corrected) {
                std::copy(z, z + d, out);
                continue;
            }
            float* __restrict__ sum = total;
            std::fill(sum, sum + d_pad, 0.0f);
            const float* r = R.colptr(j);
            for (int k = 0; k < K; ++k) {
                const float r_k = r[k];
                const float* __restrict__ direction = &combined[static_cast<size_t>(k) * d_pad];
                for (unsigned a = 0; a < d_pad; ++a) sum[a] += direction[a] * r_k;
            }
            for (int a = 0; a < d; ++a) out[a] = z[a] - sum[a];
        }
    });
}

// One thread's scratch space for solve_ridge_cluster, sized for the largest
// system and allocated before the clusters are solved in parallel. Each
// allocation is padded at both ends, so threads never share a cache line.
struct RidgeWorkspace {
    static constexpr size_t kPad = 32;

    RidgeWorkspace(unsigned max_dense, unsigned max_diag, unsigned d, unsigned n_batches)
        : values_(2 * kPad + size_t(max_dense) * (max_dense + max_diag + d) + size_t(max_diag) * (d + 1) + d),
          slots_(2 * kPad + 2 * size_t(n_batches), -1) {
        double* next = values_.data() + kPad;
        auto take = [&next](size_t n) {
            double* p = next;
            next += n;
            return p;
        };
        S = take(size_t(max_dense) * max_dense);
        C = take(size_t(max_dense) * max_diag);
        rhs_dense = take(size_t(d) * max_dense);
        D = take(max_diag);
        rhs_diag = take(size_t(d) * max_diag);
        column = take(d);
        dense_slot = slots_.data() + kPad;
        diag_slot = dense_slot + n_batches;
    }
    RidgeWorkspace(const RidgeWorkspace&) = delete;
    RidgeWorkspace& operator=(const RidgeWorkspace&) = delete;
    RidgeWorkspace(RidgeWorkspace&&) = default;

    double* S;          // dense block, then its Cholesky factor (max_dense x max_dense)
    double* C;          // coupling of the dense and diagonal blocks (max_dense x max_diag)
    double* rhs_dense;  // right-hand side of the dense rows, then W_dense (d x max_dense)
    double* D;          // diagonal block (max_diag)
    double* rhs_diag;   // right-hand side of the diagonal rows (d x max_diag)
    double* column;     // one coefficient column (d)
    int* dense_slot;    // row of each batch in the dense block (>= 1), or -1
    int* diag_slot;     // row of each batch in the diagonal block (>= 0), or -1

private:
    std::vector<double> values_;
    std::vector<int> slots_;
};

// Solve cluster k's ridge system and store its coefficients: the centroid
// Y(:, k) and each kept batch's correction direction M[b](:, k).
//
// The system has one row for the intercept and one per kept batch:
//
//   [ total  O_k'               ] + diag(penalties)
//   [ O_k    diag(O_k) + overlap ]
//
// where overlap holds the cells shared by kept batches of different
// covariates. The batches of diag_covariate form a diagonal block D with
// coupling C to the dense block A (the intercept and the other covariates'
// batches). Eliminating D leaves the Schur complement S = A - C D^-1 C', a
// single number with one covariate (the arrowhead inverse R harmony uses).
// S is factored by Cholesky in double precision.
//
// With lamb (or alpha) zero the system is singular when the kept batches of
// a covariate cover every cell, because the intercept row is then the sum of
// their rows. The sums are accumulated in double, so a pivot of such a
// system is zero up to rounding, about 1e-15 times the cluster's total
// weight (the largest diagonal entry). A small positive lamb leaves a pivot
// of about lamb times the number of kept batches, which the double-precision
// solve handles accurately. Pivots below 1e-10 times the largest diagonal
// entry are reported as singular.
void Harmony::solve_ridge_cluster(unsigned k, unsigned diag_covariate, const std::vector<unsigned>& keep,
                                  const std::vector<arma::mat>& RZ, const arma::mat& z_all,
                                  const arma::vec& cov_sum, const std::vector<arma::mat>& pair_sums,
                                  RidgeWorkspace& ws, std::vector<MATTYPE>& M) {
    const unsigned n_keep = keep.size();
    // Penalties: alpha * E(k, b) when lambda is estimated, otherwise
    // lambda(b + 1). The intercept is penalized only by a fixed lambda(0)
    // when every batch is kept.
    auto penalty = [&](unsigned b) {
        return static_cast<double>(lambda_estimation ? alpha * E.at(k, b) : lambda.at(b + 1));
    };
    const double intercept_penalty =
        (!lambda_estimation && n_keep == static_cast<unsigned>(B)) ? static_cast<double>(lambda.at(0)) : 0.0;

    unsigned n = 1, m = 0;
    for (unsigned b : keep) {
        if (batch_covariate[b] == diag_covariate) ws.diag_slot[b] = m++;
        else ws.dense_slot[b] = n++;
    }
    // S (n x n), C (n x m) and D (m) are column-major. Column x of
    // rhs_dense (d x n) and column j of rhs_diag (d x m) hold the right-hand
    // side of dense row x and diagonal row j.
    double* S = ws.S;
    double* C = ws.C;
    double* D = ws.D;
    double* rhs_dense = ws.rhs_dense;
    double* rhs_diag = ws.rhs_diag;
    std::fill(S, S + static_cast<size_t>(n) * n, 0.0);
    std::fill(C, C + static_cast<size_t>(n) * m, 0.0);
    std::fill(rhs_dense, rhs_dense + d, 0.0);

    double total = 0.0;
    for (unsigned b : keep) {
        const double o = RZ[b].at(k, d);
        total += o;
        double* rhs;
        if (ws.diag_slot[b] >= 0) {
            const unsigned j = ws.diag_slot[b];
            C[static_cast<size_t>(j) * n] = o;
            D[j] = o + penalty(b);
            rhs = rhs_diag + static_cast<size_t>(j) * d;
        } else {
            const unsigned x = ws.dense_slot[b];
            S[x] = o;
            S[static_cast<size_t>(x) * n] = o;
            S[static_cast<size_t>(x) * n + x] = o + penalty(b);
            rhs = rhs_dense + static_cast<size_t>(x) * d;
        }
        for (int i = 0; i < d; ++i) rhs[i] = RZ[b].at(k, i);
    }

    if (n_covariates > 1) {
        // Count each cell once in the intercept, even if it belongs to
        // several kept batches, and add the overlap of kept batches.
        S[0] = cov_sum.at(k);
        for (int i = 0; i < d; ++i) rhs_dense[i] = z_all.at(k, i);
        for (size_t p = 0; p < covariate_pairs.size(); ++p) {
            const CovariatePair& pair = covariate_pairs[p];
            for (size_t q = 0; q < pair.batch_a.size(); ++q) {
                const unsigned a = pair.batch_a[q], b = pair.batch_b[q];
                const bool a_dense = ws.dense_slot[a] > 0, b_dense = ws.dense_slot[b] > 0;
                if (!(a_dense || ws.diag_slot[a] >= 0) || !(b_dense || ws.diag_slot[b] >= 0)) continue;
                const double overlap = pair_sums[p].at(k, q);
                if (a_dense && b_dense) {
                    S[static_cast<size_t>(ws.dense_slot[a]) * n + ws.dense_slot[b]] += overlap;
                    S[static_cast<size_t>(ws.dense_slot[b]) * n + ws.dense_slot[a]] += overlap;
                } else if (a_dense) {
                    C[static_cast<size_t>(ws.diag_slot[b]) * n + ws.dense_slot[a]] += overlap;
                } else if (b_dense) {
                    C[static_cast<size_t>(ws.diag_slot[a]) * n + ws.dense_slot[b]] += overlap;
                }
            }
        }
    } else {
        S[0] = total;
        for (unsigned b : keep)
            for (int i = 0; i < d; ++i) rhs_dense[i] += RZ[b].at(k, i);
    }
    S[0] += intercept_penalty;
    double largest_diagonal = 0.0;
    for (unsigned x = 0; x < n; ++x) largest_diagonal = std::max(largest_diagonal, S[static_cast<size_t>(x) * n + x]);

    // Eliminate the diagonal block: S -= C D^-1 C' (lower triangle) and
    // rhs_dense -= C D^-1 rhs_diag. Most couplings are zero when one
    // covariate is nested in another, so those are skipped.
    for (unsigned j = 0; j < m; ++j) {
        const double* c = C + static_cast<size_t>(j) * n;
        const double* r_j = rhs_diag + static_cast<size_t>(j) * d;
        for (unsigned x = 0; x < n; ++x) {
            if (c[x] == 0.0) continue;
            const double scale = c[x] / D[j];
            double* s_x = S + static_cast<size_t>(x) * n;
            for (unsigned y = x; y < n; ++y) s_x[y] -= scale * c[y];
            double* r_x = rhs_dense + static_cast<size_t>(x) * d;
            for (int i = 0; i < d; ++i) r_x[i] -= scale * r_j[i];
        }
    }

    // Cholesky factorization S = L L' in place (lower triangle).
    const double min_pivot = 1e-10 * largest_diagonal;
    for (unsigned c = 0; c < n; ++c) {
        double* col = S + static_cast<size_t>(c) * n;
        if (!(col[c] > min_pivot))
            numerical_error("ridge correction", "ridge system is singular; lamb (or alpha) is zero or too small");
        col[c] = std::sqrt(col[c]);
        for (unsigned r = c + 1; r < n; ++r) col[r] /= col[c];
        for (unsigned c2 = c + 1; c2 < n; ++c2) {
            const double f = col[c2];
            if (f == 0.0) continue;
            double* col2 = S + static_cast<size_t>(c2) * n;
            for (unsigned r = c2; r < n; ++r) col2[r] -= f * col[r];
        }
    }

    // Solve L L' W = rhs_dense for all d columns at once; column x of
    // rhs_dense becomes row x of W_dense.
    for (unsigned c = 0; c < n; ++c) {
        const double* col = S + static_cast<size_t>(c) * n;
        double* b_c = rhs_dense + static_cast<size_t>(c) * d;
        for (int i = 0; i < d; ++i) b_c[i] /= col[c];
        for (unsigned r = c + 1; r < n; ++r) {
            if (col[r] == 0.0) continue;
            double* b_r = rhs_dense + static_cast<size_t>(r) * d;
            for (int i = 0; i < d; ++i) b_r[i] -= col[r] * b_c[i];
        }
    }
    for (unsigned c = n; c-- > 0;) {
        const double* col = S + static_cast<size_t>(c) * n;
        double* b_c = rhs_dense + static_cast<size_t>(c) * d;
        for (unsigned r = c + 1; r < n; ++r) {
            if (col[r] == 0.0) continue;
            const double* w_r = rhs_dense + static_cast<size_t>(r) * d;
            for (int i = 0; i < d; ++i) b_c[i] -= col[r] * w_r[i];
        }
        for (int i = 0; i < d; ++i) b_c[i] /= col[c];
    }

    // Store the coefficients: W_dense row 0 is the centroid; the diagonal
    // block's rows follow from W_diag = D^-1 (rhs_diag - C' W_dense).
    float* centroid = Y.colptr(k);
    for (int i = 0; i < d; ++i) centroid[i] = static_cast<float>(rhs_dense[i]);
    double* w = ws.column;
    for (unsigned b : keep) {
        float* out = M[b].colptr(k);
        if (ws.diag_slot[b] >= 0) {
            const unsigned j = ws.diag_slot[b];
            const double* c = C + static_cast<size_t>(j) * n;
            const double* r_j = rhs_diag + static_cast<size_t>(j) * d;
            for (int i = 0; i < d; ++i) w[i] = r_j[i];
            for (unsigned x = 0; x < n; ++x) {
                if (c[x] == 0.0) continue;
                const double* w_x = rhs_dense + static_cast<size_t>(x) * d;
                for (int i = 0; i < d; ++i) w[i] -= c[x] * w_x[i];
            }
            for (int i = 0; i < d; ++i) out[i] = static_cast<float>(w[i] / D[j]);
        } else {
            const double* w_x = rhs_dense + static_cast<size_t>(ws.dense_slot[b]) * d;
            for (int i = 0; i < d; ++i) out[i] = static_cast<float>(w_x[i]);
        }
        ws.dense_slot[b] = -1;
        ws.diag_slot[b] = -1;
    }
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

    // Every cluster's R-weighted coordinate sums for each batch, in one pass
    // over the cells instead of one pass per cluster. These sums, and the
    // totals below, are accumulated in double: with a small lamb the ridge
    // system is nearly singular (the intercept is the sum of each covariate's
    // batches), and float sums that disagree in their last digits would
    // decide its solution.
    std::vector<char> used(B, 0);
    for (int k = 0; k < K; ++k)
        if (active[k])
            for (unsigned b : keeps[k]) used[b] = 1;
    std::vector<arma::mat> RZ(B);
    batch_coordinate_sums(RZ, used);

    // With several covariates, a cell belongs to one batch per covariate, so
    // the intercept terms and the overlap of kept batches need extra totals.
    arma::mat z_all;
    arma::vec cov_sum;
    std::vector<arma::mat> pair_sums(covariate_pairs.size());
    if (multiple_covariates) {
        arma::Mat<unsigned char> kept(K, B, arma::fill::zeros);
        for (int k = 0; k < K; ++k)
            if (active[k])
                for (unsigned b : keeps[k]) kept(k, b) = 1;
        multi_covariate_totals(kept, z_all, cov_sum);

        arma::mat sums;
        for (size_t p = 0; p < covariate_pairs.size(); ++p) {
            const CellGroups& groups = covariate_pairs[p].groups;
            sum_runs(R, groups.cells, groups.runs, 0, groups.runs.size(), sums);
            pair_sums[p].zeros(K, covariate_pairs[p].batch_a.size());
            for (size_t r = 0; r < groups.runs.size(); ++r)
                pair_sums[p].col(groups.runs[r].group) += sums.col(r);
        }
    }

    // The ridge system of each cluster has one row for the intercept and one
    // per kept batch. The batches of the covariate with the most kept batches
    // share no cells, so they form a diagonal block that solve_ridge_cluster
    // eliminates first; the rest (the intercept and the other covariates'
    // batches) is a small dense block. Find the largest blocks so that each
    // thread's workspace can be set aside before the clusters are solved in
    // parallel.
    std::vector<unsigned> diag_covariate(K, 0);
    unsigned max_dense = 1, max_diag = 0;
    size_t work = 0;
    for (int k = 0; k < K; ++k) {
        if (!active[k]) continue;
        std::vector<unsigned> kept_per_covariate(n_covariates, 0);
        for (unsigned b : keeps[k]) kept_per_covariate[batch_covariate[b]]++;
        const unsigned c = std::max_element(kept_per_covariate.begin(), kept_per_covariate.end())
                           - kept_per_covariate.begin();
        const size_t n_diag = kept_per_covariate[c];
        const size_t n_dense = 1 + keeps[k].size() - n_diag;
        diag_covariate[k] = c;
        max_dense = std::max<unsigned>(max_dense, n_dense);
        max_diag = std::max<unsigned>(max_diag, n_diag);
        work += n_dense * (n_dense + n_diag + d) * (n_dense + d) + n_diag * d;
    }

    // M[b](:, k) is the correction direction of batch b in cluster k (zero
    // when cluster k does not correct batch b).
    std::vector<MATTYPE> M(B);
    for (int b = 0; b < B; ++b)
        if (used[b]) M[b].zeros(d, K);

    std::vector<RidgeWorkspace> workspaces;
    workspaces.reserve(pool_threads());
    for (unsigned t = 0; t < pool_threads(); ++t) workspaces.emplace_back(max_dense, max_diag, d, B);
    run_tasks_on_threads(K, work, [&](size_t k, unsigned thread) {
        if (active[k])
            solve_ridge_cluster(k, diag_covariate[k], keeps[k], RZ, z_all, cov_sum, pair_sums,
                                workspaces[thread], M);
    });

    apply_corrections(M);

    normalise_columns(Y);
}

} // namespace harmony
