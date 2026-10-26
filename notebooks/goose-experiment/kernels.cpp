// The two hot loops of harmonypy's clustering step, as in src/harmony.cpp,
// behind a C API so that the same arrays can be run through this and the
// Goose version (kernels.goose). Single-threaded: harmonypy splits the cells
// over threads outside these loops.
//
// Column-major like Armadillo: Z is d x N (cell j at Z + j * d), Y_t is K x d
// (centroid dimension a at Y_t + a * K), dist and R are K x N.
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

// Sum in two partial sums, as harmony.cpp's accumulate.
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

}  // namespace

extern "C" {

// Harmony::assign_without_diversity: distances 2 (1 - Y' z_j) for every cell
// and cluster, then R(:, j) = softmax(-dist / sigma). Returns the number of
// cells whose normalizer was not finite and positive.
int64_t cpp_assign(const float* Z, const float* Y_t, const float* sigma, float* dist_mat, float* R,
                   int64_t N, int64_t d, int64_t K) {
    int64_t bad = 0;
    for (int64_t j = 0; j < N; ++j) {
        const float* x = Z + j * d;
        float* __restrict__ dist = dist_mat + j * K;
        for (int64_t k = 0; k < K; ++k) dist[k] = 0.0f;
        for (int64_t a = 0; a < d; ++a) {
            const float x_a = x[a];
            const float* __restrict__ y_a = Y_t + a * K;
            for (int64_t k = 0; k < K; ++k) dist[k] += x_a * y_a[k];
        }
        float* logits = R + j * K;
        for (int64_t k = 0; k < K; ++k) {
            const float distance = 2.0f * (1.0f - dist[k]);
            dist[k] = distance;
            logits[k] = (-distance) / sigma[k];
        }
        if (!softmax_column(logits, static_cast<unsigned>(K))) ++bad;
    }
    return bad;
}

// The inner loop of Harmony::reassign_block_runs for the cells of one run,
// all in batch `group`, with one batch variable: logits from the distances
// plus the batch's log diversity penalty, a softmax, the sum of the new
// assignments over the run, and the run's objective terms. Returns the flags
// (1: a bad normalizer, 2: a column that does not sum to one).
int64_t cpp_reassign(const float* dist_mat, const float* sigma, const float* lead_div, const uint32_t* cells,
                     int64_t n_cells, int64_t K, float* R, float* sums, double* error_out,
                     double* entropy_out) {
    for (int64_t k = 0; k < K; ++k) sums[k] = 0.0f;
    double error = 0.0, entropy = 0.0;
    int64_t found = 0;
    for (int64_t i = 0; i < n_cells; ++i) {
        const uint32_t j = cells[i];
        const float* dist = dist_mat + static_cast<int64_t>(j) * K;
        float* logits = R + static_cast<int64_t>(j) * K;
        for (int64_t k = 0; k < K; ++k) {
            const float scaled = (-dist[k]) / sigma[k];
            logits[k] = scaled + lead_div[k];
        }
        float max_logit = -std::numeric_limits<float>::infinity();
        for (int64_t k = 0; k < K; ++k)
            if (logits[k] > max_logit) max_logit = logits[k];
        double weighted = 0.0, weighted_shifted = 0.0;
        for (int64_t k = 0; k < K; ++k) {
            const float shifted = logits[k] - max_logit;
            const float e = std::exp(shifted);
            logits[k] = e;
            const double w = double(sigma[k]) * e;
            weighted += w;
            weighted_shifted += w * shifted;
        }
        const float total = accumulate(logits, static_cast<unsigned>(K));
        if (!(std::isfinite(total) && total > 0.0f)) {
            found |= 1;
            continue;
        }
        double cell_error = 0.0;
        for (int64_t k = 0; k < K; ++k) {
            const float p = logits[k] / total;
            logits[k] = p;
            sums[k] += p;
            cell_error += double(p) * dist[k];
        }
        const float column_sum = accumulate(logits, static_cast<unsigned>(K));
        if (!std::isfinite(column_sum) || std::abs(column_sum - 1.0f) > 1e-4f) found |= 2;
        error += cell_error;
        entropy += (weighted_shifted - std::log(double(total)) * weighted) / total;
    }
    *error_out = error;
    *entropy_out = entropy;
    return found;
}

}  // extern "C"
