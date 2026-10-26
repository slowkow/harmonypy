// The two hot loops of harmonypy's clustering step (kernels.cpp), in Goose.
// Compiled to C with `goose -o kernels.c --header kernels.h`, then built
// into a shared library next to the C++ one, so bench.py can run both on the
// same arrays. Column-major like Armadillo: Z is d x N, Y_t is K x d, dist
// and R are K x N.
//
// The loops run over a slice (`for v, k in dist`) rather than over `K`, and
// the slices' lengths are asserted equal, so that Goose can drop the bounds
// checks of the inner loops (goose --bce-lines reports which remain).
import std;
import math;

// Sum in two partial sums, as harmony.cpp's accumulate.
fn accumulate(x: const f32[:]) -> f32 {
    var acc1: f32 = 0.0;
    var acc2: f32 = 0.0;
    let n = x.len;
    var j: i64 = 1;
    while j < n {
        acc1 += x[j - 1];
        acc2 += x[j];
        j += 2;
    }
    if j - 1 < n { acc1 += x[j - 1]; }
    return acc1 + acc2;
}

fn finite_positive(total: f32) -> bool {
    let t = total as f64;
    return !is_nan(t) && !is_inf(t) && t > 0.0;
}

// Softmax of one column of logits in place; false if the sum is not finite
// and positive.
fn softmax_column(x: f32[:]) -> bool {
    var max_logit = x[0];
    for v in x { if v > max_logit { max_logit = v; } }
    for v, i in x { x[i] = exp(v - max_logit); }
    let total = accumulate(x);
    if !finite_positive(total) { return false; }
    for v, i in x { x[i] = v / total; }
    return true;
}

// Distances 2 (1 - Y' z_j) for every cell and cluster, then R(:, j) =
// softmax(-dist / sigma). Returns the number of cells whose normalizer was
// not finite and positive.
export fn goose_assign(Z: const f32[:], Y_t: const f32[:], sigma: const f32[:], dist_mat: f32[:], R: f32[:],
                       N: i64, d: i64, K: i64) -> i64 {
    assert(sigma.len == K);
    var bad: i64 = 0;
    for j in N {
        let x = Z[j * d..(j + 1) * d];
        let dist = dist_mat[j * K..(j + 1) * K];
        for v, k in dist { dist[k] = 0.0; }
        for x_a, a in x {
            let y_a = Y_t[a * K..(a + 1) * K];
            assert(y_a.len == dist.len);
            for y, k in y_a { dist[k] += x_a * y; }
        }
        let logits = R[j * K..(j + 1) * K];
        assert(logits.len == dist.len);
        assert(sigma.len == dist.len);
        for v, k in dist {
            let distance = 2.0 * (1.0 - v);
            dist[k] = distance;
            logits[k] = (-distance) / sigma[k];
        }
        if !softmax_column(logits) { bad += 1; }
    }
    return bad;
}

// The inner loop of reassign_block_runs for the cells of one run, all in one
// batch, with one batch variable: logits from the distances plus the batch's
// log diversity penalty, a softmax, the sum of the new assignments over the
// run, and the run's objective terms. Returns the flags (1: a bad normalizer,
// 2: a column that does not sum to one).
export fn goose_reassign(dist_mat: const f32[:], sigma: const f32[:], lead_div: const f32[:], cells: const u32[:],
                         K: i64, R: f32[:], sums: f32[:], error_out: f64[:], entropy_out: f64[:]) -> i64 {
    assert(sigma.len == K);
    assert(lead_div.len == K);
    assert(sums.len == K);
    for v, k in sums { sums[k] = 0.0; }
    var error: f64 = 0.0;
    var entropy: f64 = 0.0;
    var found: i64 = 0;
    for j32 in cells {
        let j = j32 as i64;
        let dist = dist_mat[j * K..(j + 1) * K];
        let logits = R[j * K..(j + 1) * K];
        assert(logits.len == dist.len);
        assert(sigma.len == dist.len);
        assert(lead_div.len == dist.len);
        assert(sums.len == dist.len);
        for v, k in dist {
            let scaled = (-v) / sigma[k];
            logits[k] = scaled + lead_div[k];
        }
        var max_logit = logits[0];
        for v in logits { if v > max_logit { max_logit = v; } }
        var weighted: f64 = 0.0;
        var weighted_shifted: f64 = 0.0;
        for v, k in logits {
            let shifted = v - max_logit;
            let e = exp(shifted);
            logits[k] = e;
            let w = (sigma[k] as f64) * e;
            weighted += w;
            weighted_shifted += w * shifted;
        }
        let total = accumulate(logits);
        if !finite_positive(total) {
            found |= 1;
            continue;
        }
        var cell_error: f64 = 0.0;
        for v, k in logits {
            let p = v / total;
            logits[k] = p;
            sums[k] += p;
            cell_error += (p as f64) * dist[k];
        }
        let column_sum = accumulate(logits);
        let cs = column_sum as f64;
        if is_nan(cs) || is_inf(cs) || abs(column_sum - 1.0) > 1e-4 { found |= 2; }
        error += cell_error;
        entropy += (weighted_shifted - log(total as f64) * weighted) / (total as f64);
    }
    error_out[0] = error;
    entropy_out[0] = entropy;
    return found;
}
