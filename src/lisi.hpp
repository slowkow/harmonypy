// LISI - The Local Inverse Simpson Index
// C++ implementation replacing sklearn-based Python version.
// Copyright (C) 2018  Ilya Korsunsky
//               2019  Kamil Slowikowski <kslowikowski@gmail.com>

#ifndef LISI_HPP
#define LISI_HPP

#include <armadillo>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

namespace lisi {

// Simple kd-tree for k-nearest-neighbor search
struct KDNode {
    int split_dim;
    double split_val;
    int point_idx;        // -1 for internal nodes
    int left, right;      // child indices (-1 for none)
};

class KDTree {
public:
    const arma::mat& data;  // N x d, row-major access
    std::vector<KDNode> nodes;
    int root;

    KDTree(const arma::mat& data) : data(data) {
        std::vector<int> indices(data.n_rows);
        for (size_t i = 0; i < data.n_rows; ++i) indices[i] = i;
        nodes.reserve(2 * data.n_rows);
        root = build(indices, 0, indices.size(), 0);
    }

    // Find k nearest neighbors for query point (excluding itself)
    void knn(int query_idx, int k,
             std::vector<int>& nn_indices,
             std::vector<double>& nn_dists) const {
        // Max-heap of (distance, index)
        std::vector<std::pair<double, int>> heap;
        heap.reserve(k + 1);
        double max_dist = std::numeric_limits<double>::max();
        knn_search(root, query_idx, k, heap, max_dist);

        // Sort by distance
        std::sort(heap.begin(), heap.end());
        nn_indices.resize(heap.size());
        nn_dists.resize(heap.size());
        for (size_t i = 0; i < heap.size(); ++i) {
            nn_dists[i] = std::sqrt(heap[i].first);  // convert squared to actual distance
            nn_indices[i] = heap[i].second;
        }
    }

private:
    int build(std::vector<int>& indices, int begin, int end, int depth) {
        if (begin >= end) return -1;
        if (end - begin == 1) {
            int idx = nodes.size();
            nodes.push_back({-1, 0, indices[begin], -1, -1});
            return idx;
        }
        int dim = depth % data.n_cols;
        int mid = begin + (end - begin) / 2;
        std::nth_element(indices.begin() + begin, indices.begin() + mid,
                         indices.begin() + end,
                         [&](int a, int b) { return data(a, dim) < data(b, dim); });
        int idx = nodes.size();
        nodes.push_back({dim, data(indices[mid], dim), indices[mid], -1, -1});
        nodes[idx].left = build(indices, begin, mid, depth + 1);
        nodes[idx].right = build(indices, mid + 1, end, depth + 1);
        return idx;
    }

    double sq_dist(int a, int b) const {
        double d = 0;
        for (size_t j = 0; j < data.n_cols; ++j) {
            double diff = data(a, j) - data(b, j);
            d += diff * diff;
        }
        return d;
    }

    void knn_search(int node_idx, int query_idx, int k,
                    std::vector<std::pair<double, int>>& heap,
                    double& max_dist) const {
        if (node_idx < 0) return;
        const KDNode& node = nodes[node_idx];

        int pt = node.point_idx;
        if (pt != query_idx) {
            double d = sq_dist(query_idx, pt);
            if (static_cast<int>(heap.size()) < k) {
                heap.push_back({d, pt});
                std::push_heap(heap.begin(), heap.end());
                if (static_cast<int>(heap.size()) == k)
                    max_dist = heap.front().first;
            } else if (d < max_dist) {
                std::pop_heap(heap.begin(), heap.end());
                heap.back() = {d, pt};
                std::push_heap(heap.begin(), heap.end());
                max_dist = heap.front().first;
            }
        }

        if (node.left < 0 && node.right < 0) return;

        double diff = data(query_idx, node.split_dim) - node.split_val;
        int near = diff <= 0 ? node.left : node.right;
        int far = diff <= 0 ? node.right : node.left;

        knn_search(near, query_idx, k, heap, max_dist);
        if (diff * diff < max_dist || static_cast<int>(heap.size()) < k)
            knn_search(far, query_idx, k, heap, max_dist);
    }
};

// Compute Simpson's index for one cell given its neighbor distances and labels
inline double compute_simpson_one(
    const double* distances, const int* indices, int n_neighbors,
    const int* labels, int n_categories, double perplexity, double tol = 1e-5
) {
    double logU = std::log(perplexity);
    double beta = 1.0;
    double betamin = -std::numeric_limits<double>::infinity();
    double betamax = std::numeric_limits<double>::infinity();

    std::vector<double> P(n_neighbors);
    double H = 0, P_sum = 0;

    // Binary search for beta that gives the target perplexity
    for (int t = 0; t < 50; ++t) {
        P_sum = 0;
        double sum_dP = 0;
        for (int j = 0; j < n_neighbors; ++j) {
            P[j] = std::exp(-distances[j] * beta);
            P_sum += P[j];
            sum_dP += distances[j] * P[j];
        }
        if (P_sum == 0) {
            H = 0;
            for (int j = 0; j < n_neighbors; ++j) P[j] = 0;
        } else {
            H = std::log(P_sum) + beta * sum_dP / P_sum;
            for (int j = 0; j < n_neighbors; ++j) P[j] /= P_sum;
        }

        double Hdiff = H - logU;
        if (std::abs(Hdiff) < tol) break;

        if (Hdiff > 0) {
            betamin = beta;
            beta = std::isfinite(betamax) ? (beta + betamax) / 2 : beta * 2;
        } else {
            betamax = beta;
            beta = std::isfinite(betamin) ? (beta + betamin) / 2 : beta / 2;
        }
    }

    if (H == 0) return -1.0;

    // Simpson's index: sum of squared category probabilities
    std::vector<double> cat_prob(n_categories, 0.0);
    for (int j = 0; j < n_neighbors; ++j) {
        cat_prob[labels[indices[j]]] += P[j];
    }
    double simpson = 0.0;
    for (int c = 0; c < n_categories; ++c) {
        simpson += cat_prob[c] * cat_prob[c];
    }
    return simpson;
}

// Compute LISI for all cells
// X: N x d data matrix
// labels: N-length array of integer labels (0-indexed)
// n_categories: number of distinct categories
// perplexity: target perplexity (default 30)
// Returns: N-length vector of LISI values
inline arma::vec compute_lisi_impl(
    const arma::mat& X, const int* labels, int n_categories, double perplexity
) {
    int N = X.n_rows;
    int k = static_cast<int>(perplexity * 3);
    arma::vec lisi_result(N);

    KDTree tree(X);

    for (int i = 0; i < N; ++i) {
        std::vector<int> nn_idx;
        std::vector<double> nn_dist;
        tree.knn(i, k, nn_idx, nn_dist);

        double simpson = compute_simpson_one(
            nn_dist.data(), nn_idx.data(), nn_idx.size(),
            labels, n_categories, perplexity
        );
        lisi_result(i) = (simpson > 0) ? 1.0 / simpson : 0.0;
    }
    return lisi_result;
}

} // namespace lisi

#endif // LISI_HPP
