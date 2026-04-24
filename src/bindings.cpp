// harmonypy - Python bindings for Harmony algorithm
// Copyright (C) 2018  Ilya Korsunsky
//               2019  Kamil Slowikowski <kslowikowski@gmail.com>

#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>
#include "harmony.hpp"
#include "lisi.hpp"

namespace nb = nanobind;
using namespace harmony;

// Input array types (C-contiguous, CPU)
using NpDouble2D = nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using NpDouble1D = nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using NpInt64_2D = nb::ndarray<int64_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

// Convert NumPy 2D array (double, row-major) to Armadillo matrix (col-major)
arma::mat numpy_to_arma_mat(NpDouble2D arr) {
    size_t nrows = arr.shape(0), ncols = arr.shape(1);
    const double* ptr = arr.data();
    arma::mat result(nrows, ncols);
    for (size_t i = 0; i < nrows; ++i)
        for (size_t j = 0; j < ncols; ++j)
            result(i, j) = ptr[i * ncols + j];
    return result;
}

// Convert NumPy 1D array to Armadillo vector
arma::vec numpy_to_arma_vec(NpDouble1D arr) {
    size_t n = arr.shape(0);
    const double* ptr = arr.data();
    arma::vec result(n);
    for (size_t i = 0; i < n; ++i)
        result(i) = ptr[i];
    return result;
}

// Convert NumPy 2D int64 array to Armadillo int64 matrix (row-major → col-major)
arma::Mat<int64_t> numpy_to_arma_imat(NpInt64_2D arr) {
    size_t nrows = arr.shape(0), ncols = arr.shape(1);
    const int64_t* ptr = arr.data();
    arma::Mat<int64_t> result(nrows, ncols);
    for (size_t i = 0; i < nrows; ++i)
        for (size_t j = 0; j < ncols; ++j)
            result(i, j) = ptr[i * ncols + j];
    return result;
}

// Convert Armadillo matrix to NumPy array (returns owned memory via capsule)
nb::ndarray<nb::numpy, double, nb::ndim<2>> arma_mat_to_numpy(const arma::mat& m) {
    size_t nrows = m.n_rows, ncols = m.n_cols;
    double* data = new double[nrows * ncols];
    for (size_t i = 0; i < nrows; ++i)
        for (size_t j = 0; j < ncols; ++j)
            data[i * ncols + j] = m(i, j);

    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    size_t shape[2] = { nrows, ncols };
    return nb::ndarray<nb::numpy, double, nb::ndim<2>>(data, 2, shape, std::move(owner));
}

// Wrapper class that handles numpy conversion
class HarmonyWrapper {
public:
    std::unique_ptr<Harmony> harmony;

    HarmonyWrapper(
        NpDouble2D Z,
        NpInt64_2D batch_of_cell,  // n_cov x N int64 — compact, O(N) memory
        NpDouble1D Pr_b,
        NpDouble1D sigma,
        NpDouble1D theta,
        NpDouble1D lambda,
        double alpha,
        int max_iter_harmony,
        int max_iter_kmeans,
        double epsilon_kmeans,
        double epsilon_harmony,
        int K,
        double block_size,
        std::vector<int> B_vec,
        double batch_proportion_cutoff,
        bool verbose,
        int random_state,
        std::function<void(const std::string&)> log_fn
    ) {
        harmony = std::make_unique<Harmony>(
            numpy_to_arma_mat(Z),
            numpy_to_arma_imat(batch_of_cell),
            numpy_to_arma_vec(Pr_b),
            numpy_to_arma_vec(sigma),
            numpy_to_arma_vec(theta),
            numpy_to_arma_vec(lambda),
            alpha,
            max_iter_harmony,
            max_iter_kmeans,
            epsilon_kmeans,
            epsilon_harmony,
            K,
            block_size,
            B_vec,
            batch_proportion_cutoff,
            verbose,
            random_state,
            std::move(log_fn)
        );
    }

    nb::ndarray<nb::numpy, double, nb::ndim<2>> result() const { return arma_mat_to_numpy(harmony->result()); }
    nb::ndarray<nb::numpy, double, nb::ndim<2>> Z_corr() const { return arma_mat_to_numpy(harmony->get_Z_corr()); }
    nb::ndarray<nb::numpy, double, nb::ndim<2>> Z_orig() const { return arma_mat_to_numpy(harmony->get_Z_orig()); }
    nb::ndarray<nb::numpy, double, nb::ndim<2>> Z_cos() const { return arma_mat_to_numpy(harmony->get_Z_cos()); }
    nb::ndarray<nb::numpy, double, nb::ndim<2>> R() const { return arma_mat_to_numpy(harmony->get_R()); }
    nb::ndarray<nb::numpy, double, nb::ndim<2>> Y() const { return arma_mat_to_numpy(harmony->get_Y()); }
    int K() const { return harmony->K; }
    int N() const { return harmony->N; }
    int d() const { return harmony->d; }
    std::vector<double> objective_harmony() const {
        return std::vector<double>(harmony->objective_harmony.begin(), harmony->objective_harmony.end());
    }
    std::vector<double> objective_kmeans() const {
        return std::vector<double>(harmony->objective_kmeans.begin(), harmony->objective_kmeans.end());
    }
    std::vector<int> kmeans_rounds() const { return harmony->kmeans_rounds; }
};

NB_MODULE(_harmony_cpp, m) {
    m.doc() = "C++ implementation of Harmony algorithm (matches R package)";

    nb::class_<HarmonyWrapper>(m, "HarmonyCpp")
        .def(nb::init<
            NpDouble2D,            // Z
            NpInt64_2D,            // batch_of_cell (n_cov x N)
            NpDouble1D,            // Pr_b
            NpDouble1D,            // sigma
            NpDouble1D,            // theta
            NpDouble1D,            // lambda
            double,                // alpha
            int,                   // max_iter_harmony
            int,                   // max_iter_kmeans
            double,                // epsilon_kmeans
            double,                // epsilon_harmony
            int,                   // K (nclust)
            double,                // block_size
            std::vector<int>,      // B_vec
            double,                // batch_proportion_cutoff
            bool,                  // verbose
            int,                   // random_state
            std::function<void(const std::string&)>  // log_fn
        >(),
            nb::arg("Z"),
            nb::arg("batch_of_cell"),
            nb::arg("Pr_b"),
            nb::arg("sigma"),
            nb::arg("theta"),
            nb::arg("lambda"),
            nb::arg("alpha"),
            nb::arg("max_iter_harmony"),
            nb::arg("max_iter_kmeans"),
            nb::arg("epsilon_kmeans"),
            nb::arg("epsilon_harmony"),
            nb::arg("K"),
            nb::arg("block_size"),
            nb::arg("B_vec"),
            nb::arg("batch_proportion_cutoff"),
            nb::arg("verbose"),
            nb::arg("random_state"),
            nb::arg("log_fn")
        )
        .def("result", &HarmonyWrapper::result, nb::rv_policy::move,
             "Get the corrected data matrix")
        .def_prop_ro("Z_corr", &HarmonyWrapper::Z_corr, nb::rv_policy::move,
                      "Corrected data matrix (d x N)")
        .def_prop_ro("Z_orig", &HarmonyWrapper::Z_orig, nb::rv_policy::move,
                      "Original data matrix (d x N)")
        .def_prop_ro("Z_cos", &HarmonyWrapper::Z_cos, nb::rv_policy::move,
                      "L2-normalized data matrix (d x N)")
        .def_prop_ro("R", &HarmonyWrapper::R, nb::rv_policy::move,
                      "Soft cluster assignments (K x N)")
        .def_prop_ro("Y", &HarmonyWrapper::Y, nb::rv_policy::move,
                      "Cluster centroids (d x K)")
        .def_prop_ro("K", &HarmonyWrapper::K, "Number of clusters")
        .def_prop_ro("N", &HarmonyWrapper::N, "Number of cells")
        .def_prop_ro("d", &HarmonyWrapper::d, "Number of dimensions")
        .def_prop_ro("objective_harmony", &HarmonyWrapper::objective_harmony,
                      "Harmony objective values per iteration")
        .def_prop_ro("objective_kmeans", &HarmonyWrapper::objective_kmeans,
                      "K-means objective values")
        .def_prop_ro("kmeans_rounds", &HarmonyWrapper::kmeans_rounds,
                      "Number of k-means rounds per harmony iteration");

    // LISI function
    using NpDouble2D_RO = nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
    using NpInt32_1D_RO = nb::ndarray<int, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

    m.def("compute_lisi_cpp",
        [](NpDouble2D_RO X, NpInt32_1D_RO labels, int n_categories, double perplexity)
            -> nb::ndarray<nb::numpy, double, nb::ndim<1>> {
            size_t N = X.shape(0), d = X.shape(1);
            const double* x_ptr = X.data();
            // Convert row-major numpy to arma::mat (col-major)
            arma::mat X_arma(N, d);
            for (size_t i = 0; i < N; ++i)
                for (size_t j = 0; j < d; ++j)
                    X_arma(i, j) = x_ptr[i * d + j];

            arma::vec result = lisi::compute_lisi_impl(
                X_arma, labels.data(), n_categories, perplexity
            );

            double* out = new double[N];
            std::memcpy(out, result.memptr(), N * sizeof(double));
            nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<double*>(p); });
            size_t shape[1] = { N };
            return nb::ndarray<nb::numpy, double, nb::ndim<1>>(out, 1, shape, std::move(owner));
        },
        nb::arg("X"), nb::arg("labels"), nb::arg("n_categories"), nb::arg("perplexity"),
        nb::rv_policy::move,
        "Compute LISI for one label column. Returns N-length array of LISI values."
    );
}
