// harmonypy - Python bindings for Harmony algorithm
// Copyright (C) 2018  Ilya Korsunsky
//               2019  Kamil Slowikowski <kslowikowski@gmail.com>

#include <memory>
#include <stdexcept>
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

// Return the columns of a float matrix as the rows of a NumPy array (one row
// per cell). Armadillo stores columns contiguously, so this is a straight
// float-to-double copy.
nb::ndarray<nb::numpy, double, nb::ndim<2>> columns_as_rows(const MATTYPE& m) {
    size_t nrows = m.n_cols, ncols = m.n_rows;
    double* data = new double[m.n_elem];
    const float* src = m.memptr();
    for (size_t i = 0; i < m.n_elem; ++i) data[i] = src[i];

    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    size_t shape[2] = { nrows, ncols };
    return nb::ndarray<nb::numpy, double, nb::ndim<2>>(data, 2, shape, std::move(owner));
}

// Wrapper class that handles numpy conversion
class HarmonyWrapper {
public:
    std::unique_ptr<Harmony> harmony;

    HarmonyWrapper(
        NpDouble2D Z,              // N x d (cells x PCs)
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
        int ncores,
        std::function<void(const std::string&)> log_fn
    ) {
        // A row-major N x d array has the memory layout of a column-major
        // d x N matrix, so Armadillo can read it in place.
        const arma::mat Z_cols(const_cast<double*>(Z.data()), Z.shape(1), Z.shape(0), false, true);
        harmony = std::make_unique<Harmony>(
            Z_cols,
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
            ncores,
            std::move(log_fn)
        );
    }

    nb::ndarray<nb::numpy, double, nb::ndim<2>> result() const { return columns_as_rows(harmony->Z_corr); }
    nb::ndarray<nb::numpy, double, nb::ndim<2>> Z_corr() const { return columns_as_rows(harmony->Z_corr); }
    nb::ndarray<nb::numpy, double, nb::ndim<2>> Z_orig() const { return columns_as_rows(harmony->Z_orig); }
    nb::ndarray<nb::numpy, double, nb::ndim<2>> Z_cos() const { return columns_as_rows(harmony->Z_corr); }
    nb::ndarray<nb::numpy, double, nb::ndim<2>> R() const { return columns_as_rows(harmony->R); }
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

    m.def("_objective_converged", &objective_converged,
          nb::arg("obj_old"), nb::arg("obj_new"), nb::arg("epsilon"));

    m.def("_assignment_probabilities", [](
        NpDouble2D distances,
        NpDouble1D sigma,
        NpDouble2D E,
        NpDouble2D O,
        NpDouble1D theta,
        NpInt64_2D batch_ids
    ) {
        MATTYPE logits = assignment_logits(
            arma::conv_to<MATTYPE>::from(numpy_to_arma_mat(distances)),
            arma::conv_to<VECTYPE>::from(numpy_to_arma_vec(sigma)),
            arma::conv_to<MATTYPE>::from(numpy_to_arma_mat(E)),
            arma::conv_to<MATTYPE>::from(numpy_to_arma_mat(O)),
            arma::conv_to<VECTYPE>::from(numpy_to_arma_vec(theta)),
            arma::conv_to<arma::Mat<arma::uword>>::from(numpy_to_arma_imat(batch_ids))
        );
        ROWTYPE normalizers = exponentiate_shifted_logits(logits);
        if (!normalizers.is_finite() || normalizers.min() <= 0.0f)
            throw std::runtime_error("assignment normalizers must be finite and positive");
        logits.each_row() /= normalizers;
        arma::mat result = arma::conv_to<arma::mat>::from(logits);
        return arma_mat_to_numpy(result);
    },
        nb::arg("distances"), nb::arg("sigma"), nb::arg("E"), nb::arg("O"),
        nb::arg("theta"), nb::arg("batch_ids"), nb::rv_policy::move);

    nb::class_<HarmonyWrapper>(m, "HarmonyCpp")
        .def(nb::init<
            NpDouble2D,            // Z (N x d)
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
            int,                   // ncores
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
            nb::arg("ncores"),
            nb::arg("log_fn")
        )
        .def("result", &HarmonyWrapper::result, nb::rv_policy::move,
             "Get the corrected data matrix (N x d)")
        .def_prop_ro("Z_corr", &HarmonyWrapper::Z_corr, nb::rv_policy::move,
                      "Corrected data matrix (N x d)")
        .def_prop_ro("Z_orig", &HarmonyWrapper::Z_orig, nb::rv_policy::move,
                      "Original data matrix (N x d)")
        .def_prop_ro("Z_cos", &HarmonyWrapper::Z_cos, nb::rv_policy::move,
                      "L2-normalized data matrix (N x d)")
        .def_prop_ro("R", &HarmonyWrapper::R, nb::rv_policy::move,
                      "Soft cluster assignments (N x K)")
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
