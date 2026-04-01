// harmonypy - Python bindings for Harmony algorithm
// Copyright (C) 2018  Ilya Korsunsky
//               2019  Kamil Slowikowski <kslowikowski@gmail.com>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "harmony.hpp"

namespace py = pybind11;
using namespace harmony;

// Convert NumPy array to Armadillo matrix
// NumPy uses row-major (C-style), Armadillo uses column-major (Fortran-style)
arma::mat numpy_to_arma_mat(py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Expected 2D array");
    }
    
    size_t nrows = buf.shape[0];
    size_t ncols = buf.shape[1];
    double* ptr = static_cast<double*>(buf.ptr);
    
    arma::mat result(nrows, ncols);
    for (size_t i = 0; i < nrows; ++i) {
        for (size_t j = 0; j < ncols; ++j) {
            result(i, j) = ptr[i * ncols + j];
        }
    }
    return result;
}

// Convert NumPy array to Armadillo sparse matrix (CSC format expected)
// Phi is one-hot encoded batch indicators, very sparse
arma::sp_mat numpy_to_arma_sp_mat(py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Expected 2D array");
    }
    
    size_t nrows = buf.shape[0];
    size_t ncols = buf.shape[1];
    double* ptr = static_cast<double*>(buf.ptr);
    
    // Count non-zeros first
    std::vector<arma::uword> row_indices;
    std::vector<arma::uword> col_indices;
    std::vector<double> values;
    
    for (size_t i = 0; i < nrows; ++i) {
        for (size_t j = 0; j < ncols; ++j) {
            double val = ptr[i * ncols + j];
            if (val != 0.0) {
                row_indices.push_back(i);
                col_indices.push_back(j);
                values.push_back(val);
            }
        }
    }
    
    // Create locations matrix
    arma::umat locations(2, row_indices.size());
    for (size_t i = 0; i < row_indices.size(); ++i) {
        locations(0, i) = row_indices[i];
        locations(1, i) = col_indices[i];
    }
    
    arma::vec vals(values);
    
    return arma::sp_mat(locations, vals, nrows, ncols);
}

// Convert NumPy array to Armadillo vector
arma::vec numpy_to_arma_vec(py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Expected 1D array");
    }
    
    arma::vec result(buf.size);
    double* ptr = static_cast<double*>(buf.ptr);
    for (size_t i = 0; i < static_cast<size_t>(buf.size); ++i) {
        result(i) = ptr[i];
    }
    return result;
}

// Convert Armadillo matrix to NumPy array
py::array_t<double> arma_mat_to_numpy(const arma::mat& m) {
    py::array_t<double> result({static_cast<ssize_t>(m.n_rows), static_cast<ssize_t>(m.n_cols)});
    auto buf = result.mutable_unchecked<2>();
    for (size_t i = 0; i < m.n_rows; ++i) {
        for (size_t j = 0; j < m.n_cols; ++j) {
            buf(i, j) = m(i, j);
        }
    }
    return result;
}

// Wrapper class that handles numpy conversion
class HarmonyWrapper {
public:
    std::unique_ptr<Harmony> harmony;
    
    HarmonyWrapper(
        py::array_t<double> Z,
        py::array_t<double> Phi,       // Will be converted to sparse
        py::array_t<double> Pr_b,
        py::array_t<double> sigma,
        py::array_t<double> theta,
        py::array_t<double> lambda,
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
        int random_state
    ) {
        harmony = std::make_unique<Harmony>(
            numpy_to_arma_mat(Z),
            numpy_to_arma_sp_mat(Phi),  // Convert to sparse
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
            random_state
        );
    }
    
    py::array_t<double> result() const { return arma_mat_to_numpy(harmony->result()); }
    py::array_t<double> Z_corr() const { return arma_mat_to_numpy(harmony->get_Z_corr()); }
    py::array_t<double> Z_orig() const { return arma_mat_to_numpy(harmony->get_Z_orig()); }
    py::array_t<double> Z_cos() const { return arma_mat_to_numpy(harmony->get_Z_cos()); }
    py::array_t<double> R() const { return arma_mat_to_numpy(harmony->get_R()); }
    py::array_t<double> Y() const { return arma_mat_to_numpy(harmony->get_Y()); }
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

PYBIND11_MODULE(_harmony_cpp, m) {
    m.doc() = "C++ implementation of Harmony algorithm (matches R package)";
    
    py::class_<HarmonyWrapper>(m, "HarmonyCpp")
        .def(py::init<
            py::array_t<double>,   // Z
            py::array_t<double>,   // Phi (converted to sparse internally)
            py::array_t<double>,   // Pr_b
            py::array_t<double>,   // sigma
            py::array_t<double>,   // theta
            py::array_t<double>,   // lambda
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
            int                    // random_state
        >(),
            py::arg("Z"),
            py::arg("Phi"),
            py::arg("Pr_b"),
            py::arg("sigma"),
            py::arg("theta"),
            py::arg("lambda"),
            py::arg("alpha"),
            py::arg("max_iter_harmony"),
            py::arg("max_iter_kmeans"),
            py::arg("epsilon_kmeans"),
            py::arg("epsilon_harmony"),
            py::arg("K"),
            py::arg("block_size"),
            py::arg("B_vec"),
            py::arg("batch_proportion_cutoff"),
            py::arg("verbose"),
            py::arg("random_state")
        )
        .def("result", &HarmonyWrapper::result, "Get the corrected data matrix")
        .def_property_readonly("Z_corr", &HarmonyWrapper::Z_corr, "Corrected data matrix (d x N)")
        .def_property_readonly("Z_orig", &HarmonyWrapper::Z_orig, "Original data matrix (d x N)")
        .def_property_readonly("Z_cos", &HarmonyWrapper::Z_cos, "L2-normalized data matrix (d x N)")
        .def_property_readonly("R", &HarmonyWrapper::R, "Soft cluster assignments (K x N)")
        .def_property_readonly("Y", &HarmonyWrapper::Y, "Cluster centroids (d x K)")
        .def_property_readonly("K", &HarmonyWrapper::K, "Number of clusters")
        .def_property_readonly("N", &HarmonyWrapper::N, "Number of cells")
        .def_property_readonly("d", &HarmonyWrapper::d, "Number of dimensions")
        .def_property_readonly("objective_harmony", &HarmonyWrapper::objective_harmony, 
                              "Harmony objective values per iteration")
        .def_property_readonly("objective_kmeans", &HarmonyWrapper::objective_kmeans,
                              "K-means objective values")
        .def_property_readonly("kmeans_rounds", &HarmonyWrapper::kmeans_rounds,
                              "Number of k-means rounds per harmony iteration");
}
