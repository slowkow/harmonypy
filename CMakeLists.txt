cmake_minimum_required(VERSION 3.15...3.27)
project(harmonypy LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Find Python and nanobind
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)

# --- Armadillo: header-only mode ---
# Try system Armadillo first (for local development),
# fall back to FetchContent (for wheel builds)
find_package(Armadillo QUIET)
if(ARMADILLO_FOUND)
    message(STATUS "Using system Armadillo ${ARMADILLO_VERSION}")
    set(ARMADILLO_INCLUDE ${ARMADILLO_INCLUDE_DIRS})
else()
    message(STATUS "System Armadillo not found, fetching headers...")
    include(FetchContent)
    FetchContent_Declare(
        armadillo
        URL https://sourceforge.net/projects/arma/files/armadillo-14.2.2.tar.xz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    # Use FetchContent_Populate (not MakeAvailable) to get headers only,
    # without adding Armadillo's install targets to the wheel.
    FetchContent_Populate(armadillo)
    set(ARMADILLO_INCLUDE ${armadillo_SOURCE_DIR}/include)
endif()

# Create the Python module
nanobind_add_module(_harmony_cpp
    NB_STATIC
    src/harmony.cpp
    src/bindings.cpp
)

# Armadillo in header-only mode: call BLAS/LAPACK directly
target_compile_definitions(_harmony_cpp PRIVATE ARMA_DONT_USE_WRAPPER)
target_include_directories(_harmony_cpp PRIVATE
    ${ARMADILLO_INCLUDE}
    ${CMAKE_SOURCE_DIR}/src
)

# Link BLAS/LAPACK
if(APPLE)
    target_link_libraries(_harmony_cpp PRIVATE "-framework Accelerate")
else()
    # Find OpenBLAS (provides both BLAS and LAPACK)
    find_library(OPENBLAS_LIB openblas
        PATHS /usr/lib64 /usr/lib /usr/local/lib
              /usr/lib/x86_64-linux-gnu
              /usr/lib/aarch64-linux-gnu)
    if(OPENBLAS_LIB)
        message(STATUS "Found OpenBLAS: ${OPENBLAS_LIB}")
        target_link_libraries(_harmony_cpp PRIVATE ${OPENBLAS_LIB})
    else()
        # Fallback: try generic BLAS/LAPACK
        find_package(BLAS REQUIRED)
        find_package(LAPACK REQUIRED)
        target_link_libraries(_harmony_cpp PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    endif()
endif()

# Install target
install(TARGETS _harmony_cpp LIBRARY DESTINATION harmonypy)
