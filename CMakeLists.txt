cmake_minimum_required(VERSION 3.15...3.27)
project(harmonypy LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Optional AddressSanitizer + UndefinedBehaviorSanitizer build (CI / debugging).
# Enable with: pip install -e . -C cmake.define.HARMONYPY_SANITIZE=ON
# The extension must then be imported with the ASan runtime preloaded; see the
# "asan" job in .github/workflows/python-package.yml.
option(HARMONYPY_SANITIZE "Build with AddressSanitizer and UBSan" OFF)
if(HARMONYPY_SANITIZE)
    message(STATUS "Building with AddressSanitizer + UBSan")
    add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=all
                        -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address,undefined)
endif()

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
    # DOWNLOAD_EXTRACT_TIMESTAMP exists from CMake 3.24; older versions would
    # read it as part of the URL list and fail.
    set(ARMADILLO_FETCH_OPTIONS)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(ARMADILLO_FETCH_OPTIONS DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    endif()
    FetchContent_Declare(
        armadillo
        URL https://sourceforge.net/projects/arma/files/armadillo-14.2.2.tar.xz
        ${ARMADILLO_FETCH_OPTIONS}
    )
    # Use FetchContent_Populate (not MakeAvailable) to get headers only,
    # without adding Armadillo's install targets to the wheel.
    FetchContent_Populate(armadillo)
    set(ARMADILLO_INCLUDE ${armadillo_SOURCE_DIR}/include)
endif()

# Create the Python module. NOMINSIZE keeps -O3: by default nanobind adds -Os,
# which turns off loop vectorization in the numerical code.
nanobind_add_module(_harmony_cpp
    NB_STATIC
    NOMINSIZE
    src/harmony.cpp
    src/bindings.cpp
)

# Armadillo in header-only mode, without BLAS or LAPACK: the matrix products
# and the small linear systems are computed in src/harmony.cpp, so nothing
# needs to be linked.
target_compile_definitions(_harmony_cpp PRIVATE ARMA_DONT_USE_WRAPPER ARMA_DONT_USE_BLAS ARMA_DONT_USE_LAPACK)
target_include_directories(_harmony_cpp PRIVATE
    ${ARMADILLO_INCLUDE}
    ${CMAKE_SOURCE_DIR}/src
)

# std::thread needs pthreads on Linux
find_package(Threads REQUIRED)
target_link_libraries(_harmony_cpp PRIVATE Threads::Threads)

# Install target
install(TARGETS _harmony_cpp LIBRARY DESTINATION harmonypy)
