cmake_minimum_required(VERSION 3.15...3.27)
project(harmonypy LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Find packages
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)
find_package(Armadillo REQUIRED)

# OpenMP support (disabled on macOS to avoid libomp conflicts)
if(NOT APPLE)
    find_package(OpenMP)
endif()

# Create the Python module
nanobind_add_module(_harmony_cpp
    NB_STATIC
    src/harmony.cpp
    src/bindings.cpp
)

# Link Armadillo
target_link_libraries(_harmony_cpp PRIVATE ${ARMADILLO_LIBRARIES})
target_include_directories(_harmony_cpp PRIVATE ${ARMADILLO_INCLUDE_DIRS})

# Include directories
target_include_directories(_harmony_cpp PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(_harmony_cpp PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(_harmony_cpp PRIVATE HARMONY_USE_OPENMP)
    message(STATUS "OpenMP enabled for harmonypy")
endif()

# Install target
install(TARGETS _harmony_cpp LIBRARY DESTINATION harmonypy)
