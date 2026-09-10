# 2.0.2 - 2026-09-10

### Fixed
- Fixed the ridge correction when correcting for more than one covariate
  (e.g. lab and processing day). The per-batch shortcut introduced in the C++
  rewrite assumed each cell belongs to a single batch, which double-counted
  cells in the intercept term and dropped the overlap between covariate
  groups, overcorrecting: a two-cell lab/day example was reversed from
  `[1, 2]` to `[1.67, 1.33]`. The correction now follows the R Harmony ridge
  calculation and gives `[1.4, 1.6]`, matching R harmony 2.0.4 to within
  1.2e-7. Single-covariate results are unchanged. Thanks to @jkhales for
  finding and fixing this (#54).
- Harmony no longer reports convergence when the objective *increases*
  between iterations; a non-negative relative decrease below
  `epsilon_harmony` is now required, so affected runs continue optimizing
  instead of stopping early. Mirrors R Harmony PR immunogenomics/harmony#293.
  Results are unchanged for runs whose objective decreases monotonically.
  Thanks to @fderop (#55).
- Cluster assignments are computed in log space (a shifted softmax), so very
  small `sigma` or very large `theta` no longer underflow or overflow into NaN
  assignments. Results are unchanged for ordinary parameters. (#55)
- If the optimizer state becomes non-finite, `run_harmony` now raises a
  `RuntimeError` naming the stage and parameters instead of silently
  returning NaNs. (#55)

### Development
- The sanitizer CI job preloads libstdc++ alongside libasan so C++
  exceptions thrown by the extension are handled correctly under
  AddressSanitizer.

# 2.0.1 - 2026-09-09

### Fixed
- Fixed a memory-corruption bug when correcting for more than one covariate
  (e.g. lab and processing day). `covariate_bounds` was allocated one element
  too small, so `std::partial_sum` wrote one entry past the end of the buffer —
  undefined behavior that could silently corrupt memory or crash. The
  correction results are unchanged. Thanks to @jkhales for finding and fixing
  this (#53).

### Development
- Added an AddressSanitizer + UndefinedBehaviorSanitizer CI job that builds the
  extension with `-fsanitize=address,undefined` and runs the test suite under
  the sanitizer runtime, so memory errors like the one above fail CI. Build it
  locally with `pip install -e . -C cmake.define.HARMONYPY_SANITIZE=ON`.
- Linux wheels are now built on `manylinux_2_28` (glibc 2.28+, e.g. RHEL/
  AlmaLinux 8, Ubuntu 18.10+) instead of `manylinux2014`. Recent NumPy releases
  no longer ship `manylinux2014` wheels and require GCC >= 10.3, which the old
  build image did not provide; `manylinux_2_28` matches NumPy's own wheels.

# 2.0.0 - 2026-04-22

Complete rewrite with C++ backend ([Armadillo](https://arma.sourceforge.net/) +
[nanobind](https://github.com/wjakob/nanobind)), matching the
[R harmony2 package](https://github.com/immunogenomics/harmony) step-by-step.

### New
- C++ backend using Armadillo for BLAS-accelerated dense matrix operations
  (Accelerate on macOS, OpenBLAS on Linux). Custom scatter/gather kernels
  replace all sparse matrix operations by exploiting Phi's one-hot structure.
- Pre-built wheels for Linux (x86_64, aarch64) and macOS (x86_64, arm64),
  Python 3.9–3.13. Armadillo headers fetched at build time — no system
  install required.
- K-means initialization matches R exactly: Gumbel-max cosine-distance
  sampling followed by `arma::kmeans` refinement.
- `ncores` parameter to control BLAS thread count (0 = all cores, default).
- `batch_prop_cutoff` parameter (default 1e-5) excludes underrepresented
  batches from correction in each cluster.
- Arrowhead matrix inverse for fast single-covariate batch correction.
- Accepts pandas DataFrame, dict of arrays, or NumPy array for `meta_data`.
- Non-numeric DataFrame columns (e.g. barcodes) are dropped automatically.
- Stricter input validation with clear error messages for shape mismatches.
- C++ progress messages (e.g. "Iteration 1 of 10") now go through Python's
  `logging` module instead of `std::cout`, so they appear immediately and
  integrate with downstream packages' logging configuration. Thanks to
  Yakir Reshef (@yakirr) for reporting this.

### Breaking changes
- `lamb` now defaults to automatic lambda estimation (was fixed `1`).
  Pass `lamb=1` explicitly to restore previous behavior.
- Default parameters changed to match R harmony2:
  `max_iter_kmeans` 20→4, `epsilon_cluster` 1e-5→1e-3, `epsilon_harmony` 1e-4→1e-2.
- Requires a C++ compiler and BLAS library for building from source (pre-built
  wheels are available on PyPI).
- Only `numpy` is required at runtime (previously required pandas, scipy,
  scikit-learn, torch).

### Performance
- 858k cells in ~36s on Apple M1 Ultra (vs ~340s in v0.1.0, ~38s in R harmony2).
- Correlation with R harmony2: ≥0.998 across all PCs on test data.
- No sparse matrices allocated — memory usage ~50% lower than sparse approach.

# 0.2.0 - 2025-01-09

- PyTorch backend with GPU acceleration (CUDA, Apple Silicon MPS) and optimized CPU execution.
- Requires Python >= 3.9.

# 0.1.0 - 2025-01-08

- Pure NumPy implementation matching R package v1.2.4 formulas for improved accuracy.
- Correlation with R harmony results: >0.95 for all PCs.
- Performance benchmarks:
  - Small (3.5k cells): 1.88s
  - Medium (69k cells): 56.22s
  - Large (858k cells): 340.32s

# 0.0.10 - 2024-07-04

- Migrate to hatch to ease development and include multiple authors.
- Add @johnarevalo to the author list.

# 0.0.9 - 2022-11-23

- Stop excluding `README.md` from the build, because setup.py depends on this
  file.

# 0.0.8 - 2022-11-22

- Replace `scipy.cluster.vq.kmeans2` with the faster function
  `sklearn.cluster.KMeans`. Thanks to @johnarevalo for providing details about
  the running time with both functions in PR #20.

# 0.0.6 - 2022-02-02

- Replace `scipy.cluster.vq.kmeans` with `scipy.cluster.vq.kmeans2` to address
  issue #10 where we learned that kmeans does not always return k centroids,
  but kmeans2 does return k centroids. Thanks to @onionpork and @DennisPost10
  for reporting this.

# 0.0.5 - 2020-08-11

- Expose `max_iter_harmony` as a new top-level argument, in addition to the
  previously exposed `max_iter_kmeans`. This more closely resembles the
  original interface in the harmony R package. Thanks to @pinin4fjords
  for pull request #8

# 0.0.4 - 2020-03-02

- Fix a bug in the LISI code that sometimes causes computation to break. Thanks
  to @tariqdaouda for reporting it in issue #1

- Fix a bug that prevents controlling the number of iterations. Thanks to
  @liboxun for reporting it in issue #3

- Fix a bug causing slightly different results than expected. Thanks to
  @bli25broad for pull request #2

- Add support for multiple categorical batch variables.

# 0.0.3 - 2019-12-26

- Speed up the Harmony algorithm. It should now be as fast as the R package.

# 0.0.2 - 2019-12-20

- Add Local Inverse Simpson Index (LISI) functions from the lisi R package.
  <https://github.com/immunogenomics/LISI>

# 0.0.1 - 2019-12-19

- Initial release. Code ported directly from the harmony R package.
  <https://github.com/immunogenomics/harmony>
