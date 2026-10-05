# 2.1.0 - 2026-10-05

### Performance
- The C++ backend is 20-50x faster on large datasets: on an Apple M1 Ultra,
  858k cells corrected for batch take 1.9 s instead of 68 s, and corrected
  for batch and sample (870 levels) 3.1 s instead of 160 s. Most of the
  time was spent outside BLAS on a single core:
  - The ridge correction made two passes over the data per cluster. It now
    sums the terms of every cluster together, in one pass over the cells per
    batch variable, corrects each cell once, and solves each cluster's system
    by eliminating the diagonal
    block of the covariate with the most kept levels, instead of inverting a
    matrix with one row per level when correcting for several covariates.
  - Cluster assignments are updated in one pass per cell (logits, softmax,
    objective terms and checks), without copying the assignment and distance
    matrices twice per round.
  - The k-means start reproduces `arma::kmeans` (the same arithmetic and
    tie-breaking), split across threads.
  - The extension was built with `-Os` (nanobind's default), which turned off
    loop vectorization; it is now built with `-O3`.
- The work, including the large matrix products and each cluster's ridge
  system, runs on a pool of threads (no OpenMP). Results are identical for
  any `ncores`. On one thread, the 858k-cell run takes 18.0 s. On a 6-core
  Linux laptop (AMD Ryzen 5 5560U), it takes 5.2 s instead of 114 s with the
  2.0.2 wheel.
- Peak memory for the 858k-cell run is 1.7 GB instead of 2.7 GB (the whole
  process, on macOS with default settings).
- On Tahoe-100M (95.6 million cells, 1,344 samples), all cells take 4.9 min
  on a 64-core server instead of 4.5 h with 2.0.2 (which ran 3 Harmony
  iterations to this version's 1), with a peak of 214 GB instead of 294 GB.
  See `notebooks/tahoe-100m-benchmark/`.
- String and other non-numeric batch labels are numbered in one pass
  through a hash table, as `pandas.factorize` does, instead of by sorting
  every label with `np.unique`. Only the distinct labels are sorted, and
  labels that a hash table would number differently (equal labels that hash
  differently, such as `np.float32(0.1)` and `0.1`) still go through
  `np.unique`, so the numbering is unchanged. On an M1 Ultra, numbering 16 million string labels with 800
  distinct values takes 0.5-1.5 s instead of 7-23 s, depending on how they
  are stored, and the batch and sample labels of the 858k-cell dataset, as
  pandas 3 string columns, take 0.14 s instead of 1.3 s. A pandas
  `Categorical`, as in AnnData's `obs`, is renumbered from its codes (0.08 s
  for 16 million labels). `compute_lisi` numbers its labels the same way.

### Changed
- harmonypy no longer uses BLAS or LAPACK. Each cluster's ridge system is
  solved in double precision by a Cholesky factorization, and Armadillo is
  compiled without BLAS and LAPACK, so the extension links only the system C
  and C++ runtime libraries. The Linux wheels no longer bundle OpenBLAS and
  libgfortran: the Python 3.12 aarch64 wheel is 218 kB instead of 5.1 MB for
  2.0.2 (523 kB installed), and building from source needs only a C++17
  compiler and the Python headers (CMake and the Armadillo headers are
  downloaded if they are missing).
- `ncores` sets the number of threads harmonypy uses, at most the number of
  CPUs available to the process. The default (0) uses one thread per
  physical core of those CPUs: a second thread per core (hyperthreading) made
  Harmony about 5% faster on a 6-core laptop and no faster on a 64-core
  server, where it used 70% more CPU time. Previously it set
  `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS` after the BLAS library had
  read them, which had no effect, and these variables are no longer set.
- The objective is accumulated in double precision. The float32 sums were
  off by about 0.7% on 858k cells, and by much more for inputs with large
  values, which could flip the convergence check when the objective barely
  changes; such runs may now stop after a different number of iterations.
  Otherwise the corrected coordinates agree closely with 2.0.2, though not to
  float32 rounding: the per-PC correlation is at least 0.99999 over 16
  configurations of 3.5k-1M cells with 1-3 covariates on macOS, and at least
  0.99998 against the 2.0.2 wheel on Linux. The per-PC correlation with
  stored R Harmony outputs is the same to 3 decimals.
- `Harmony.Z_corr`, `Z_orig`, `Z_cos` and `R` are now C-contiguous arrays.
- A batch variable with missing labels (None, NaN, NaT or `pandas.NA`)
  raises a `ValueError` that names the column, in `run_harmony` and
  `compute_lisi`. Before, string labels with a missing value raised a
  `TypeError` from sorting, and in numeric labels the missing cells became a
  batch of their own, or one batch each in an object column.
- Coordinates with NaN or infinite values raise `ValueError`; before, a NaN
  could make the k-means start loop forever. A `sigma`, `theta` or `lamb` of
  the wrong length raises `ValueError`, as does a negative or non-finite `lamb`
  (other than -1, which like `None` selects the estimate) or, when `lamb` is
  estimated, a negative or non-finite `alpha`. A singular ridge system raises
  a `RuntimeError` that says so. `lamb=0` or `alpha=0` makes it singular
  whenever a cluster corrects every level of a batch variable, which with one
  batch variable is always the case.

### Fixed
- With a small fixed `lamb`, the ridge correction could be far from the
  exact regression. The intercept is the sum of each variable's batch
  indicators, so with a small `lamb` the ridge system is nearly singular, and
  its solution depended on the last digits of sums accumulated in float32.
  With `lamb=0.001`, the rms error of one correction step was 6-14% of the
  correction itself with one batch variable (synthetic data, 68k-200k cells;
  in one case the coordinates became non-finite), and 1.3 to 13 times the
  correction with two or three (69k-858k cells). The ridge sums are now
  accumulated in double precision and the system is solved in double
  precision, so one correction step matches a float64 solution to within
  2.5e-7 (root mean square), also with the default `lamb`, where 2.0.2 was
  off by up to 8e-3.
- `lamb` and `theta` given as NumPy values failed: an array `lamb` with more
  than one value raised `ValueError` ("The truth value of an array ... is
  ambiguous"), and a NumPy integer or float32 `lamb` or `theta` raised
  `TypeError`. They are now accepted like lists and Python numbers. A `theta`
  of the wrong length raises `ValueError` instead of failing an `assert`, and
  so do values that are not numbers. A fixed `lamb` vector whose first value
  was negative silently selected the estimate (the backend's internal signal
  for it); it now raises `ValueError`.
- `nclust=1` with the default `sigma` failed with a `TypeError` (an
  `AttributeError` with `verbose=False`), because a single `sigma` was
  expanded to one value per cluster only when there were several clusters;
  an integer `sigma` failed the same way for any `nclust`.
- Building from source with CMake older than 3.24 (e.g. Ubuntu 22.04's 3.22)
  failed when Armadillo was not installed, because the header download used
  a CMake 3.24 option.
- `Harmony.Z_cos` now returns the corrected coordinates with each cell scaled
  to unit length, as documented: the cosine-normalized coordinates Harmony
  clusters on. Since 2.0.0 it returned the same values as `Z_corr`.
- `batch_prop_cutoff` now compares the fraction of each batch's cells
  soft-assigned to a cluster with the cutoff, as R Harmony does, when
  correcting for more than one covariate. Since 2.0.0, each batch's cell count
  was divided by the number of covariates, so with two covariates (e.g. lab
  and day) and a cutoff of 0.1, a batch with 6% of its cells in a cluster
  counted as 12% and passed the cutoff there, where R Harmony skips it. In
  effect the cutoff was `batch_prop_cutoff / n_covariates`. Single-covariate
  results are unchanged. On a 68,785-cell dataset corrected for donor and
  batch, output before and after the fix is identical at the default cutoff
  (1e-5); at 0.01, the minimum per-PC correlation with R improves from 0.923
  to 0.970. See `notebooks/pr56-batch-prop-cutoff/`. Thanks to @jkhales for
  finding and fixing this (#56).

### Development
- `scripts/compare_outputs.py` runs the same configurations with two builds
  and compares their results and run times.
- Wheel builds run all of `tests/test_harmony.py`, not just the pbmc test.
- Wheels are built with cibuildwheel 4.2, which adds CPython 3.14 wheels;
  2.0.2 shipped none because cibuildwheel 2.21 predates 3.14.
- New tests check one ridge step against a direct float64 solve (1-3
  covariates in either order, estimated and small fixed `lamb`, clusters that
  drop batches, and a small `lamb` with 30,000-cell clusters), that `lamb=0`
  raises (also from a worker thread), that results do not depend on `ncores`
  (with the per-cluster solves on the thread pool), and the input checks.
- CI and the local Docker test script no longer install OpenBLAS.
- The local large-dataset run in `tests/test_harmony.py` read the row index
  in the unnamed first column of `acute_myeloid_pcs.tsv.gz` as a 29th PC.
  Unnamed columns are now skipped, so it corrects the 28 PCs, as R does; the
  minimum per-PC correlation with the reference rose from 0.953 to 0.990.

# 2.0.2 - 2026-09-16

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
- The pbmc test now compares against the tracked R harmony2 reference
  (`data/pbmc_3500_pcs_harmony2.tsv.gz`, generated by
  `scripts/generate_harmony2_reference.R`) and requires per-PC correlation
  >= 0.99. Previously that file was untracked, so CI silently fell back to a
  2022 R v1 reference with a 0.9 threshold and could not catch fidelity
  regressions. The old reference file is removed.

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
