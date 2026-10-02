#!/usr/bin/env bash
# Reproduce every result and figure in this folder.
#
# Builds harmonypy twice (tag v2.0.2 and PR #56 as merged) into throwaway
# venvs, builds R harmony from the harmony2 branch with LAPACK enabled, runs
# all three on the same inputs, then plots.
#
# Needs: git, uv, a C++ toolchain, and R with data.table and harmony's build
# dependencies installed (Rcpp, RcppArmadillo, RcppProgress and its Imports;
# install.packages("harmony") pulls them in). Set R_HARMONY_REPO to a local
# clone of immunogenomics/harmony to skip the network clone. Scratch files go
# to $WORK (default: a new temp dir).
set -euo pipefail
cd "$(dirname "$0")"
REPO=$(git rev-parse --show-toplevel)
WORK=${WORK:-$(mktemp -d)}
PR56_COMMIT=${PR56_COMMIT:-edb549c}      # PR #56 as merged (same tree as PR head d2ef71c)
R_HARMONY_COMMIT=${R_HARMONY_COMMIT:-83f3e8a}  # harmony2 branch, linked by the PR
R_HARMONY_REPO=${R_HARMONY_REPO:-https://github.com/immunogenomics/harmony}
echo "Scratch directory: $WORK"

build_harmonypy() {  # label, commit
    local label=$1 commit=$2
    git -C "$REPO" worktree add -q --detach "$WORK/src-$label" "$commit"
    uv venv -q --python 3.12 "$WORK/venv-$label"
    uv pip install -q --python "$WORK/venv-$label/bin/python" \
        -C build-dir="$WORK/build-$label" "$WORK/src-$label" pandas
    git -C "$REPO" worktree remove --force "$WORK/src-$label"
}

build_harmonypy v2.0.2 v2.0.2
build_harmonypy pr56 "$PR56_COMMIT"

# The two-column ridge step calls inv(), which needs LAPACK. The harmony2
# configure script probes for LAPACK by compiling a call to sgesv_ that passes
# ints where pointers are expected; current compilers (Apple clang 17, GCC 14+)
# reject it, so configure disables LAPACK even when it is available, and
# two-column runs fail with "inv(): use of LAPACK must be enabled". Skip
# configure and write src/Makevars directly. CRAN R's bundled LAPACK on macOS
# lacks sgesv_, so link Accelerate there. The Linux branch has not been run.
git clone -q "$R_HARMONY_REPO" "$WORK/r-harmony"
git -C "$WORK/r-harmony" checkout -q "$R_HARMONY_COMMIT"
mkdir -p "$WORK/Rlib"
if [[ $(uname) == Darwin ]]; then
    printf 'PKG_CXXFLAGS = -DARMA_USE_CURRENT\nPKG_LIBS = -framework Accelerate $(FLIBS)\n' \
        > "$WORK/r-harmony/src/Makevars"
else
    printf 'PKG_CXXFLAGS = -DARMA_USE_CURRENT\nPKG_LIBS = $(LAPACK_LIBS) $(BLAS_LIBS) $(FLIBS)\n' \
        > "$WORK/r-harmony/src/Makevars"
fi
R CMD INSTALL --no-configure -l "$WORK/Rlib" "$WORK/r-harmony"

mkdir -p results
uv run --no-project --with numpy --with pandas python make_toy_data.py
for label in v2.0.2 pr56; do
    "$WORK/venv-$label/bin/python" minimal.py | tee "results/minimal_$label.txt"
    "$WORK/venv-$label/bin/python" run_harmonypy.py "$label"
done
R_LIBS="$WORK/Rlib" Rscript run_r_harmony.R
uv run --no-project --with pandas --with matplotlib python plot.py
