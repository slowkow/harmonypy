#!/bin/bash
# Test the CI pipeline locally using Docker.
#
# This simulates the GitHub Actions Ubuntu environment to catch issues
# (missing dependencies, import errors, build failures) before pushing.
#
# Usage:
#   bash scripts/test_ci_local.sh          # test with Python 3.12 (default)
#   bash scripts/test_ci_local.sh 3.9      # test with Python 3.9
#   bash scripts/test_ci_local.sh all      # test all Python versions

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_VERSIONS=("3.9" "3.10" "3.11" "3.12" "3.13")

test_version() {
    local pyver="$1"
    local image="python:${pyver}-slim"

    echo ""
    echo "============================================================"
    echo "Testing Python ${pyver} (${image})"
    echo "============================================================"

    docker run --rm \
        -v "${REPO_DIR}:/work" \
        -w /work \
        "${image}" \
        bash -c "
            set -e
            echo '--- Installing system dependencies ---'
            apt-get update -qq && apt-get install -y -qq build-essential cmake libopenblas-dev > /dev/null 2>&1

            echo '--- Building and installing harmonypy ---'
            pip install --upgrade pip -q
            pip install -e '.[test]' -q

            echo '--- Running tests ---'
            pytest tests/ -v

            echo '--- Python ${pyver}: ALL TESTS PASSED ---'
        "
}

if [ "${1:-3.12}" = "all" ]; then
    for ver in "${PYTHON_VERSIONS[@]}"; do
        test_version "$ver"
    done
    echo ""
    echo "============================================================"
    echo "All Python versions passed!"
    echo "============================================================"
else
    test_version "${1:-3.12}"
fi
