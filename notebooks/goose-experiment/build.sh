#!/bin/bash
# Build the C++ and the Goose kernels as shared libraries in build/.
# GOOSE names the Goose compiler (default: goose on PATH). Both are compiled
# with the same C compiler and flags: -O3, as harmonypy's extension is built.
set -eu
cd "$(dirname "$0")"
GOOSE=${GOOSE:-goose}
CFLAGS=${CFLAGS:--O3 -DNDEBUG}
EXT=$([ "$(uname)" = Darwin ] && echo dylib || echo so)
mkdir -p build
"$GOOSE" -o build/kernels.c --header build/kernels.h -O2 kernels.goose
"$GOOSE" --emit-runtime build/goose_runtime.c
cc $CFLAGS -shared -fPIC build/kernels.c build/goose_runtime.c -o build/libgoose_kernels.$EXT -lm -pthread
c++ -std=c++17 $CFLAGS -shared -fPIC kernels.cpp -o build/libcpp_kernels.$EXT
echo "built with: cc $CFLAGS"
