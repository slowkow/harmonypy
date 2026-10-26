# Could Goose speed up harmonypy? No.

[Goose](https://aardappel.github.io/goose/) is a memory-safe systems language
that compiles to C. This folder tests whether writing harmonypy's hot loops in
Goose would make them faster than the C++ they are written in now. It would
not: on an M1 Ultra the Goose code runs at the same speed as the C++, and on
an x86-64 laptop it is 28% slower on the heaviest loop. The experiment was
run on 2026-10-25 with Goose v2026.3.

## What was measured

The two loops that take most of the clustering time in `src/harmony.cpp`,
on the shape of the 858k-cell acute myeloid dataset (28 PCs, 100 clusters),
on one thread:

- **assign**: the distances `2 (1 - Y' z_j)` of every cell to every centroid
  and the softmax of `-dist / sigma` (`Harmony::assign_without_diversity`).
  This is 858k cells × 100 clusters × 28 dimensions of multiply-adds, then
  100 exponentials per cell.
- **reassign**: the inner loop of `Harmony::reassign_block_runs` for one
  block of 5% of the cells: logits from the distances and the diversity
  penalty, a softmax, the sums of the new assignments, and the objective
  terms.

`kernels.cpp` holds the C++ loops as they are in `harmony.cpp`, behind a C
API. `kernels.goose` holds the same loops in Goose, exported to C with
`export fn`. `build.sh` compiles both into shared libraries with the same C
compiler and flags (`-O3`, as the extension is built), and `bench.py` calls
both through ctypes on the same arrays, checks that the outputs agree, and
times them (best of 5).

For scale, the whole `run_harmony` call on this dataset takes 17.9 s on one
thread and 1.9 s on 20 threads of the M1 Ultra.

## Results

Time per call, best of 5, and the ratio Goose / C++:

| Machine, compiler | assign C++ | assign Goose | reassign C++ | reassign Goose |
|---|---:|---:|---:|---:|
| M1 Ultra, Apple clang 17 | 498 ms | 503 ms (1.01×) | 28.1 ms | 28.4 ms (1.01×) |
| Ryzen 5 5560U, gcc 11.4 | 505 ms | 646 ms (1.28×) | 29.3 ms | 29.8 ms (1.02×) |

- The outputs are identical on Linux. On the Mac the distances differ by at
  most 5e-7, because the Goose C turns off fused multiply-adds
  (`fp-contract=off`) and clang uses them in the C++ loop; with
  `-ffp-contract=off` for both, the outputs are identical there too and the
  times are the same.
- `-O2`, `-O3` and `-march=native` give the same picture.
- A first, literal port of the loops (indexing with `k in K`) was 2.0-2.4×
  slower than the C++ on the x86-64 machine with gcc, and level on the Mac
  with clang. Goose keeps a bounds check on every index it cannot prove in
  range, and the check inside the inner loop stopped gcc from vectorizing
  it. Iterating over the slices (`for y, k in y_a`) and asserting that the
  slices have equal lengths let Goose drop every check in the inner loops
  (`goose --bce-lines` shows what remains), which gave the numbers above.
- The 28% that remains on gcc is in the distance loop, `dist[k] += x_a * y`.
  The C++ marks `dist` and `y_a` as `__restrict__`; the C that Goose
  generates has no `restrict`, so gcc must allow for the two slices to
  overlap.

## What we learned

- Goose does what it says: memory-safe code at the speed of C, when the C
  compiler is clang. Its C output is plain enough to compile into a Python
  extension's shared library (a `{data, len}` struct per slice, and one
  `goose_init` call before the first export).
- Nothing in these loops is faster in Goose than in C++. The speed of both
  comes from the C compiler vectorizing a simple loop; Goose adds bounds
  checks that have to be reasoned away first, and withholds the aliasing
  information (`restrict`) that gcc needs. Goose's `simd fn`
  multi-versioning (AVX2 and AVX-512 builds selected at run time) could
  help the x86-64 wheels, which are built for baseline x86-64, but it works
  only with clang, and C++ offers the same through
  `__attribute__((target_clones))`.
- So the experiment stops here, as agreed: there is no gain to pursue.

## Run it again

```bash
# Goose: unpack a release from https://github.com/aardappel/goose/releases
cd notebooks/goose-experiment
GOOSE=/path/to/goose ./build.sh                 # CFLAGS="-O3 -DNDEBUG" by default
uv run --with numpy python bench.py             # --cells, --pcs, --clusters, --repeats
```
