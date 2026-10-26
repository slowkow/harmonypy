"""Time harmonypy's two hot clustering loops in C++ (kernels.cpp) and in Goose
(kernels.goose), on the same arrays, single-threaded, through ctypes.

usage: build.sh && uv run --with numpy python bench.py [--cells N] [--repeats R]

The arrays follow the 858k-cell acute myeloid dataset's shape by default:
d = 28 PCs, K = 100 clusters. Z is cosine-normalized like Z_cos, and the
centroids are normalized cells, so the distances and logits look like
harmonypy's. Every kernel is checked against the C++ output.
"""
import argparse
import ctypes
import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


class Slice(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("data", ctypes.c_void_p), ("len", ctypes.c_int64)]


def slice_of(a):
    return Slice(a.ctypes.data, a.size)


def load():
    ext = "dylib" if sys.platform == "darwin" else "so"
    cpp = ctypes.CDLL(os.path.join(HERE, "build", f"libcpp_kernels.{ext}"))
    goose = ctypes.CDLL(os.path.join(HERE, "build", f"libgoose_kernels.{ext}"))
    P, I = ctypes.c_void_p, ctypes.c_int64
    cpp.cpp_assign.restype = I
    cpp.cpp_assign.argtypes = [P, P, P, P, P, I, I, I]
    cpp.cpp_reassign.restype = I
    cpp.cpp_reassign.argtypes = [P, P, P, P, I, I, P, P, P, P]
    goose.goose_init.argtypes = [ctypes.c_int, ctypes.c_void_p]
    goose.goose_init(0, None)
    goose.goose_assign.restype = I
    goose.goose_assign.argtypes = [Slice] * 5 + [I, I, I]
    goose.goose_reassign.restype = I
    goose.goose_reassign.argtypes = [Slice] * 4 + [I] + [Slice] * 4
    return cpp, goose


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cells", type=int, default=857_663)
    p.add_argument("--pcs", type=int, default=28)
    p.add_argument("--clusters", type=int, default=100)
    p.add_argument("--repeats", type=int, default=5)
    a = p.parse_args()
    N, d, K = a.cells, a.pcs, a.clusters
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(N, d)).astype(np.float32)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    Y = Z[rng.choice(N, K, replace=False)]  # K x d, unit length
    Y_t = np.ascontiguousarray(Y.T)  # d x K: dimension a at row a, as Armadillo's Y.t() column-major
    sigma = np.full(K, 0.1, dtype=np.float32)
    lead_div = rng.normal(scale=0.05, size=K).astype(np.float32)
    cells = np.arange(N, dtype=np.uint32)
    rng.shuffle(cells)
    cells = cells[: N // 20]  # one block of 5% of the cells, in random order

    cpp, goose = load()
    out = {}
    times = {}

    def run(name, f, *args):
        best = []
        for _ in range(a.repeats):
            start = time.perf_counter()
            r = f(*args)
            best.append(time.perf_counter() - start)
        times[name] = best
        return r

    # Kernel 1: distances and the softmax without diversity.
    for lang in ("cpp", "goose"):
        dist = np.empty((N, K), dtype=np.float32)
        R = np.empty((N, K), dtype=np.float32)
        if lang == "cpp":
            bad = run("assign cpp", cpp.cpp_assign, Z.ctypes.data, Y_t.ctypes.data, sigma.ctypes.data,
                      dist.ctypes.data, R.ctypes.data, N, d, K)
        else:
            bad = run("assign goose", goose.goose_assign, slice_of(Z), slice_of(Y_t), slice_of(sigma),
                      slice_of(dist), slice_of(R), N, d, K)
        assert bad == 0, bad
        out[lang] = (dist, R)
    for i, what in ((0, "distances"), (1, "assignments")):
        c, g = out["cpp"][i], out["goose"][i]
        diff = np.abs(c - g).max()
        same = np.array_equal(c, g)
        print(f"{what}: {'identical' if same else f'max |C++ - Goose| = {diff:.2e}'}")
        assert diff < 1e-5, what
    dist = out["cpp"][0]

    # Kernel 2: reassigning one block's cells with the diversity penalty.
    res = {}
    for lang in ("cpp", "goose"):
        R = out["cpp"][1].copy()
        sums = np.empty(K, dtype=np.float32)
        err, ent = np.empty(1), np.empty(1)
        if lang == "cpp":
            flags = run("reassign cpp", cpp.cpp_reassign, dist.ctypes.data, sigma.ctypes.data, lead_div.ctypes.data,
                        cells.ctypes.data, len(cells), K, R.ctypes.data, sums.ctypes.data, err.ctypes.data,
                        ent.ctypes.data)
        else:
            flags = run("reassign goose", goose.goose_reassign, slice_of(dist), slice_of(sigma), slice_of(lead_div),
                        slice_of(cells), K, slice_of(R), slice_of(sums), slice_of(err), slice_of(ent))
        assert flags == 0, flags
        res[lang] = (R, sums.copy(), err[0], ent[0])
    c, g = res["cpp"], res["goose"]
    same = np.array_equal(c[0], g[0]) and np.array_equal(c[1], g[1]) and c[2] == g[2] and c[3] == g[3]
    print("reassign: " + ("identical" if same else
          f"max |C++ - Goose| = {np.abs(c[0] - g[0]).max():.2e} (R), {np.abs(c[1] - g[1]).max():.2e} (sums), "
          f"{abs(c[2] - g[2]):.2e} (error), {abs(c[3] - g[3]):.2e} (entropy)"))
    assert np.abs(c[0] - g[0]).max() < 1e-5

    print(f"N={N:,} d={d} K={K}; one thread; best / median of {a.repeats}")
    for name, t in times.items():
        print(f"  {name:16s} {min(t) * 1e3:8.1f} ms / {statistics.median(t) * 1e3:8.1f} ms")
    for kernel in ("assign", "reassign"):
        c, g = min(times[f"{kernel} cpp"]), min(times[f"{kernel} goose"])
        print(f"  {kernel}: Goose / C++ = {g / c:.2f}")


if __name__ == "__main__":
    main()
