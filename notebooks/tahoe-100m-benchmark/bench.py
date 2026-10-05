#!/usr/bin/env python
"""Benchmark one harmonypy build on one Tahoe-100M dataset (tahoe-ilya).

Each call runs Harmony once, in its own process, and appends one JSON line to
the results file. The timed part is what a user waits for: run_harmony and
reading the corrected coordinates (h.Z_corr). Loading the data is timed
separately. run.sh calls this for every configuration; plot.py draws the
figure from the results.

usage: bench.py --build NAME --dataset NAME [--sweep NAME] [--ncores N]
                [--kwargs JSON] [--save PATH] [--results results.jsonl]

--dataset is "full" (all 95.6M cells) or a subsample: 1M, 2M, 4M, 8M, 16M
(800 samples each) or 50B ... 800B (1M cells with that many samples).
--kwargs holds extra run_harmony arguments as JSON, for example
'{"max_iter_harmony": 10, "epsilon_harmony": -1}'.
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import platform
import resource
import socket
import sys
import time
import traceback

# The Tahoe-100M PCs and metadata (see README.md), or the TAHOE_DATA variable.
DATA = os.environ.get("TAHOE_DATA", "/projects/home/ks38/work/github.com/slowkow/harmonypy/data/tahoe-ilya")
BATCH_VAR = "sample"


def status_bytes(field):
    """A memory field of /proc/self/status (VmRSS, VmHWM) in bytes."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024
    raise KeyError(field)


def reset_peak_memory():
    """Set the peak resident memory (VmHWM) to the current value (Linux >= 4.0)."""
    with open("/proc/self/clear_refs", "w") as f:
        f.write("5")


def machine_busy_seconds():
    """CPU time used by every process on the machine, from /proc/stat."""
    with open("/proc/stat") as f:
        user, nice, system, idle, iowait, irq, softirq, steal = map(int, f.readline().split()[1:9])
    return (user + nice + system + irq + softirq + steal) / os.sysconf("SC_CLK_TCK")


def cpu_ranges(cpus):
    """[0, 1, 2, 5] -> '0-2,5'."""
    cpus, out, start = sorted(cpus), [], None
    for i, c in enumerate(cpus):
        if start is None:
            start = c
        if i + 1 == len(cpus) or cpus[i + 1] != c + 1:
            out.append(f"{start}-{c}" if c != start else f"{c}")
            start = None
    return ",".join(out)


def read_text(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def harmonypy_libraries():
    """Shared libraries of the harmonypy package mapped into this process.

    The PyPI 2.0.2 wheel bundles OpenBLAS and libgfortran in harmonypy.libs;
    main bundles none. NumPy's own OpenBLAS is loaded in both, so libraries
    outside the harmonypy package are not listed.
    """
    names = set()
    with open("/proc/self/maps") as f:
        for line in f:
            path = line.split()[-1]
            if ("/site-packages/harmonypy/" in path or "/site-packages/harmonypy.libs/" in path) and ".so" in path:
                names.add(os.path.basename(path))
    return sorted(names)


def extension_sha256(package_dir):
    """First 16 hex digits of the sha256 of the compiled extension."""
    paths = glob.glob(os.path.join(package_dir, "_harmony_cpp*.so"))
    if not paths:
        return None
    with open(paths[0], "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def load(dataset):
    import h5py
    import numpy as np
    import pandas as pd

    if dataset == "full":
        pca_path, meta_path = f"{DATA}/pca.h5", f"{DATA}/meta-aligned.parquet"
    else:
        pca_path = f"{DATA}/subsample/pca-{dataset}.h5"
        meta_path = f"{DATA}/subsample/meta-{dataset}.parquet"
    meta = pd.read_parquet(meta_path, columns=[BATCH_VAR])
    with h5py.File(pca_path, "r") as f:
        # Read as float32, as scanpy stores PCs; the subsamples are stored as float64.
        pca = f["pca"].astype(np.float32)[:]
    return pca, meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--build", required=True, help="label of the harmonypy build, e.g. 2.0.2 or main-623ff51")
    p.add_argument("--dataset", required=True)
    p.add_argument("--sweep", default="", help="the experiment the run belongs to (samples, cells, threads, ...)")
    p.add_argument("--ncores", type=int, default=None, help="passed to run_harmony; default: its own default")
    p.add_argument("--kwargs", default="{}", help="extra run_harmony arguments, as JSON")
    p.add_argument("--save", default=None, help="save the corrected coordinates (float32 .npy) here")
    p.add_argument("--results", default="results.jsonl")
    a = p.parse_args()

    available = sorted(os.sched_getaffinity(0))
    record = {
        "build": a.build,
        "dataset": a.dataset,
        "sweep": a.sweep,
        "ncores": a.ncores,
        "kwargs": json.loads(a.kwargs),
        "available_cpus": len(available),
        "cpus": cpu_ranges(available),
        "run_id": os.environ.get("BENCH_RUN_ID"),
        "host": socket.gethostname(),
        "kernel": platform.release(),
        "numa_balancing": read_text("/proc/sys/kernel/numa_balancing"),
        "governor": read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "python": platform.python_version(),
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    try:
        t_load = time.perf_counter()
        pca, meta = load(a.dataset)
        record["load_seconds"] = round(time.perf_counter() - t_load, 2)

        import numpy as np
        import pandas as pd
        import harmonypy as hm
        import harmonypy.harmony as hh

        package_dir = os.path.dirname(hm.__file__)
        record.update(
            harmonypy_version=hm.__version__,
            harmonypy_path=package_dir,
            extension_sha256=extension_sha256(package_dir),
            harmonypy_libraries=harmonypy_libraries(),
            numpy=np.__version__,
            pandas=pd.__version__,
            n_cells=int(pca.shape[0]),
            n_pcs=int(pca.shape[1]),
            n_batches=int(meta[BATCH_VAR].nunique()),
        )

        # Time the C++ part (the HarmonyCpp constructor runs the algorithm) to
        # separate it from run_harmony's Python preprocessing. Both builds look
        # HarmonyCpp up in harmonypy.harmony when run_harmony is called.
        cpp_seconds = []
        cpp_class = hh.HarmonyCpp

        def timed_cpp(*args, **kwargs):
            start = time.perf_counter()
            result = cpp_class(*args, **kwargs)
            cpp_seconds.append(time.perf_counter() - start)
            return result

        hh.HarmonyCpp = timed_cpp

        kwargs = dict(record["kwargs"])
        if a.ncores is not None:
            kwargs["ncores"] = a.ncores
        rss_loaded = status_bytes("VmRSS")
        reset_peak_memory()
        load_before = os.getloadavg()
        busy_before = machine_busy_seconds()
        cpu_before = resource.getrusage(resource.RUSAGE_SELF)
        start = time.perf_counter()
        h = hm.run_harmony(pca, meta, BATCH_VAR, verbose=False, **kwargs)
        ran = time.perf_counter()
        z = h.Z_corr  # every user reads the corrected coordinates
        end = time.perf_counter()
        cpu_after = resource.getrusage(resource.RUSAGE_SELF)
        busy_after = machine_busy_seconds()
        peak = status_bytes("VmHWM")
        rss_after = status_bytes("VmRSS")
        cpu = (cpu_after.ru_utime - cpu_before.ru_utime) + (cpu_after.ru_stime - cpu_before.ru_stime)
        wall = end - start

        record.update(
            wall_seconds=round(wall, 3),
            run_harmony_seconds=round(ran - start, 3),
            extract_seconds=round(end - ran, 3),
            cpp_seconds=round(sum(cpp_seconds), 3) if cpp_seconds else None,
            cpu_seconds=round(cpu, 3),
            # CPU time of everything else on the machine while this ran: other
            # users' jobs that compete with ours for cores.
            others_cpu_seconds=round(max(0.0, busy_after - busy_before - cpu), 1),
            iterations=len(h.objective_harmony) - 1,
            rss_after_loading_gb=round(rss_loaded / 1e9, 3),
            peak_rss_gb=round(peak / 1e9, 3),
            rss_after_run_gb=round(rss_after / 1e9, 3),
            loadavg_before=[round(x, 2) for x in load_before],
            finished=datetime.datetime.now().isoformat(timespec="seconds"),
        )
        if a.save:
            np.save(a.save, np.asarray(z, dtype=np.float32))
    except Exception as e:  # recorded, so plot.py can show the failure
        record.update(failed=True, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:])

    with open(a.results, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)
    if record.get("failed"):
        sys.exit(1)


if __name__ == "__main__":
    main()
