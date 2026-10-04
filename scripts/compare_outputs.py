#!/usr/bin/env python
"""Compare harmonypy results and run times between two builds.

Run the same configurations with each build, then compare the saved outputs:

    git checkout master && uv pip install -e . --reinstall-package harmonypy
    uv run python scripts/compare_outputs.py run --out /tmp/master.npz

    git checkout cpp-speedup && uv pip install -e . --reinstall-package harmonypy
    uv run python scripts/compare_outputs.py run --out /tmp/branch.npz

    uv run python scripts/compare_outputs.py compare /tmp/master.npz /tmp/branch.npz

The configurations use the datasets in data/ that are present. Add --large to
include the 858k-cell acute myeloid dataset (several minutes per run with
harmonypy 2.0.2). Requires pandas.
"""

import argparse
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")

# (name, dataset, covariates, keyword arguments)
CONFIGS = [
    ("pbmc", "pbmc", ["donor"], {}),
    ("pbmc_lamb1", "pbmc", ["donor"], {"lamb": 1.0}),
    ("pbmc_cutoff", "pbmc", ["donor"], {"batch_prop_cutoff": 0.05}),
    ("ircolitis", "ircolitis", ["batch"], {}),
    ("ircolitis_2cov", "ircolitis", ["batch", "donor"], {}),
]
LARGE_CONFIGS = [
    ("aml", "acute_myeloid", ["batch"], {}),
    ("aml_2cov", "acute_myeloid", ["batch", "sample_id"], {}),
]

DATASETS = {
    "pbmc": ("pbmc_3500_meta.tsv.gz", "pbmc_3500_pcs.tsv.gz", "pbmc_3500_pcs_harmony2.tsv.gz"),
    "ircolitis": ("ircolitis_blood_cd8_obs.tsv.gz", "ircolitis_blood_cd8_pcs.tsv.gz",
                  "ircolitis_blood_cd8_pcs_harmony2.tsv.gz"),
    "acute_myeloid": ("acute_myeloid_obs.tsv.gz", "acute_myeloid_pcs.tsv.gz",
                      "acute_myeloid_pcs_harmony2.tsv.gz"),
}


def load(dataset):
    """Return (PCs as cells x PCs, metadata, R harmony2 reference or None)."""
    import pandas as pd
    meta_file, pcs_file, ref_file = DATASETS[dataset]
    meta_path = os.path.join(DATA, meta_file)
    pcs_path = os.path.join(DATA, pcs_file)
    if not (os.path.exists(meta_path) and os.path.exists(pcs_path)):
        return None
    meta = pd.read_csv(meta_path, sep="\t", low_memory=False)
    pcs = pd.read_csv(pcs_path, sep="\t")
    pc_columns = [c for c in pcs.columns if str(c).startswith("PC")]
    ref_path = os.path.join(DATA, ref_file)
    ref = None
    if os.path.exists(ref_path):
        ref_table = pd.read_csv(ref_path, sep="\t")
        ref = ref_table[[c for c in ref_table.columns if str(c).startswith("PC")]].to_numpy()
    return pcs[pc_columns].to_numpy(dtype=np.float64), meta, ref


def run(args):
    import logging
    import harmonypy as hm
    logging.getLogger("harmonypy").setLevel(logging.WARNING)
    configs = CONFIGS + (LARGE_CONFIGS if args.large else [])
    out = {"version": np.array(hm.__version__)}
    cache = {}
    for name, dataset, covariates, kwargs in configs:
        if dataset not in cache:
            cache[dataset] = load(dataset)
        if cache[dataset] is None:
            print(f"{name:16s} skipped (data not found)")
            continue
        X, meta, _ = cache[dataset]
        start = time.time()
        result = hm.run_harmony(X, meta, covariates, verbose=False, **kwargs)
        elapsed = time.time() - start
        out[f"{name}/Z"] = result.Z_corr.astype(np.float32)
        out[f"{name}/objective"] = np.array(result.objective_harmony)
        out[f"{name}/rounds"] = np.array(result.kmeans_rounds)
        out[f"{name}/seconds"] = np.array(elapsed)
        out[f"{name}/dataset"] = np.array(dataset)
        # The R references were made with one covariate and default settings.
        out[f"{name}/has_reference"] = np.array(len(covariates) == 1 and not kwargs)
        print(f"{name:16s} {elapsed:8.2f} s  {len(result.objective_harmony) - 1} iterations")
    np.savez(args.out, **out)


def per_pc_correlation(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))


def compare(args):
    old, new = np.load(args.old), np.load(args.new)
    names = sorted({key.split("/")[0] for key in old.files if "/" in key} &
                   {key.split("/")[0] for key in new.files if "/" in key},
                   key=lambda n: [c[0] for c in CONFIGS + LARGE_CONFIGS].index(n))
    print(f"old: harmonypy {old['version']}   new: harmonypy {new['version']}")
    print(f"{'config':16s} {'iterations':>10s} {'max |dZ|':>10s} {'min PC r':>11s} "
          f"{'R ref r old':>12s} {'R ref r new':>12s} {'seconds old':>12s} {'seconds new':>12s}")
    refs = {}
    for name in names:
        Z_old = old[f"{name}/Z"].astype(np.float64)
        Z_new = new[f"{name}/Z"].astype(np.float64)
        dataset = str(old[f"{name}/dataset"])
        if dataset not in refs:
            loaded = load(dataset)
            refs[dataset] = loaded[2] if loaded else None
        ref = refs[dataset]
        ref_old = ref_new = "-"
        if ref is not None and bool(old[f"{name}/has_reference"]) and ref.shape == Z_old.shape:
            ref_old = f"{per_pc_correlation(Z_old, ref).min():.4f}"
            ref_new = f"{per_pc_correlation(Z_new, ref).min():.4f}"
        iterations = f"{len(old[f'{name}/objective']) - 1}/{len(new[f'{name}/objective']) - 1}"
        print(f"{name:16s} {iterations:>10s} {np.abs(Z_old - Z_new).max():10.2e} "
              f"{per_pc_correlation(Z_old, Z_new).min():11.7f} {ref_old:>12s} {ref_new:>12s} "
              f"{float(old[f'{name}/seconds']):12.2f} {float(new[f'{name}/seconds']):12.2f}")
    print("R ref r: minimum per-PC correlation with the R harmony2 reference (default settings, one covariate).")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="run every configuration and save the outputs")
    run_parser.add_argument("--out", required=True, help="output .npz file")
    run_parser.add_argument("--large", action="store_true", help="include the 858k-cell dataset")
    compare_parser = sub.add_parser("compare", help="compare two saved outputs")
    compare_parser.add_argument("old")
    compare_parser.add_argument("new")
    args = parser.parse_args()
    {"run": run, "compare": compare}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
