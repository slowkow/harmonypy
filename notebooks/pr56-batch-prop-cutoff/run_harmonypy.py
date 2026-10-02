"""Run the installed harmonypy on the toy and ircolitis examples.

    python run_harmonypy.py LABEL

LABEL names the build being tested (for example "v2.0.2" or "pr56").
Results go to results/. run_all.sh calls this once per build.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import harmonypy as hm

HERE = Path(__file__).parent
REPO_DATA = HERE.parents[1] / "data"
RESULTS = HERE / "results"

TOY_CUTOFFS = np.round(np.arange(0, 0.3001, 0.005), 3)
TOY_SCATTER_CUTOFF = 0.15
REAL_CUTOFFS = [1e-5, 1e-2]


def lab_gap(xy, toy, cell_type):
    """Distance between the lab A and lab B centroids within one cell type."""
    in_type = toy.cell_type.values == cell_type
    a = xy[in_type & (toy.lab.values == "A")].mean(axis=0)
    b = xy[in_type & (toy.lab.values == "B")].mean(axis=0)
    return float(np.linalg.norm(a - b))


def run_toy(label):
    toy = pd.read_csv(HERE / "data" / "toy.tsv", sep="\t", dtype={"day": str})
    xy = toy[["x", "y"]].values
    rows = []
    for cutoff in TOY_CUTOFFS:
        res = hm.run_harmony(
            xy, toy, ["lab", "day"], nclust=2, theta=0, lamb=1,
            batch_prop_cutoff=cutoff, ncores=1, verbose=False,
        )
        rows.append({
            "cutoff": cutoff,
            "gap_type1": lab_gap(res.Z_corr, toy, 1),
            "gap_type2": lab_gap(res.Z_corr, toy, 2),
        })
        if np.isclose(cutoff, TOY_SCATTER_CUTOFF):
            pd.DataFrame(res.Z_corr, columns=["x", "y"]).to_csv(
                RESULTS / f"toy_corrected_{label}.tsv", sep="\t",
                index=False, float_format="%.6f",
            )
    pd.DataFrame(rows).to_csv(
        RESULTS / f"toy_sweep_{label}.tsv", sep="\t", index=False,
        float_format="%.6f",
    )


def run_ircolitis(label):
    obs = pd.read_csv(
        REPO_DATA / "ircolitis_blood_cd8_obs.tsv.gz", sep="\t",
        usecols=["donor", "batch"],
    )
    pcs = pd.read_csv(REPO_DATA / "ircolitis_blood_cd8_pcs.tsv.gz", sep="\t")
    pcs = pcs.filter(regex=r"^PC\d+$")
    rows = []
    for cutoff in REAL_CUTOFFS:
        res = hm.run_harmony(
            pcs.values, obs, ["donor", "batch"],
            batch_prop_cutoff=cutoff, verbose=False,
        )
        pd.DataFrame(res.Z_corr, columns=pcs.columns).to_csv(
            RESULTS / f"ircolitis_{label}_cutoff{cutoff:g}.tsv.gz",
            sep="\t", index=False, float_format="%.6g",
        )
        # Mean assignment of each (cluster, level) pair, using final R.
        # R harmony compares this value to the cutoff. With two correction
        # columns, harmonypy 2.0.2 compared twice this value instead.
        for col in ["donor", "batch"]:
            codes = pd.Categorical(obs[col]).codes
            counts = np.bincount(codes)
            sums = np.zeros((counts.size, res.R.shape[1]))
            np.add.at(sums, codes, res.R)
            mean_r = sums / counts[:, None]
            rows.append({
                "cutoff": cutoff,
                "column": col,
                "pairs": mean_r.size,
                "above_cutoff": int((mean_r > cutoff).sum()),
                "included_by_v2_0_2_only": int(
                    ((mean_r > cutoff / 2) & (mean_r <= cutoff)).sum()
                ),
            })
    pd.DataFrame(rows).to_csv(
        RESULTS / f"ircolitis_levels_{label}.tsv", sep="\t", index=False,
    )


if __name__ == "__main__":
    label = sys.argv[1]
    RESULTS.mkdir(exist_ok=True)
    print(f"harmonypy {hm.__version__} from {hm.__file__} as '{label}'")
    run_toy(label)
    run_ircolitis(label)
