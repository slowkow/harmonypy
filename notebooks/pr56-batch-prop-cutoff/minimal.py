"""Smallest reproducer for PR #56: four cells, two correction columns.

Every cell has the same direction after cosine normalization, so each cell is
assigned 1/2 to each of the two clusters. Each lab and each day then has a
mean assignment of exactly 0.5 in each cluster. With batch_prop_cutoff=0.5,
no level is above the cutoff, so Harmony should leave the coordinates alone.

    uv run python notebooks/pr56-batch-prop-cutoff/minimal.py

harmonypy 2.0.2 prints moved coordinates; with PR #56 they are unchanged.
(R harmony refuses to run on fewer than 6 cells, so this one is Python only.)
"""
import numpy as np
import harmonypy as hm

coords = np.array([[1.0], [2.0], [3.0], [4.0]])
meta = {"lab": ["a", "a", "b", "b"], "day": ["1", "2", "1", "2"]}

res = hm.run_harmony(
    coords, meta, ["lab", "day"], nclust=2, theta=0, lamb=1,
    max_iter_harmony=1, max_iter_kmeans=1, batch_prop_cutoff=0.5,
    ncores=1, verbose=False,
)

print("assignments R:", res.R.ravel())
print("input:        ", coords.ravel())
print("output:       ", res.Z_corr.ravel())
print("unchanged:    ", np.array_equal(res.Z_corr, coords))
