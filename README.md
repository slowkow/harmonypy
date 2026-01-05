harmonypy
=========

[![Latest PyPI Version][badge-pypi]][pypi] [![PyPI Downloads][badge-downloads]][pypi] [![DOI][badge-zenodo]][zenodo]

[badge-pypi]: https://img.shields.io/pypi/v/harmonypy.svg
[pypi]: https://pypi.org/project/harmonypy/
[badge-downloads]: https://img.shields.io/pypi/dm/harmonypy?label=pypi%20downloads
[badge-zenodo]: https://zenodo.org/badge/229105533.svg
[zenodo]: https://zenodo.org/badge/latestdoi/229105533

[Harmony] is an algorithm for integrating multiple high-dimensional datasets.

harmonypy is a PyTorch-accelerated Python implementation that is derived from the [harmony] R package by [Ilya Korsunsky]. It supports GPU acceleration (CUDA, Apple Silicon MPS) and optimized CPU execution.

Example
-------

<p align="center">
  <img src="https://github.com/user-attachments/assets/2cc25f1b-ae25-4ecc-84af-d96f913dd2ef">
</p>

This animation shows the Harmony alignment of three single-cell RNA-seq datasets from different donors. Before Harmony, we can see that cells are separated by donor — this is likely a technical batch effect that we would like to remove before we proceed with clustering analysis. After Harmony, we can see that the cells from different donors are well-mixed while preserving the general shape of the data.

[→ How to make this animation.](https://slowkow.com/notes/harmony-animation/)

Installation
------------

Requires Python >= 3.9.

Use [pip] to install:

```bash
pip install harmonypy
```

This will also install PyTorch if not already present. For GPU support, ensure you have the appropriate PyTorch version installed for your system (see [PyTorch installation guide](https://pytorch.org/get-started/locally/)).

Usage
-----

Here is a brief example using the data that comes with the R package:

```python
import pandas as pd
import numpy as np
import harmonypy as hm

# Load data
meta = pd.read_csv("data/pbmc_3500_meta.tsv.gz", sep="\t")
pcs = pd.read_csv("data/pbmc_3500_pcs.tsv.gz", sep="\t")

# Run Harmony to adjust PCs by meta['donor']
ho = hm.run_harmony(pcs, meta, ['donor'])

# Get corrected PCs as NumPy array
corrected_pcs = ho.Z_corr.T  # Transpose to cells x PCs

# Write to file
res = pd.DataFrame(corrected_pcs)
res.columns = ['PC{}_harmony'.format(i + 1) for i in range(res.shape[1])]
res.to_csv("data/pcs_harmonized.tsv.gz", sep="\t", index=False)
```

It is possible to access all of the internal arrays after running Harmony if you want to inspect the results more closely (see the [Supplement](https://static-content.springer.com/esm/art%3A10.1038%2Fs41592-019-0619-0/MediaObjects/41592_2019_619_MOESM1_ESM.pdf) for more information):

```python
ho = hm.run_harmony(data_mat, meta_data, ['batch'])

# All properties return NumPy arrays with descriptive docstrings:
ho.Z_corr    # Corrected embedding (d x N) - batch effects removed
ho.Z_orig    # Original embedding (d x N) - input data
ho.Z_cos     # L2-normalized embedding (d x N) - used for clustering
ho.R         # Soft cluster assignments (K x N) - P(cell i in cluster k)
ho.Y         # Cluster centroids (d x K) - cluster centers
ho.O         # Observed batch-cluster counts (K x B)
ho.E         # Expected batch-cluster counts (K x B)
ho.Phi       # Batch indicator matrix (B x N) - one-hot encoding
ho.Phi_moe   # Batch indicator with intercept ((B+1) x N)
ho.Pr_b      # Batch proportions (B,)
ho.theta     # Diversity penalty parameters (B,)
ho.sigma     # Clustering bandwidth (K,)
ho.lamb      # Ridge regression penalty ((B+1),)
```

GPU Acceleration
----------------

harmonypy automatically detects and uses the best available device:

- **CUDA** (NVIDIA GPUs)
- **MPS** (Apple Silicon)
- **CPU** (fallback)

To explicitly specify a device:

```python
# Force CPU
ho = hm.run_harmony(data_mat, meta_data, ['batch'], device='cpu')

# Force CUDA
ho = hm.run_harmony(data_mat, meta_data, ['batch'], device='cuda')

# Force Apple Silicon GPU
ho = hm.run_harmony(data_mat, meta_data, ['batch'], device='mps')
```

R Package Compatibility
-----------------------

As of v0.1.0, harmonypy implements the same algorithm formulas as the latest [harmony] R package (v1.2.4), ensuring consistent results between Python and R implementations. Key updates include:

- Updated clustering objective function
- Dynamic lambda estimation (`lamb=-1`)
- Improved numerical stability

[Harmony]: https://www.nature.com/articles/s41592-019-0619-0
[harmony]: https://github.com/immunogenomics/harmony
[Ilya Korsunsky]: https://github.com/ilyakorsunsky
[pip]: https://pip.readthedocs.io/

Citation
--------

If you use harmonypy in your work, please cite the original manuscript describing the algorithm:

- Korsunsky I, Millard N, Fan J, Slowikowski K, Zhang F, Wei K, et al. **Fast, sensitive and accurate integration of single-cell data with Harmony.** *Nat Methods.* 2019. doi:[10.1038/s41592-019-0619-0](https://doi.org/10.1038/s41592-019-0619-0)
 