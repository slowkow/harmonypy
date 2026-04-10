# harmonypy

[![PyPI][pb]][pypi] [![Downloads][db]][pypi] [![Tests][gb]][yml] [![DOI][zb]][zen]

[pb]: https://img.shields.io/pypi/v/harmonypy.svg
[pypi]: https://pypi.org/project/harmonypy/
[db]: https://img.shields.io/pypi/dm/harmonypy?label=downloads
[gb]: https://github.com/slowkow/harmonypy/actions/workflows/python-package.yml/badge.svg
[yml]: https://github.com/slowkow/harmonypy/actions/workflows/python-package.yml
[zb]: https://img.shields.io/badge/DOI-10.5281/zenodo.4531400-blue
[zen]: https://doi.org/10.5281/zenodo.4531400

**harmonypy** is a Python package for the [Harmony] algorithm for integrating multiple high-dimensional datasets. It uses a C++ backend (Armadillo) for fast linear algebra, matching the [R harmony2 package][Harmony] step-by-step.

<p align="center">
  <img src="https://github.com/user-attachments/assets/018f82a7-ebb2-47a7-a340-dc9427c51b50">
</p>

This animation shows Harmony aligning three single-cell RNA-seq datasets from different donors. [→ How to make this animation](https://slowkow.com/notes/harmony-animation/). Before Harmony, you can clearly distinguish cells from each of the three donors. After Harmony, the cells from different donors are mixed while preserving the overall shape of the data.


## Installation

Install from PyPI (pre-built wheels for Linux and macOS):

```bash
pip install harmonypy
```

### Building from source

Building from source requires a C++ compiler, CMake, and a BLAS library:

**macOS** (uses Apple Accelerate, no extra dependencies):

```bash
pip install .
```

**Linux** (requires OpenBLAS):

```bash
# Debian/Ubuntu
sudo apt install libopenblas-dev cmake

# RHEL/Fedora
sudo dnf install openblas-devel cmake

pip install .
```


## Quick Start

```python
import harmonypy as hm
import pandas as pd

# Load the principal components and metadata
pcs = pd.read_csv("data/pbmc_3500_pcs.tsv.gz", sep="\t")
meta = pd.read_csv("data/pbmc_3500_meta.tsv.gz", sep="\t")

# Run Harmony to correct for batch effects (donor)
harmony_out = hm.run_harmony(pcs, meta, "donor")

# Save corrected PCs (same shape as input)
result = pd.DataFrame(harmony_out.Z_corr, columns=pcs.columns)
result.to_csv("pbmc_3500_pcs_harmony.tsv", sep="\t", index=False)
```


## Usage with Scanpy

```python
import scanpy as sc
import harmonypy as hm

# Load and preprocess your data
adata = sc.read_h5ad("my_data.h5ad")
sc.pp.pca(adata)

# Get PCs from the AnnData object
pcs = adata.obsm['X_pca']
print(pcs.shape)  # (n_cells, n_pcs)

# Run Harmony on the PCA embedding
harmony_out = hm.run_harmony(pcs, adata.obs, "batch")

# Store corrected PCs back in the AnnData object
adata.obsm['X_pca_harmony'] = harmony_out.Z_corr

# Use harmonized PCs for downstream analysis
sc.pp.neighbors(adata, use_rep='X_pca_harmony')
sc.tl.umap(adata)
sc.tl.leiden(adata)
```


## Parameters

`run_harmony` accepts the same parameters as the R package:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `theta` | 2 | Diversity penalty per batch variable |
| `sigma` | 0.1 | Kernel bandwidth for soft clustering |
| `nclust` | min(N/30, 100) | Number of clusters |
| `max_iter_harmony` | 10 | Maximum Harmony iterations |
| `max_iter_kmeans` | 4 | K-means iterations per Harmony round |
| `epsilon_harmony` | 1e-2 | Convergence threshold |
| `ncores` | 1 | BLAS/OpenMP threads (Linux only) |
| `lamb` | None | Ridge penalty (None = auto-estimate) |

On Linux, `ncores > 1` enables parallel BLAS via OpenBLAS for matrix operations. On macOS, `ncores` has no effect (Accelerate does not support OpenMP).


## Performance

The script in `tests/test_harmony.py` on an Apple M1 (2022) chip reports:

```
  Dataset                    Time    RSS delta
  ---------------------- -------- ------------
  Small (3.5k cells)        0.74s     66.3 MB
  Medium (69k cells)       20.15s    321.6 MB
  Large (858k cells)       38.38s   2519.1 MB
```


## Citation

If you use Harmony in your work, please cite the original paper:

> Korsunsky, I., Millard, N., Fan, J. et al. **Fast, sensitive and accurate integration of single-cell data with Harmony.** *Nat Methods* 16, 1289–1296 (2019). https://doi.org/10.1038/s41592-019-0619-0

The [Supplementary Information PDF][supp] provides detailed mathematical descriptions and implementation notes.

To learn more about Harmony 2, please see the preprint here:

> Patikas, Nikolaos, Hongcheng Yao, Roopa Madhu, Soumya Raychaudhuri, Martin Hemberg, and Ilya Korsunsky. 2026. **Integration of Large, Complex Single-Cell Datasets with Harmony2.** *bioRxiv*. https://doi.org/10.64898/2026.03.16.711825

[Harmony]: https://github.com/immunogenomics/harmony
[supp]: https://static-content.springer.com/esm/art%3A10.1038%2Fs41592-019-0619-0/MediaObjects/41592_2019_619_MOESM1_ESM.pdf
