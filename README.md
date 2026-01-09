# harmonypy

[![PyPI][pb]][pypi] [![Downloads][db]][pypi] [![Tests][gb]][yml] [![DOI][zb]][zen]

[pb]: https://img.shields.io/pypi/v/harmonypy.svg
[pypi]: https://pypi.org/project/harmonypy/
[db]: https://img.shields.io/pypi/dm/harmonypy?label=downloads
[gb]: https://github.com/slowkow/harmonypy/actions/workflows/python-package.yml/badge.svg
[yml]: https://github.com/slowkow/harmonypy/actions/workflows/python-package.yml
[zb]: https://img.shields.io/badge/DOI-10.5281/zenodo.4531400-blue
[zen]: https://doi.org/10.5281/zenodo.4531400

**harmonypy** is a Python implementation of the [Harmony] algorithm for integrating multiple high-dimensional datasets.

<p align="center">
  <img src="https://i.imgur.com/lqReopf.gif">
</p>

This animation shows Harmony aligning three single-cell RNA-seq datasets from different donors. [→ How to make this animation](https://slowkow.com/notes/harmony-animation/). Before Harmony, you can clearly distinguish cells from each of the three donors. After Harmony, the cells from different donors are mixed while preserving the overall shape of the data. This makes it easier to run clustering algorithms to find similar cell types that are present in different batches of data.


## Installation

```bash
pip install harmonypy
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

## Performance

Apple M1 Ultra (2022) with PyTorch MPS backend:

```
  Small (3.5k cells x 30 PCs):    3.48s
  Medium (69k cells x 50 PCs):    9.26s
  Large (858k cells x 29 PCs):    21.75s
```

Note: For small datasets, the NumPy-only version (v0.1.0) may be faster due to GPU overhead.


## Citation

If you use Harmony in your work, please cite the original paper:

> Korsunsky, I., Millard, N., Fan, J. et al. **Fast, sensitive and accurate integration of single-cell data with Harmony.** *Nat Methods* 16, 1289–1296 (2019). https://doi.org/10.1038/s41592-019-0619-0

The [Supplementary Information PDF][supp] provides detailed mathematical descriptions and implementation notes.

[Harmony]: https://github.com/immunogenomics/harmony
[supp]: https://static-content.springer.com/esm/art%3A10.1038%2Fs41592-019-0619-0/MediaObjects/41592_2019_619_MOESM1_ESM.pdf
