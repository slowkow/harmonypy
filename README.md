harmonypy
=========

[![Latest PyPI Version][pb]][pypi] [![PyPI Downloads][db]][pypi] [![tests][gb]][yml]  [![DOI](https://zenodo.org/badge/229105533.svg)](https://zenodo.org/badge/latestdoi/229105533)

[gb]: https://github.com/slowkow/harmonypy/actions/workflows/python-package.yml/badge.svg
[yml]: https://github.com/slowkow/harmonypy/actions/workflows/python-package.yml
[pb]: https://img.shields.io/pypi/v/harmonypy.svg
[pypi]: https://pypi.org/project/harmonypy/

[db]: https://img.shields.io/pypi/dm/harmonypy?label=pypi%20downloads

Harmony is an algorithm for integrating multiple high-dimensional datasets.

harmonypy is a PyTorch-accelerated Python implementation of the [harmony] R package by [Ilya Korsunsky]. It supports GPU acceleration (CUDA, Apple Silicon MPS) and optimized CPU execution.

Example
-------

<p align="center">
  <img src="https://i.imgur.com/lqReopf.gif">
</p>

This animation shows the Harmony alignment of three single-cell RNA-seq datasets from different donors.

[→ How to make this animation.](https://slowkow.com/notes/harmony-animation/)

Installation
------------

Requires Python >= 3.9.

Use [pip] to install:

```bash
pip install harmonypy
```

This will also install PyTorch if not already present. For GPU support, ensure you have the appropriate PyTorch version installed for your system (see [PyTorch installation guide](https://pytorch.org/get-started/locally/)).

Performance
-----------

harmonypy v0.1.0 uses PyTorch for significant performance improvements over the previous NumPy implementation:

| Dataset | Cells | NumPy (v0.0.x) | PyTorch (v0.1.0) | Speedup |
|---------|-------|----------------|------------------|---------|
| Small   | 3.5k  | 1.88s          | 3.66s            | -       |
| Medium  | 69k   | 56.22s         | 9.33s            | 6x      |
| Large   | 858k  | 340s           | 23s (MPS)        | 14x     |

*Benchmarks on Apple M3 Max. GPU acceleration provides the largest speedups for medium to large datasets.*

Usage
-----

Here is a brief example using the data that comes with the R package:

```python
import pandas as pd
import numpy as np
import harmonypy as hm

# Load data
meta_data = pd.read_csv("data/meta.tsv.gz", sep="\t")
data_mat = pd.read_csv("data/pcs.tsv.gz", sep="\t")
data_mat = np.array(data_mat)

# Run Harmony
ho = hm.run_harmony(data_mat, meta_data, ['dataset'])

# Get corrected PCs as NumPy array
corrected_pcs = ho.Z_corr.T  # Transpose to cells x PCs

# Write to file
res = pd.DataFrame(corrected_pcs)
res.columns = ['PC{}'.format(i + 1) for i in range(res.shape[1])]
res.to_csv("data/adj.tsv.gz", sep="\t", index=False)
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

As of v0.1.0, harmonypy implements the same algorithm formulas as the latest [harmony] R package, ensuring consistent results between Python and R implementations. Key updates include:

- Updated clustering objective function
- Dynamic lambda estimation (`lamb=-1`)
- Improved numerical stability

[harmony]: https://github.com/immunogenomics/harmony
[Ilya Korsunsky]: https://github.com/ilyakorsunsky
[pip]: https://pip.readthedocs.io/
