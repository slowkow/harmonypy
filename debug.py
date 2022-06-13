from pathlib import Path

import numpy as np
import pandas as pd

from harmonypy import run_harmony

data_dir = Path("tests/data")

metadata = pd.read_csv(data_dir / "metadata.csv.gz", index_col="cell_id")

datamat = np.loadtxt(data_dir / "data_mat.csv.gz", delimiter=",")

res = run_harmony(datamat, metadata, ["dataset"])
