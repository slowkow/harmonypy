"""Write the toy dataset shared by the Python and R runs.

Two well-separated cell types in 2-D. Lab B is shifted by (0.8, -0.8) and
has only 10 of its 100 cells in cell type 2, so its mean assignment to that
cluster is 0.10. Day is balanced within every lab and cell type.

    uv run --no-project --with numpy --with pandas python make_toy_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent

rng = np.random.default_rng(1)
rows = []
for lab, cell_type, n in [("A", 1, 50), ("A", 2, 50), ("B", 1, 90), ("B", 2, 10)]:
    for i in range(n):
        rows.append({"lab": lab, "day": str(1 + i % 2), "cell_type": cell_type})
df = pd.DataFrame(rows)

center = np.where(df[["cell_type"]] == 1, [6.0, 1.0], [1.0, 6.0])
shift = np.where(df[["lab"]] == "B", [0.8, -0.8], [0.0, 0.0])
xy = center + shift + rng.normal(0, 0.3, (len(df), 2))
df["x"], df["y"] = xy[:, 0], xy[:, 1]

(HERE / "data").mkdir(exist_ok=True)
df.to_csv(HERE / "data" / "toy.tsv", sep="\t", index=False, float_format="%.6f")
print(pd.crosstab([df.lab, df.day], df.cell_type))
