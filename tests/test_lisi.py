import harmonypy as hm
import pandas as pd

X = pd.read_csv("data/lisi_x.tsv.gz", sep = "\t")
metadata = pd.read_csv("data/lisi_metadata.tsv.gz", sep = "\t")
label_colnames = metadata.columns
perplexity = 30

lisi = hm.compute_lisi(X, metadata, label_colnames, perplexity)

lisi = pd.DataFrame(lisi)
lisi.columns = label_colnames
lisi.to_csv("data/lisi_lisi.tsv.gz", sep = "\t")

def timereps(reps, func):
    from time import time
    start = time()
    for i in range(0, reps):
        func()
    end = time()
    return (end - start) / reps

# 0.3 seconds per loop (too slow)
timereps(100, lambda: compute_lisi(X, metadata, label_colnames, n_neighbors))
