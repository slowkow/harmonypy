import pandas as pd
import numpy as np
from scipy.cluster.vq import kmeans

meta_data = pd.read_csv("data/meta.tsv.gz", sep = "\t")
data_mat = pd.read_csv("data/pcs.tsv.gz", sep = "\t")
data_mat = np.array(data_mat)
vars_use = ['dataset']

# meta_data
#
#                  cell_id dataset  nGene  percent_mito cell_type
# 0    half_TGAAATTGGTCTAG    half   3664      0.017722    jurkat
# 1    half_GCGATATGCTGATG    half   3858      0.029228      t293
# 2    half_ATTTCTCTCACTAG    half   4049      0.015966    jurkat
# 3    half_CGTAACGACGAGAG    half   3443      0.020379    jurkat
# 4    half_ACGCCTTGTTTACC    half   2813      0.024774      t293
# ..                   ...     ...    ...           ...       ...
# 295  t293_TTACGTACGACACT    t293   4152      0.033997      t293
# 296  t293_TAGAATTGTTGGTG    t293   3097      0.021769      t293
# 297  t293_CGGATAACACCACA    t293   3157      0.020411      t293
# 298  t293_GGTACTGAGTCGAT    t293   2685      0.027846      t293
# 299  t293_ACGCTGCTTCTTAC    t293   3513      0.021240      t293

# [300 rows x 5 columns]

# data_mat[:5,:5]
#
# array([[ 0.0071695 , -0.00552724, -0.0036281 , -0.00798025,  0.00028931],
#        [-0.011333  ,  0.00022233, -0.00073589, -0.00192452,  0.0032624 ],
#        [ 0.0091214 , -0.00940727, -0.00106816, -0.0042749 , -0.00029096],
#        [ 0.00866286, -0.00514987, -0.0008989 , -0.00821785, -0.00126997],
#        [-0.00953977,  0.00222714, -0.00374373, -0.00028554,  0.00063737]])

import harmonypy as hm
ho = hm.run_harmony(data_mat, meta_data, vars_use)

# Write the adjusted PCs to a new file.
res = pd.DataFrame(ho.Z_corr)
res.columns = ['X{}'.format(i + 1) for i in range(res.shape[1])]
res.to_csv("data/adj.tsv.gz", sep = "\t", index = False)

