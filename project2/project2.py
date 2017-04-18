#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


np.set_printoptions(precision=7, suppress=True)

df = pd.read_csv("cluster_set.csv", header=0, sep=',', names=['a', 'b', 'dist'])
df.set_index(["a", "b"])

a = list(set(df["a"]))
b = list(set(df["b"]))
total = list(sorted(set(a + b)))
sim = np.ones((len(total), len(total)))
sim = pd.DataFrame(sim, index=total, columns=total)
for index, row in df.iterrows():
    sim[row['a']][row['b']] = (1/(1+int(row['dist'])))
for i in range(len(sim)):
    sim.iloc[i][i] = 0


clust = linkage(sim.values, 'ward')

plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('cluster size')
plt.ylabel('distance')
dendrogram(
    clust,
    truncate_mode='lastp',
    p=20,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
)

plt.show()


clusters = fcluster(clust, 20, criterion='distance')
cluster_id = list(zip(sim.index.tolist(), clusters.tolist()))


c = [[]] * len(set(clusters.tolist()))
for i in cluster_id:
    c[i[1] - 1].append(i[0])


