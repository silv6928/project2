#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Set the precision to be 7 decimal points
np.set_printoptions(precision=7, suppress=True)
# Read in the CSV data. Note: This csv file is the original file minus the first four rows
df = pd.read_csv("cluster_set.csv", header=0, sep=',', names=['a', 'b', 'dist'])
df.set_index(["a", "b"])

# Create a list of all of the ingredients
a = list(set(df["a"]))
b = list(set(df["b"]))
total = list(sorted(set(a + b)))
# From the list of all the ingredients, create a distance matrix
dist = np.ones((len(total), len(total)))
dist = pd.DataFrame(dist, index=total, columns=total)
# Fill in the distance matrix based on the number of shared compounds between all ingredients
# If the ingredients contain a 1, then the two ingredients do not share an compounds
# The closer the metric is to 0 the 'closer' the items are together
for index, row in df.iterrows():
    dist[row['a']][row['b']] = (1/(1+int(row['dist'])))
for i in range(len(dist)):
    dist.iloc[i][i] = 0

# Perform the clustering using Ward method
clust = linkage(dist.values, 'ward')
# Getting the cluster assignments based on the maximum distance of 20.
clusters = fcluster(clust, 30, criterion='distance')
cluster_id = list(zip(dist.index.tolist(), clusters.tolist()))

# Get the contents of each cluster
c = [[] for i in range(len(set(clusters.tolist())))]
for i in cluster_id:
    c[i[1] - 1].append(i[0])
# Print the cluster contents
for i in c:
    print(i)


# Print the visualization
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



