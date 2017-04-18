#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


from sklearn import metrics
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist

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

print(sim.head())






