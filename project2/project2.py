#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import random

# Set the precision to be 7 decimal points
np.set_printoptions(precision=7, suppress=True)


def get_ingredients_data():
    # Read in the CSV data. Note: This csv file is the original file minus the first four rows
    # I changed the name to cluster_set.csv
    df = pd.read_csv("cluster_set.csv", header=0, sep=',', names=['a', 'b', 'dist'])
    df.set_index(["a", "b"])
    return df


# Returns the list of all the ingredients in the data set
def get_list_ingredients(df):
    # Create a list of all of the ingredients
    a = list(set(df["a"]))
    b = list(set(df["b"]))
    total = list(sorted(set(a + b)))
    return total


# Create a distance matrix of all ingredients
# This takes the number of compounds shared and transforms it into a usable distance measure
def create_dist_matrix(total, df):
    # From the list of all the ingredients, create a distance matrix
    dist = np.ones((len(total), len(total)))
    dist = pd.DataFrame(dist, index=total, columns=total)
    # Fill in the distance matrix based on the number of shared compounds between all ingredients
    # If the ingredients contain a 1, then the two ingredients do not share an compounds
    # The closer the metric is to 0 the 'closer' the items are together
    for index, row in df.iterrows():
        dist[row['a']][row['b']] = (1/(1+int(row['dist'])))
        dist[row['b']][row['a']] = (1/(1+int(row['dist'])))
    for i in range(len(dist)):
        dist.iloc[i][i] = 0
    return dist


def perform_clustering(dist):
    # Perform the clustering using K-Means Clustering
    KM = KMeans(n_clusters=15, random_state=1)
    KM.fit(dist)
    clusters = KM.labels_.tolist()
    contents = {'ingredients': dist.index.tolist(), 'cluster_num': clusters}
    contents = pd.DataFrame(contents, columns=['ingredients', 'cluster_num'])
    can_labels = []
    for i in range(15):
        c = contents.loc[contents["cluster_num"] == i]
        c = list(c["ingredients"])
        c = random.sample(c, 2)
        print(c)
    two_dim = PCA(n_components=2)
    two_dim = two_dim.fit_transform(dist)
    first_dim = two_dim[:, 0]
    second_dim = two_dim[:, 1]
    components = pd.DataFrame(dict(first=first_dim, second=second_dim, labels=clusters))

    x = np.array(components["first"])
    y = np.array(components["second"])
    t = np.array(components["labels"])
    plt.scatter(x, y, c=t)
    plt.title("K-Means Clustering")
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.show()
    return

df = get_ingredients_data()
total = get_list_ingredients(df)
dist = create_dist_matrix(total, df)
perform_clustering(dist)

