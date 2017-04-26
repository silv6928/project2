#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import random
from collections import OrderedDict

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


# This function performs a lot of the requirements with Phase 1. First it will perform K-means clustering with K
# set to 15. You can change the K assignment to however many clusters you want. Once the algorithm runs it will
# create a list of labels for each cluster, this will take cluster and extract two random ingredients and concatenate
# them to create a label. Once that is done, the function will perform Principe Component Analysis against the
# distance matrix that was created. This will project the data into 2-dimensional space. The first two Principle
# Components will be plotted against each other and colored based on their cluster. Then labels are added to a legend
# for each cluster. And then the visualization is printed.
def perform_clustering(dist):
    K = 15
    # Perform the clustering using K-Means Clustering
    KM = KMeans(n_clusters=K, random_state=1)
    KM.fit(dist)

    # Save the cluster labels into a list and create a dataframe that combines the ingredients with their
    # cluster assignment.
    clusters = KM.labels_.tolist()
    contents = {'ingredients': dist.index.tolist(), 'cluster_num': clusters}
    contents = pd.DataFrame(contents, columns=['ingredients', 'cluster_num'])
    can_labels = []

    # Creates a list of labels for the 15 clusters
    # Selects two random ingredients to join together to create the label.
    for i in range(K):
        c = contents.loc[contents["cluster_num"] == i]
        c = list(c["ingredients"])
        c = random.sample(c, 2)
        c = '-'.join(c)
        can_labels.append(c)

    # Perform Principle component analysis on the data set to better visualize the clusters
    two_dim = PCA(n_components=2)
    two_dim = two_dim.fit_transform(dist)
    first_dim = two_dim[:, 0]
    second_dim = two_dim[:, 1]
    # Combine the PC's into a data frame with their associated cluster
    components = pd.DataFrame(dict(first=first_dim, second=second_dim, labels=clusters))
    labels = np.array(can_labels) # change labels from list to array for matplotlib

    # This will make it easier to map the scatter plots of each cluster
    colors = cm.jet(np.linspace(0, 1, len(labels)))
    color_map = {}
    label_map = {}
    for i in range(len(labels)):
        color_map[i] = colors[i]
        label_map[i] = labels[i]
    # Change size of figure to fit the legend on
    plt.figure(figsize=(10, 10))
    # Plot each cluster with a different label and different color on a 2D graph
    for i in label_map:
        plt.scatter(np.asarray(components["first"].where(components["labels"] == i)),
                    np.asarray(components["second"].where(components["labels"] == i)),
                    color=color_map[i], label=label_map[i])

    # Move the legend to the upper right side outside of the graph
    plt.legend(loc='upper right', fontsize='small', bbox_to_anchor=(1.2, .8))
    # Plot all the titles and show graph
    plt.title("K-Means Clustering with PCA")
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.show()
    return

df = get_ingredients_data()
total = get_list_ingredients(df)
dist = create_dist_matrix(total, df)
perform_clustering(dist)

