#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nltk
import numpy as np
import pandas as pd
import random
import sklearn

# 39774 different Receipes
# 6714 total unique ingredients
# Salt is the most common, followed by onions, and olive oil

# A lot of this code was used from the sentdex videos as provided in Unit 8
# Very helpful videos!

df = pd.read_json("yummly.json")

col_names = ['id', 'cuisine', 'ingredients']
df = df.reindex(columns=col_names)

documents = [(row[3], row[2])
            for row in df.itertuples(index=True, name='Pandas')]


random.shuffle(documents)

all_ingredients = []

for i in df.ingredients:
    all_ingredients.extend(i)

all_ingredients = nltk.FreqDist(all_ingredients)

# Get the 4000 most common ingredients
ingredient_features = [i[0] for i in all_ingredients.most_common()[:1500]]
# Need to add the target value to the feature set for our algorithms to work


# Helper function taken from sentdex videos
def find_features(document):
    ingredients = set(document)
    features = {}
    for i in ingredient_features:
            features[i] = (i in ingredients)
    return features

featuresets = [find_features(row[0])  for row in documents]

featuresets = pd.DataFrame(featuresets)
featuresets = pd.concat([df["cuisine"], featuresets], axis=1)

print(featuresets.head())


