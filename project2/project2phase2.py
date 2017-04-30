#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nltk
import numpy as np
import pandas as pd
import random
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import MultinomialNB as NB
from sklearn.metrics import classification_report, accuracy_score
from sklearn import svm
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

# Get the 1500 most common ingredients
ingredient_features = [i[0] for i in all_ingredients.most_common()[:2200]]
# Need to add the target value to the feature set for our algorithms to work


# Helper function taken from sentdex videos in unit 8
def find_features(document):
    ingredients = set(document)
    features = {}
    for i in ingredient_features:
            features[i] = (i in ingredients)
    return features

featuresets = [find_features(row[0]) for row in documents]

featuresets = pd.DataFrame(featuresets)
featuresets = pd.concat([df["cuisine"], featuresets], axis=1)
predictors = list(featuresets.columns.values)[1:]
target = list(featuresets.columns.values)[0]
train, test = train_test_split(featuresets, test_size=0.3, random_state=44, stratify=featuresets["cuisine"].tolist())


nb = NB()
nb.fit(train[predictors], train[target])
predictions = nb.predict(test[predictors])
actual = test[target].tolist()

print(nb.score(test[predictors], test[target]))
print(classification_report(actual, predictions))
print(accuracy_score(actual, predictions))

clf = svm.SVC(decision_function_shape='ovo')
clf.fit(train[predictors], train[target])
predictions = clf.predict(test[predictors])
actual = test[target].tolist()

print(clf.score(test[predictors], test[target]))
print(classification_report(actual, predictions))
print(accuracy_score(actual, predictions))
