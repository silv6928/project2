#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nltk
import numpy as np
import pandas as pd
import pickle
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import MultinomialNB as NB
from sklearn.naive_bayes import BernoulliNB as NBB
from sklearn.metrics import classification_report, accuracy_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model

# 39774 different Receipes
# 6714 total unique ingredients
# Salt is the most common, followed by onions, and olive oil


# A lot of this code was used from the sentdex videos as provided in Unit 8
# Very helpful videos!
# Helper function taken from sentdex videos in unit 8
# def find_features(document):
#     ingredients = set(document)
#     features = {}
#     for i in ingredient_features:
#             features[i] = (i in ingredients)
#     return features


def create_feature_set():
    df = pd.read_json("yummly.json")
    col_names = ['id', 'cuisine', 'ingredients']
    df = df.reindex(columns=col_names)
    all_ingredients = []
    for i in df.ingredients:
        all_ingredients.extend(i)
    all_ingredients = nltk.FreqDist(all_ingredients)
    # Get the 2200 most common ingredients
    ingredient_features = [i[0] for i in all_ingredients.most_common()[:2200]]
    # Need to add the target value to the feature set for our algorithms to work

    def find_features(document):
        ingredients = set(document)
        features = {}
        for i in ingredient_features:
            features[i] = (i in ingredients)
        return features
    featuresets = [find_features(row[3]) for row in df.itertuples(index=True, name='Pandas')]
    featuresets = pd.DataFrame(featuresets)
    featuresets = pd.concat([df["cuisine"], featuresets], axis=1)
    return featuresets


def naive_bayes_model(featuresets):
    predictors = list(featuresets.columns.values)[1:]
    target = list(featuresets.columns.values)[0]
    train, test = train_test_split(featuresets, test_size=0.3, random_state=44)
    nb = NBB()
    nb.fit(train[predictors], train[target])
    predictions = nb.predict(test[predictors])
    actual = test[target].tolist()
    print(classification_report(actual, predictions))
    print(nb.score(test[predictors], test[target]))


def support_vector_model(featuresets):
    predictors = list(featuresets.columns.values)[1:]
    target = list(featuresets.columns.values)[0]
    train, test = train_test_split(featuresets, test_size=0.3, random_state=44)
    clf = svm.LinearSVC()
    clf.fit(train[predictors], train[target])
    predictions = clf.predict(test[predictors])
    actual = test[target].tolist()
    print(classification_report(actual, predictions))
    print(clf.score(test[predictors], test[target]))


def log_reg_model(featuresets):
    predictors = list(featuresets.columns.values)[1:]
    target = list(featuresets.columns.values)[0]
    train, test = train_test_split(featuresets, test_size=0.3, random_state=44)
    clf = linear_model.LogisticRegression()
    clf.fit(train[predictors], train[target])
    predictions = clf.predict(test[predictors])
    actual = test[target].tolist()
    print(classification_report(actual, predictions))
    print(clf.score(test[predictors], test[target]))
    filename = 'finalized_model.sav'
    pickle.dump(clf, open(filename, 'wb'))


# Returns the loaded classifier from the pickle file
def load_model():
    filename = 'finalized_model.sav'
    clf = pickle.load(open(filename, 'rb'))
    return clf

# Make a prediction on a single instance

featuresets = create_feature_set()
#log_reg_model(featuresets)
clf = load_model()

temp = featuresets.loc[1, list(featuresets.columns.values)[1:]]
temp = temp.values.reshape(1, -1)
print(clf.predict(temp))
print(featuresets.loc[1, list(featuresets.columns.values)[0]])
