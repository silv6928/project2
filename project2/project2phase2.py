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
    filename = 'finalized_model.sav'
    pickle.dump(clf, open(filename, 'wb'))


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
    filename = 'models/finalized_model.sav'
    clf = pickle.load(open(filename, 'rb'))
    return clf

# Make a prediction on a single instance
# Used to test predictions!
# temp = featuresets.loc[1, list(featuresets.columns.values)[1:]]
# temp = temp.values.reshape(1, -1)
# print(clf.predict(temp))
# print(featuresets.loc[1, list(featuresets.columns.values)[0]])


def grab_df_ingredients():
    df = pd.read_json("data/yummly.json")
    col_names = ['id', 'cuisine', 'ingredients']
    df = df.reindex(columns=col_names)
    all_ingredients = []
    for i in df.ingredients:
        all_ingredients.extend(i)
    all_ingredients = nltk.FreqDist(all_ingredients)
    # Get the 2200 most common ingredients
    ingredient_features = [i[0] for i in all_ingredients.most_common()[:2200]]
    return df, ingredient_features


def user_input_to_feature(document, ingredient_features):
    ingredients = set(document)
    feature = {}
    for i in ingredient_features:
        feature[i] = (i in ingredients)
    return feature


def ui():
    print("Welcome to the Cuisine Prediction System!")
    print("Please wait as the system collects the data")
    df, ingredient_features = grab_df_ingredients()
    print("Data Collected!")
    print("Please enter 1 to proceed with Cuisine Prediction.")
    print("Enter 0 to Exit the System.")
    clf = load_model()
    num = 99
    while num != 0:
        num = int(str(input('--> ')))
        if num == 1:
            print("You selected Cuisine Prediction")
            print("You will enter ingredient by ingredient")
            print("When you have typed in an ingredient press ENTER")
            print("When you are done please ENTER in the number 5")
            check = 99
            recipe = []
            while check != 5:
                ing = str(input("Enter Ingredient --> "))
                if ing == "5":
                    check = 5
                    break
                else:
                    recipe.append(ing)
            if len(recipe) == 0:
                print("You didn't type in any ingredients")
                print("Welcome to the Cuisine Prediction System!")
                print("Please enter 1 to proceed with Cuisine Prediction.")
                print("Enter 0 to Exit the System.")
            else:
                # Insert vectorize ingredient list
                feature = user_input_to_feature(recipe, ingredient_features)
                temp = pd.Series(feature)
                temp = temp.values.reshape(1, -1)
                print("Your Cuisine Type is: ", clf.predict(temp))
        elif num == 0:
            print("Thanks for using the Cuisine Predictor")
        else:
            print("Please enter a valid input!")
