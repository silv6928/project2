#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Tony Silva
# CS 5970 - Text Analytics
# Project 2 Phase 2

import nltk
import pandas as pd
import pickle
import os, sys
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB as NBB
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors


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

# Creates the data set that was used for generation of the machine learning models
# as well as the usage of K-Nearest Neighbors search. The data set is formatted in a pandas dataframe
# where each column is an ingredient and each row is a recipe from the yummly.json file provided for the project
# the contents of each cell of the data frame is True or False. True means the ingredient designated as that column is
# contained in the row's recipe. False means the ingredient designated as that column is not contained in the recipe
def create_feature_set():
    dir_path = os.path.dirname(sys.argv[0])
    df = pd.read_json(dir_path + "/data/yummly.json")
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


# This function trains a naive bayes model on the featureset that was generated in the create_feature_set module.
# It trains a model on 70% training test, and tests accuracy on the other 30% test data.
# Accuracy is around 73%
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


# This function trains a support vector machine learning model on the featureset data that was generated.
# It trains a model on 70% training test, and tests accuracy on the other 30% test data.
# Accuracy is around 75%.
# This model was used in making predictions in the Cuisine Prediction System. The model was saved as a pickle file
# The pickle file is then used to quickly pull in the trained classifier and generate predictions.
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


# This function trains a logistic regression model on the featureset data that was generated.
# It trains a model on 70% training test, and tests accuracy on the other 30% test data.
# Accuracy was about 75%-77%.
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


# Helper function
# Returns the loaded classifier from the pickle file.
def load_model():
    dir_path = os.path.dirname(sys.argv[0])
    filename = "/models/finalized_model.sav"
    clf = pickle.load(open(dir_path + filename, 'rb'))
    return clf


# This grabs the list of ingredient features from the data set
# The return value is used to generate the user's ingredients into a feature vector
def grab_df_ingredients():
    dir_path = os.path.dirname(sys.argv[0])
    df = pd.read_json(dir_path + "/data/yummly.json")
    col_names = ['id', 'cuisine', 'ingredients']
    df = df.reindex(columns=col_names)
    all_ingredients = []
    for i in df.ingredients:
        all_ingredients.extend(i)
    all_ingredients = nltk.FreqDist(all_ingredients)
    # Get the 2200 most common ingredients
    ingredient_features = [i[0] for i in all_ingredients.most_common()[:2200]]
    return ingredient_features


# This actually takes the users input and all of the ingredient features and generates a single
# feature vector for the user's input. So that the single vector can have a prediction generated for its cuisine
def user_input_to_feature(document, ingredient_features):
    ingredients = set(document)
    feature = {}
    for i in ingredient_features:
        feature[i] = (i in ingredients)
    return feature


# Generate a K-Nearest Neighbor search over the data set. It will return the Recipe ID's of the two nearest neighbors.
def find_nearest_neighbors(temp, df):
    predictors = list(df.columns.values)[1:]
    knn = NearestNeighbors(n_neighbors=2, algorithm='auto',).fit(df[predictors])
    indices = knn.kneighbors(temp, 2, return_distance=False)
    return indices[0][0], indices[0][1]


# This is the command line User Interface that executes phase 2. This makes calls the functions above to pull in the
# the data, request ingredients to make predictions on, predict the cuisine, and find the 2 nearest neighbors.
def ui():
    print("Welcome to the Cuisine Prediction System!")
    print("Please wait as the system collects the data. (This takes 1-2 minutes)")
    ingredient_features = grab_df_ingredients()
    featureset = create_feature_set()
    print("Data Collected!")
    clf = load_model()
    num = 99
    while num != 0:
        print("Please enter 1 to proceed with Cuisine Prediction.")
        print("Enter 0 to Exit the System.")
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
            else:
                # Insert vectorize ingredient list
                feature = user_input_to_feature(recipe, ingredient_features)
                temp = pd.Series(feature)
                temp = temp.values.reshape(1, -1)
                print("Your Cuisine Type is: ", clf.predict(temp))
                print("Finding the 2 Closest Recipe IDs (This takes 1-2 minutes)")
                print("Your Recipe IDs are: ", find_nearest_neighbors(temp, featureset))
        elif num == 0:
            print("Thanks for using the Cuisine Predictor")
        else:
            print("Please enter a valid input!")
