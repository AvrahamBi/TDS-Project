import math
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2

####### CONSTS ######
THRESHOLD = 10000

# load Data set from csv file
def ds_loader(filename, y_index):
    ds = pd.read_csv(filename, header=None, dtype='unicode').astype(str)
    # get pandas DF of features
    features_ = ds.iloc[0].drop(index=y_index)
    features = []
    # convert to list
    for i in features_:
        features.append(i)
    ds_x = ds.drop(ds.columns[[y_index]], axis=1)
    ds_y = ds[y_index]
    # encode x
    oe = OrdinalEncoder()
    oe.fit(ds_x)
    ds_x = oe.transform(ds_x)
    # encode y
    le = LabelEncoder()
    le.fit(ds_y)
    ds_y = le.transform(ds_y)
    return ds_x, ds_y, features

# discard columns and reduce data set
def reduce_features(x, y, k):
    selector = SelectKBest(score_func=chi2, k=k)
    reduced_x = selector.fit(x, y).transform(x)
    return reduced_x, selector

# sort scores and features list into 2 sorted lists
def sort_scores(selector, features):
    dict = {}
    for i in range(len(selector.scores_)):
        dict[features[i]] = selector.scores_[i]
    sorted_dict = {}
    sorted_keys = sorted(dict, key=dict.get)
    for i in sorted_keys:
        sorted_dict[i] = dict[i]
    features = []
    scores = []
    for i in sorted_dict:
        features.append(i)
        scores.append(sorted_dict[i])
    for i in range(len(scores)):
        scores[i] = math.floor(scores[i])
    return features, scores

# print to console the features and their scores
def showScores(features, scores, msg):
    print(msg)
    for i in range(len(features)):
        print("Feature:", features[i] + ",   Score:", scores[i])

# show graoh of features and scores to user
def showGraph(features, scores, msg):
    # plot the scores
    pyplot.bar([features[i] for i in range(len(scores))], scores)
    pyplot.title(msg)
    pyplot.xticks(rotation=90)
    pyplot.show()
    return features, scores

# function finds how many features have score above threshold
def getK_threshold(scores, threshold = 10000):
    indexToSelect = 0
    for i in range(len(scores)):
        if (threshold < scores[i]):
            indexToSelect = i
            break
    # K is number of features with score above threshold
    k = len(scores) - indexToSelect
    return k

# finds an elbow point in features scores
def getK_long_tail(scores):
    sizeMulti = []
    indexToSelect = 0
    for i in range(len(scores)-1):
        differ = scores[i+1] / scores[i]
        sizeMulti.append(differ)
    for i in range(len(sizeMulti)):
        if sizeMulti[i] == max(sizeMulti):
            indexToSelect = i
    # K is number of features above elbow point
    k = len(scores) - 1 - indexToSelect
    return k


# the main function that executes the process
def chooseK(ds, target_column_index):
    print("Start working on:", ds)
    # load data
    X, y, features = ds_loader(ds, target_column_index)
    # print some info of features and scores before reduction
    print("")
    print("Original number of features:", X.shape[1])
    selector = SelectKBest(score_func=chi2, k='all')
    selector.fit(X, y)
    features, scores = sort_scores(selector, features)
    showScores(features, scores, "Original features with their scores:")
    showGraph(features, scores, "Original features with their scores:")
    print("")
    #
    threshold_k = getK_threshold(scores, threshold=THRESHOLD)
    long_tail_k = getK_long_tail(scores)
    k = min(threshold_k, long_tail_k)
    print("K chosen by long_tail is:", long_tail_k)
    print("K chosen by threshold (" + str(THRESHOLD) + ") is:", threshold_k)
    print("Minimal K is:", k)
    print("")

    # here we got reduced dataset and reduced features and scores lists
    reduced_x, selector = reduce_features(X, y, k)
    reduced_features = features[-k:]
    reduced_scores = scores[-k:]

    # print some info of features and scores after reduction
    showScores(reduced_features, reduced_scores, "After reduction:")
    showGraph(reduced_features, reduced_scores, "After reduction:")


############ MAIN #############

chooseK("wine_ds.csv", 4)               # Target column: Points
chooseK("income_ds.csv", 14)            # Target column: income
chooseK("titanic_ds.csv", 4)            # Target column: Survived
chooseK("video_games_ds.csv", 9)        # Target column: Global_Sales


