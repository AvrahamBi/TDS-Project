import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer, load_wine, load_iris, load_digits, load_linnerud, load_diabetes
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from matplotlib import pyplot


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
    # oe = OneHotEncoding() # better but slower
    oe.fit(ds_x)
    ds_x = oe.transform(ds_x)
    # encode y
    le = LabelEncoder()
    le.fit(ds_y)
    ds_y = le.transform(ds_y)
    return ds_x, ds_y, features

def select_features(x, y):
    selector = SelectKBest(score_func=chi2, k=k)
    selector.fit(x, y)
    reduced_x = selector.transform(x)
    return reduced_x, selector

def sort_scores(selector, features):
    dict = {}
    for i in range(len(selector.scores_)):
        dict[features[i]] = selector.scores_[i]
    sorted_dict = {}
    sorted_keys = sorted(dict, key=dict.get)
    for w in sorted_keys:
        sorted_dict[w] = dict[w]
    features = []
    scores = []
    for i in sorted_dict:
        features.append(i)
        scores.append(sorted_dict[i])
    return features, scores

def showGraph(selector, features, target_index):
    features, scores = sort_scores(selector, features)
    for i in range(len(features)):
         print(features[i], scores[i])
    # calculate best K
    getK_line(scores)
    getK_scores(scores)
    # plot the scores
    pyplot.bar([features[i] for i in range(len(scores))], scores)
    pyplot.show()
    return features, scores

# function finds how many features have score above thrshold
def getK_scores(scores, threshold = 10000):
    indexToSelect = 0
    for i in range(len(scores)):
        if (threshold < scores[i]):
            indexToSelect = i
            break
    k = len(scores) - indexToSelect # K is number of features with score above threshold
    return k


# function find the biggest jump of scores
def getK_line(scores):
    sizeMulti = []
    indexToSelect = 0
    for i in range(len(scores)-1):
        differ = scores[i+1] / scores[i]
        sizeMulti.append(differ)
    for i in range(len(sizeMulti)):
        if sizeMulti[i] == max(sizeMulti):
            indexToSelect = i
    k = len(scores) - 1 - indexToSelect  # K is the number of best features we want to have in our reduced DS
    return k

############ MAIN #############

X, y, features = ds_loader("video_games_ds.csv", 10) # target columns is: Critic_Score
reduced_x , selector = select_features(X, y)
features, scores = showGraph(selector, features, 10)


# X, y, features = ds_loader("wine_ds.csv", 4) # target columns is: Points
# reduced_x , selector = select_features(X, y)
# showGraph(selector, features, 4)






















#
# def showScore():
#     data = pd.read_csv("train.csv")
#     X = data.iloc[:,0:20]  #independent columns
#     y = data.iloc[:,-1]    #target column i.e price range
#
#     #apply SelectKBest class to extract top 10 best features
#     bestfeatures = SelectKBest(score_func=chi2, k=10)
#     fit = bestfeatures.fit(X,y)
#     dfscores = pd.DataFrame(fit.scores_)
#     dfcolumns = pd.DataFrame(X.columns)
#     #concat two dataframes for better visualization
#     featureScores = pd.concat([dfcolumns,dfscores],axis=1)
#     featureScores.columns = ['Specs','Score']  #naming the dataframe columns
#     print(featureScores.nlargest(10,'Score'))  #print 10 best features
#
# def showGraph():
#     X_, y_ = load_iris(return_X_y=True)
#     X = pd.DataFrame(X_)
#     y = pd.DataFrame(y_)
#     #data = pd.read_csv("train.csv")
#     X = data.iloc[:, 0:20]  # independent columns
#     y = data.iloc[:, -1]  # target column i.e price range
#     model = ExtraTreesClassifier()
#     model.fit(X, y)
#     print(model.feature_importances_)  # use inbuilt class feature_importances of tree based classifiers
#     # plot graph of feature importances for better visualization
#     feat_importances = pd.Series(model.feature_importances_, index=X.columns)
#     feat_importances.nlargest(15).plot(kind='barh')
#     plt.show()