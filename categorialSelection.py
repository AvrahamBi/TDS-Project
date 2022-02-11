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
    #ds = ds.astype(str)
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
    selector = SelectKBest(score_func=chi2, k=3)
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
    # plot the scores
    compareSizes(scores)
    #
    pyplot.bar([features[i] for i in range(len(scores))], scores)
    pyplot.show()
    return features, scores

# function find the biggest jump of scores
def compareSizes(scores):
    sizeMulti = []
    for i in range(len(scores)-1):
        differ = scores[i+1] / scores[i]
        sizeMulti.append(differ)
    print("SIZE NULTI:", sizeMulti)
    print("MAX MULTI:", max(sizeMulti))


############ MAIN #############

# X, y, features = ds_loader("video_games_ds.csv", 10) # target columns is: Critic_Score
# reduced_x , selector = select_features(X, y)
# features, scores = showGraph(selector, features, 10)


X, y, features = ds_loader("wine_ds.csv", 4) # target columns is: Points
reduced_x , selector = select_features(X, y)
showGraph(selector, features, 4)


















# def reduce(ds, method="k_best", k=3, p=10):
#     dataSet = ds
#     X = dataSet.data
#     y = dataSet.target
#     # Convert to categorical data by converting data to integers
#     X = X.astype(int)
#     # Two features with highest chi-squared statistics are selected
#     if (method == "k_best"): chi2_features = SelectKBest(chi2, k=k)
#     if (method == "percentile"): chi2_features = SelectPercentile(chi2,percentile=10)
#
#     X_kbest = chi2_features.fit_transform(X, y)
#     # Reduced features
#     print('Original feature number:', X.shape[1])
#     print('Reduced feature number:', X_kbest.shape[1])



#checkGraphs()

#X, y = load_iris(return_X_y=True)
#iris - load_iris()
#digits = load_digits()
# diabetes = load_diabetes()
# wine = load_wine()
#cancer = load_breast_cancer()
# ds = [iris, digits, diabetes, wine, cancer]
# for d in ds:
#     check(d)


































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