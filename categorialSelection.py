import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer, load_wine, load_iris, load_digits, load_linnerud, load_diabetes
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

def reduce(ds, method="k_best", k=3, p=10):
    dataSet = ds
    X = dataSet.data
    y = dataSet.target
    # Convert to categorical data by converting data to integers
    X = X.astype(int)
    # Two features with highest chi-squared statistics are selected
    if (method == "k_best"): chi2_features = SelectKBest(chi2, k=k)
    if (method == "percentile"): chi2_features = SelectPercentile(chi2,percentile=10)

    X_kbest = chi2_features.fit_transform(X, y)
    # Reduced features
    print('Original feature number:', X.shape[1])
    print('Reduced feature number:', X_kbest.shape[1])

def showScore():
    data = pd.read_csv("train.csv")
    X = data.iloc[:,0:20]  #independent columns
    y = data.iloc[:,-1]    #target column i.e price range

    #apply SelectKBest class to extract top 10 best features
    bestfeatures = SelectKBest(score_func=chi2, k=10)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    #concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    print(featureScores.nlargest(10,'Score'))  #print 10 best features

def showGraph():
    data = pd.read_csv("train.csv")
    X = data.iloc[:, 0:20]  # independent columns
    y = data.iloc[:, -1]  # target column i.e price range

    model = ExtraTreesClassifier()
    model.fit(X, y)
    print(model.feature_importances_)  # use inbuilt class feature_importances of tree based classifiers
    # plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.show()


def check(ds):
    reduce(ds, method="k_best", k=3)
    reduce(ds, method="percentile", p=10)

def checkGraphs():
    showScore()
    showGraph()


############ MAIN #############


checkGraphs()

#X, y = load_iris(return_X_y=True)
#iris - load_iris()
# digits = load_digits()
# diabetes = load_diabetes()
# wine = load_wine()
# cancer = load_breast_cancer()
# ds = [iris, digits, diabetes, wine, cancer]
# for d in ds:
#     check(d)
