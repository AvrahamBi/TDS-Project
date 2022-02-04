from sklearn.datasets import load_breast_cancer, load_wine, load_iris, load_digits, load_linnerud, load_diabetes
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import chi2

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


def check(ds):
    reduce(ds, method="k_best", k=3)
    reduce(ds, method="percentile", p=10)

############ MAIN #############

iris = load_iris()
digits = load_digits()
diabetes = load_diabetes()
wine = load_wine()
cancer = load_breast_cancer()

ds = [iris, digits, diabetes, wine, cancer]
for d in ds:
    check(d)
