import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split

# read data
df=pd.read_csv(r"https://raw.githubusercontent.com/srivatsan88/YouTubeLI/master/dataset/churn_data_st.csv",sep=",")
df.head()

# view types of columns
df.dtypes

# feature encoding for each categorial feature
cat_df = df
cat_df['gender'] = cat_df['gender'].map({'Female':1,'Male':0})
cat_df["Contract"] = cat_df["Contract"].map({'Month-to-month':0, 'One year':1, 'Two year':2})
cat_df["PaperlessBilling"] = cat_df["PaperlessBilling"].map({"Yes":0,"No":1})
cat_df.head()

# Chi squared test
x = cat_df.iloc[:,:-1]  #Independent variable
y = cat_df.iloc[:,-1]   #Target variable
f_score = chi2(x,y)   #returns f score and p value
print(f_score)