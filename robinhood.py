# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 23:49:03 2020

@author: jkim0
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb

from seaborn import heatmap
SEED = 42

# Read in data and format columns
df = pd.read_csv("robinhood_data.csv")
df.columns = [x.replace("\n","").strip() for x in df.columns.tolist()]

# Remove columns with more than 20% missing values, and replace NaNs with mean
# of the column for the remaining columns
df = df.loc[:, df.isna().mean() < 0.2]
df.fillna(df.mean(), inplace = True)

# Devide X and ys into wanted columns
col_y = ['No. of job placements per individual placed','% placed', 'No. of job placements','No. of individuals placed in full-time jobs','% placed in full-time jobs','No. of individuals placed in temporary jobs','% placed in temporary jobs']
y = df[col_y]
df.drop(col_y, axis = "columns", inplace = True)
# scale % fulltime and %temporary to entire population
y["%fulltime"] = y.loc[:,"% placed"] * y.loc[:,"% placed in full-time jobs"]
y["%temporary"] = y.loc[:,"% placed"] * y.loc[:,"% placed in temporary jobs"]

X = df[df.columns[5:]]
y = y[["% placed", "%fulltime", "%temporary"]]

# Split train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = SEED)

# LASSO Regression
lasso_model = dict()
lasso_results = dict()

# SVM Regression
svm_model = dict()
svm_results = dict()

# Random Forest Regression
rf_model = dict()
rf_results = dict()

# XGBOOST Regression
xgb_model = dict()
xgb_results = dict()

models = ["LASSO", "SVM", "RF", "XGB"]

for model in models:
    for target in y.columns:
        # create model
        if model == "LASSO":
            reg = LassoCV(cv=3, random_state = SEED)
        elif model == "SVM":
            reg = SVR()
        elif model == "RF":
            reg = RandomForestRegressor(random_state = SEED)
        elif model == "XGB":
            reg = xgb.XGBRegressor(objective ='reg:squarederror', scoring='r2', random_state = SEED)

        # train model
        reg.fit(X_train, y_train[target])

        # score the model
        train_score = reg.score(X_train, y_train[target])
        test_score = reg.score(X_test, y_test[target])

        # store results
        if model == "LASSO":
            lasso_model[target] = reg
            lasso_results[target] = (train_score, test_score)
        elif model == "SVM":
            svm_model[target] = reg
            svm_results[target] = (train_score, test_score)
        elif model == "RF":
            rf_model[target] = reg
            rf_results[target] = (train_score, test_score)
        elif model == "XGB":
            xgb_model[target] = reg
            xgb_results[target] = (train_score, test_score)

# gather all results into one data frame
lasso = pd.DataFrame(lasso_results).T
svm = pd.DataFrame(svm_results).T
rf = pd.DataFrame(rf_results).T
xgboost = pd.DataFrame(xgb_results).T

lasso["model"] = "LASSO"
svm["model"] = "SVM"
rf["model"] = "RF"
xgboost["model"] = "XGBoost"
result = lasso.append([svm,rf,xgboost])

# get the feature importance for random forest - which had the best prediction results
feat_imp = [model.feature_importances_ for model in rf_model.values()]
feat_imp = pd.DataFrame(feat_imp, columns = X.columns)
feat_imp.index = y.columns

print(result)
print(feat_imp)
plt.figure(figsize=(20,9))
heatmap(feat_imp.T, cmap="YlGnBu")
plt.savefig("featimp.png")
