import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score



def prepare_Y(df_in):
    df_in = df_in
    df_in = df_in['SalePrice']
    y = df_in
    return y

def onehotencoding_and_prepare_X(df_in):
    data = df_in
    data.drop(['SalePrice'], axis=1, inplace=True)
    col = []
    run = [col.append(i) for i in data.columns]

    i = 0
    while i < len(col):
        add_columns = pd.get_dummies(data[col[i]])
        data.drop([col[i]], axis=1, inplace=True)
        data = data.join(add_columns, how='left', lsuffix=f'{col[i]}_left', rsuffix=f'{col[i]}_right')
        i += 1

    return data



def evaluate_metric(model, X_test, y_test):
    return f1_score(y_test, model.predict(X_test), average='micro')



def forward_feature_selection(X_train, X_test, y_train, y_test, n):
    feature_set = []
    for num_features in range(n):
        metric_list = []
        model = SGDClassifier()
        for feature in X_train.columns:
            if feature not in feature_set:
                f_set = feature_set.copy()
                f_set.append(feature)
                model.fit(X_train[f_set], y_train)
                metric_list.append((evaluate_metric(model, X_test[f_set], y_test), feature))

        metric_list.sort(key=lambda x : x[0], reverse=True)
        feature_set.append(metric_list[0][1])
    return feature_set

filepath_train = 'regression_train.csv'
filepath_test = 'regression_test.csv'



data_train = pd.read_csv(filepath_train)
data_train = data_train.dropna(axis=1)

data_test = pd.read_csv(filepath_test)
data_test = data_test.dropna(axis=1)

merged_dataset = pd.concat([data_train, data_test], ignore_index=True)


Y = prepare_Y(merged_dataset)
X = onehotencoding_and_prepare_X(merged_dataset)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)


fs = forward_feature_selection(X_train, X_test, y_train, y_test, 5)

print(fs)



