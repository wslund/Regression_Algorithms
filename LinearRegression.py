

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import seaborn as sns



filepath_train = 'regression_train.csv'
filepath_test = 'regression_test.csv'



data_train = pd.read_csv(filepath_train)
data_train = data_train.dropna(axis=1)

data_test = pd.read_csv(filepath_test)
data_test = data_test.dropna(axis=1)


train_col = []
run = [train_col.append(i) for i in data_train.columns]

test_col = []
run2 = [test_col.append(i) for i in data_test.columns]



#print(len(train_col))
#print(len(train_col))



merged_dataset = pd.concat([data_train, data_test], ignore_index=True)



def trimm_correlated(df_in, threshold):
    df_corr = df_in.corr()
    df_not_correlated = ~(df_corr.mask(np.tril(np.ones([len(df_corr)]*2, dtype=bool))).abs() > threshold).any()
    un_corr_idx = df_not_correlated.loc[df_not_correlated[df_not_correlated.index] == True].index
    df_out = df_in[un_corr_idx]
    return df_out


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

merged_dataset = trimm_correlated(merged_dataset, 0.95)

first_Y = prepare_Y(merged_dataset)
first_X = onehotencoding_and_prepare_X(merged_dataset)


#print(len(data_train.index))
#print(len(data_test.index))



X_train = first_X.iloc[:1000]
X_test = first_X.iloc[1000:]

y_train = first_Y.iloc[:1000]
y_test = first_Y.iloc[1000:]




lr = LinearRegression(normalize=True)
lr = lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

lr.score(X_test, y_test)

lr_mse = mean_squared_error(y_test, y_pred)
lr_rmse = np.sqrt(lr_mse)


print(lr.score(X_test, y_test))
print('Mean Squared Error %.4f' % mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error %0.4f' %lr_rmse)
print('Coefficient of Determination (r2) %.4f' % r2_score(y_test, y_pred))


plt.style.use('default')
plt.style.use('ggplot')


fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(y_test, y_pred, alpha=0.3)
plt.suptitle('Test vs Prediction', fontsize=15)
plt.xlabel('Test', fontsize=10)
plt.ylabel('Prediction', fontsize=10)

fig.tight_layout()

sns.displot((y_test-y_pred), bins=10)
plt.xticks(rotation=45)
plt.show()




