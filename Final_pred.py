import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

raw_data = pd.read_csv('Assignment1\case1Data.txt')
raw_data = raw_data.replace(' NaN', np.nan)

y = raw_data['y']
X_num = raw_data.loc[:, ' x_ 1':' x_95'].astype(float) 
X_cat = raw_data.loc[:, ' C_ 1':' C_ 5'] 

# Step 1: Categorical data
X_cat_np = X_cat.to_numpy()

X_cat_features = []
for i in range(X_cat.shape[1]):
    unique_values = X_cat.iloc[:, i].unique()
    unique_values = unique_values[~pd.isna(unique_values)]  # Filter out nan values
    unique_values = np.append(unique_values, np.nan)
    X_cat_features.append(unique_values)

encoder = OneHotEncoder(categories=X_cat_features)
X_cat_encoded = encoder.fit_transform(X_cat_np)

ada = AdaBoostRegressor(n_estimators=2000, loss='linear', learning_rate=0.5757142857142857, estimator=LassoCV(max_iter=5000))
ada.fit(X_cat_encoded, y)
y_pred = ada.predict(X_cat_encoded)

# Step 2: Numerical data
X_num["y_cat"] = y_pred
X = X_num

X_imputed = X.fillna(X.mean())
reg = Lasso(alpha=0.1281313131313131, max_iter=5000)
reg.fit(X_imputed, y)

root_mean_squared_error(y, reg.predict(X_imputed))

colmn_names = X_imputed.columns

# step 3: Final prediction
raw_data = pd.read_csv('Assignment1\case1Data_Xnew.txt')
raw_data = raw_data.replace('NaN', np.nan)
raw_data = raw_data.replace(' NaN', np.nan)
raw_data = raw_data.replace(' NaN ', np.nan)
raw_data = raw_data.replace('NaN ', np.nan)

X_num = raw_data.loc[:, 'x_ 1':' x_95'].astype(float) 
X_cat = raw_data.loc[:, ' C_ 1':' C_ 5'] 

y_pred = ada.predict(encoder.fit_transform(X_cat.to_numpy()))

X_num['y_cat'] = y_pred
X_imputed = X_num.fillna(X_num.mean())

X_imputed.columns = colmn_names

y_pred = reg.predict(X_imputed)

np.savetxt('Assignment1\case1Data_y_pred.txt', y_pred, delimiter=',')
