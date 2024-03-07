import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

###############################################################
# Read the data
###############################################################

# Read data from csv file 
raw_data = pd.read_csv('Assignment1\case1Data.txt')

# Giving NaN values the correct NaN value 
raw_data = raw_data.replace(' NaN', np.nan)

# Splitting in X and y
y = raw_data['y']
X_num = raw_data.loc[:, ' x_ 1':' x_95'].astype(float) 
X_cat = raw_data.loc[:, ' C_ 1':' C_ 5'] 



###############################################################
# Basic stats and distribution
###############################################################


# Basic stats 
raw_data.describe()


# Set the size of the overall figure
plt.figure(figsize=(20, 45))

# Loop through all columns and create a subplot for each feature
for i, column in enumerate(X_num.columns, 1):
    plt.subplot(20, 5, i)  # Adjust grid size based on your number of features
    sns.histplot(X_num[column], kde=True, bins=20)  # You can adjust bins as needed
    plt.title(column)

plt.tight_layout()
plt.show()



# Correlation matrix
correlation_matrix = X_num.corr()
np.where((correlation_matrix > 0.65) & (correlation_matrix != 1))


# Create box plots for numerical variables
plt.figure(figsize=(15, 6))  # Adjust the figure size if needed
X_num.boxplot(rot=90)
plt.title('Box Plot for Numerical Variables')
plt.xlabel('Variable')
plt.ylabel('Value')
plt.show() 


###############################################################
# Fix missing data 
###############################################################

# Fix missing data for numerical columns
for column in X_num.columns:
    if X_num[column].isnull().any():
        mean_value = X_num[column].mean()
        X_num = X_num[column].fillna(mean_value, inplace=True)

X_num.fillna(X_num.mean(), inplace=True)

# Fix missing data for categorical columns
for column in X_cat.columns:
    if X_cat[column].isnull().any():
        mode_value = X_cat[column].mode().iloc[0]  # using mode for categorical data
        X_cat = X_cat[column].fillna(mode_value, inplace=True)

# Concatenate the imputed numerical and categorical data
data = pd.concat([y, X_num, X_cat], axis=1)


###############################################################
# Branch 1: Categorical data model encoding
###############################################################

# There are three branches in the categorical data model encoding branch
# 1.1: One hot encoding (L2/L1 regularization)
# 1.2: Random forest regression
# 1.3: Boosting regression

## Imports:
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Data processing for categorical data:
X_cat_np = X_cat.to_numpy()

##############
# 1.1: One hot encoding (L2/L1 regularization)
##############
# One hot encode the categorical data
encoder = OneHotEncoder()
X_cat_encoded = encoder.fit_transform(X_cat_np)

lasso = LassoCV(cv=5)
lasso.fit(X_cat_encoded, y)
lasso.score(X_cat_encoded, y)
mean_squared_error(y, lasso.predict(X_cat_encoded))

ridge = RidgeCV(cv=5)
ridge.fit(X_cat_encoded, y)
ridge.score(X_cat_encoded, y)
mean_squared_error(y, ridge.predict(X_cat_encoded))


##############
# 1.2: Random forest regression
##############
# Label encode the categorical data, so that the labels are translated to data for random forest and boosting
label_encoders = []
for i in range(len(X_cat_np[0])):
    label_encoders.append(LabelEncoder())
    X_cat_np[:, i] = label_encoders[i].fit_transform([row[i] for row in X_cat_np])

# randomized search for random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

param_grid = {'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=100, cv=3, verbose=2, n_jobs=-1)
rf_random.fit(X_cat_np, y)
bparams = rf_random.best_params_

rf2 = RandomForestRegressor(n_estimators=bparams['n_estimators'], min_samples_split=bparams['min_samples_split'], min_samples_leaf=bparams['min_samples_leaf'], max_depth=bparams['max_depth'], bootstrap=bparams['bootstrap'])
rf2.fit(X_cat_np, y)
y_pred = rf2.predict(X_cat_np)
mean_squared_error(y, y_pred)

##############
# 1.3: Boosting regression
##############
# https://www.researchgate.net/publication/2424244_Improving_Regressors_Using_Boosting_Techniques
# Randomized search for boosting regression
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
learning_rate = np.linspace(0.01, 1, 50)
loss = ['linear', 'square', 'exponential']
estimator = [LassoCV(), RidgeCV(), None]

param_grid = {'estimator' : estimator,
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'loss': loss}

ada = AdaBoostRegressor()
ada_random = RandomizedSearchCV(estimator=ada, param_distributions=param_grid, n_iter=100, cv=3, verbose=2, n_jobs=-1)
ada_random.fit(X_cat_np, y)
bparams = ada_random.best_params_

y_pred = ada_random.predict(X_cat_np)
mean_squared_error(y, y_pred, squared=False)
bparams


###############################################################
# Branch 2: Combine numerical and categorical prediction
###############################################################

# There are three branches in the combined numerical and categorical prediction branch
# 2.1: Linear regression (L1/L2 regularization)
# 2.2: Random forest regression
# 2.3: Boosting regression


###############################################################
# Other data transformations 
###############################################################


###############################################################
# Prediction model: ??? 
###############################################################


# Indsæt white noise, for at teste om variable importance på dem er høj
# Fordi så skal det være en anden model, der kan håndtere det end RF


