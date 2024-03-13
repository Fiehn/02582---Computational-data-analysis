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
# Branch 1: Categorical data model encoding
###############################################################

# There are three branches in the categorical data model encoding branch
# 1.1: One hot encoding (L2/L1 regularization)
# 1.2: Random forest regression
# 1.3: Boosting regression

## Imports:
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV

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
estimator = [LassoCV(max_iter=5000),Lasso(max_iter=5000),Ridge(), RidgeCV(), None]

param_grid = {'estimator' : estimator,
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'loss': loss}

# Label-encoded data
ada = AdaBoostRegressor()
ada_random = RandomizedSearchCV(estimator=ada, param_distributions=param_grid, n_iter=100, cv=3, verbose=2, n_jobs=-1)
ada_random.fit(X_cat_np, y)
bparams = ada_random.best_params_

y_pred = ada_random.predict(X_cat_np)
root_mean_squared_error(y, y_pred) # 54.189355753221015

# {'n_estimators': 400, 'loss': 'linear', 'learning_rate': 0.8383673469387755, 'estimator': LassoCV(max_iter=5000)}

## One-hot encoding
ada = AdaBoostRegressor()
ada_random = RandomizedSearchCV(estimator=ada, param_distributions=param_grid, n_iter=100, cv=3, verbose=2, n_jobs=-1)
ada_random.fit(X_cat_encoded, y)
bparams = ada_random.best_params_

y_pred = ada_random.predict(X_cat_encoded)
root_mean_squared_error(y, y_pred) # 49.80523632875382
# {'n_estimators': 2000, 'loss': 'linear', 'learning_rate': 0.5757142857142857, 'estimator': LassoCV(max_iter=5000)}

ada = AdaBoostRegressor(n_estimators=2000, loss='linear', learning_rate=0.5757142857142857, estimator=LassoCV(max_iter=5000))
ada.fit(X_cat_encoded, y)
y_pred = ada.predict(X_cat_encoded)

###############################################################
# Branch 2: Combine numerical and categorical prediction
###############################################################

# There are three branches in the combined numerical and categorical prediction branch
# 2.1: Linear regression (L1/L2 regularization)
# 2.2: Random forest regression
# 2.3: Boosting regression

X_num["y_cat"] = y_pred
X = X_num

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def centerData(data):
    
    mu = np.mean(data,axis=0)
    data = data - mu
    
    return data, mu

def normalize(X):
    '''
    Function for normalizing the columns (variables) of a data matrix to unit length.
    Returns the normalized data and the euclidian lenghts of the variables 
    
    Input  (X) --------> The data matrix to be normalized 
    Output (X_pre)-----> The normalized data matrix 
    Output (d) --------> Array with the euclidian lenghts of the variables 
    '''
    d = np.linalg.norm(X,axis=0,ord=2)  # d is the the L2 norms of the variables
    d[d==0]=1                           # Avoid dividing by zero if column L2 norm is 0 
    X_pre = X / d                       # Normalize the data with the euclidian lengths
    return X_pre,d                      # Return normalized data and the euclidian lengths

###########
# 2.1: Linear regression (L1 regularization)
###########
# Set up the alpha range and other variables
alphas = np.linspace(0.9, 0.001, 100)  # Range of alphas to test
CV = 5  # Number of cross-validation folds
kf = KFold(n_splits=CV)

# Prepare arrays to store results
Err_tr = np.zeros((CV, len(alphas)))
Err_tst = np.zeros((CV, len(alphas)))

# Start cross-validation
for i, (train_index, test_index) in enumerate(kf.split(X,y)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Data preprocessing (impute, center, normalize)
    imp = SimpleImputer(strategy='mean')
    X_train_imputed = imp.fit_transform(X_train)
    X_test_imputed = imp.transform(X_test)

    # Assuming centerData and normalize are functions you've defined previously
    y_train, y_mean = centerData(y_train)  # Center training response
    y_test = y_test - y_mean                # Center test response using the same mean

    X_train_imputed, x_mu = centerData(X_train_imputed)  # Center training data
    X_test_imputed = X_test_imputed - x_mu               # Center test data using the same mean

    X_train_imputed, x_scale = normalize(X_train_imputed)  # Normalize training data
    X_test_imputed = X_test_imputed / x_scale              # Normalize test data using the same scale

    # Iterate through alphas
    for j, alpha in enumerate(alphas):
        # Fit Lasso model with current alpha
        reg = Lasso(alpha=alpha)
        reg.fit(X_train_imputed, y_train)
        
        # Predict and find error for both train and test datasets
        YhatTr = reg.predict(X_train_imputed)  # Use the model's predict method
        YhatTest = reg.predict(X_test_imputed)  # Use the model's predict method
        
        # Store the training and test errors (MSE)
        Err_tr[i, j] = root_mean_squared_error(y_train, YhatTr)  # Training error
        Err_tst[i, j] = root_mean_squared_error(y_test, YhatTest) # Test error

# Calculate the average RMSE over all CV folds for each alpha
mean_err_tr = np.sqrt(np.mean(Err_tr, axis=0))  # Average training RMSE for each alpha
mean_err_tst = np.sqrt(np.mean(Err_tst, axis=0))  # Average test RMSE for each alpha
std_err_tst = np.std(Err_tst, axis=0) / np.sqrt(CV)  # Standard error of test RMSE for each alpha

# Find the index of the smallest average test RMSE
optimal_alpha_index = np.argmin(mean_err_tst)
optimal_alpha = alphas[optimal_alpha_index]
optimal_err_tst = mean_err_tst[optimal_alpha_index]
optimal_err_ste = std_err_tst[optimal_alpha_index]

# Print the details of the optimal model
print("Optimal alpha:", optimal_alpha)
print("Associated mean test RMSE:", optimal_err_tst)
print("Standard error of the mean test RMSE:", optimal_err_ste)
print("Associated mean training RMSE for optimal alpha:", mean_err_tr[optimal_alpha_index])

# Plot the training and test RMSE against alphas
plt.figure(figsize=(10, 6))
plt.plot(alphas, mean_err_tr, label='Average Training RMSE', marker='o')
plt.plot(alphas, mean_err_tst, label='Average Test RMSE', marker='s')
plt.scatter(optimal_alpha, optimal_err_tst, color='red', zorder=5, label='Optimal Alpha')  # Highlight the optimal alpha
plt.xscale('log')  # Since alphas vary on a log scale, this makes the plot easier to interpret
plt.title('Model Performance vs. Alpha')
plt.xlabel('Alpha')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.show()


# make a prediction
reg = Lasso(alpha=optimal_alpha)
X_imputed = imp.fit_transform(X)
reg.fit(X_imputed, y)

reg.coef_

root_mean_squared_error(y,reg.predict(X_imputed))

plt.plot(reg.predict(X_imputed),'o', label='Predicted')
plt.plot(y,'x', label='Actual')
plt.legend()
plt.show()

###########
# 2.2: AdaBoost regression
###########

# Randomized search for boosting regression
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
learning_rate = np.linspace(0.01, 1, 50)
loss = ['linear', 'square', 'exponential']
estimator = [LassoCV(max_iter=50000), RidgeCV(), None]

param_grid = {'estimator' : estimator,
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'loss': loss}


X_imputed = X.fillna(X.mean())

X_imputed_scaled = normalize(X_imputed)[0]

ada = AdaBoostRegressor()
ada_random = RandomizedSearchCV(estimator=ada, param_distributions=param_grid, n_iter=50, cv=3, verbose=2, n_jobs=-1)
ada_random.fit(X_imputed_scaled, y)
bparams = ada_random.best_params_

y_pred_branch_2 = ada_random.predict(X_imputed)
root_mean_squared_error(y, y_pred_branch_2) 



