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

# Fix missing data for categorical columns
for column in X_cat.columns:
    if X_cat[column].isnull().any():
        mode_value = X_cat[column].mode().iloc[0]  # using mode for categorical data
        X_cat = X_cat[column].fillna(mode_value, inplace=True)

# Concatenate the imputed numerical and categorical data
data = pd.concat([y, X_num, X_cat], axis=1)


###############################################################
# Feature handling: categorical data 
###############################################################
# https://www.saedsayad.com/decision_tree_reg.htm
# We do decison tree regression for the categorical data
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor # it does one hot encoding in the background
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# Make it numpy
X_cat_np = X_cat.to_numpy()

# Label encode the categorical data, so that the labels are translated to usable data
label_encoders = []
for i in range(len(X_cat_np[0])):
    label_encoders.append(LabelEncoder())
    X_cat_np[:, i] = label_encoders[i].fit_transform([row[i] for row in X_cat_np])

regressor = DecisionTreeRegressor()
regressor.fit(X_cat_np, y)
y_pred = regressor.predict(X_cat_np)

mean_squared_error(y, y_pred)

importances = regressor.feature_importances_


###############################################################
# Other data transformations 
###############################################################



###############################################################
# Prediction model: ??? 
###############################################################