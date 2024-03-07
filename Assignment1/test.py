# Read data from csv file 
raw_data = pd.read_csv('Assignment1\case1Data.txt')

# Giving NaN values the correct NaN value 
raw_data = raw_data.replace(' NaN', np.nan)

# Splitting in X and y
y = raw_data['y']
X_num = raw_data.loc[:, ' x_ 1':' x_95'].astype(float) 
X_cat = raw_data.loc[:, ' C_ 1':' C_ 5'] 

