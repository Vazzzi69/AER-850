# AER850 Project 1
# Vasi Sivakumar
#501024572

# Data Processing 

import pandas as pd

df = pd.read_csv("Project 1 Data.csv")
dfinfo = df.info()

print(dfinfo)  # check to see if data is indeed in 

# Data Visualization

import matplotlib.pyplot as plt
import numpy as np

    # Sorting data
    
xc = df['X']
yc = df['Y']
zc = df['Z']
step = df['Step']
 

    # Scatter plot 
    
fig1= plt.figure()

ax1= fig1.add_subplot(111, projection='3d')
ax1.scatter(xc, yc, zc)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')    
ax1.set_zlabel('Z')
ax1.set_title('Coordinates visualization Plot for Maintenance Steps')

plt.show()

# correlation analysis 

import seaborn as sns 

crmtrx =  df.drop(columns = "Step")
corr_matrix = crmtrx.corr()
sns.heatmap(corr_matrix)

#Classification Model Development/Engineering


    # im going to check the data to see any data bias per step


strat = df.groupby('Step')

for step, group in strat:
    print(f"Step {step}:")
    num_data_points = len(group)
    print(f"Number of data points: {num_data_points}")
    
    # steps 7,8,9 have way more points... 148,221,251 rest have 24
    
    #make some test and training sets
    
from sklearn.model_selection import StratifiedShuffleSplit

# stratify data and train sets

X = df[["X", "Y","Z"]]
y = df['Step']

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(X, y):
    train_X, test_X = X.iloc[train_index], X.iloc[test_index]
    train_y, test_y= y.iloc[train_index], y.iloc[test_index]


    #models i should use? 
    
# random forest


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(train_X, train_y)
rf_model_predictions = rf_model.predict(train_X)
rf_model_mae = mean_absolute_error(rf_model_predictions, train_y)


print("Random Forest training MAE is: ", round(rf_model_mae,2))

# grid search cross validation

from sklearn.model_selection import GridSearchCV
 
param_grid ={
    
    'n_estimators': [100, 200, 300],  
    'max_depth': [None, 10, 20],     
    'min_samples_split': [2, 5, 10] 
    
    }

# might change the scoring ??


grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(train_X, train_y)
best_params = grid_search.best_params_
best_rf_model = grid_search.best_estimator_
 
print("Best Hyperparameters RF:", best_params)
 

# support vector machines 

from sklearn.svm import SVC

svm_model = SVC(C=1.0, kernel='rbf', gamma='scale')
svm_model.fit(train_X, train_y)
svm_model_predictions = svm_model.predict(train_X)
svm_model_mae = mean_absolute_error(svm_model_predictions, train_y)


param_grid ={
    
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.1, 1]
    
    }

# grid search cross validation

grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(train_X, train_y)
best_params = grid_search.best_params_
best_svm_model = grid_search.best_estimator_
 
print("Best Hyperparameters SVM:", best_params)
 
#linear regressin
    
from sklearn.linear_model import LinearRegression


linreg_model = LinearRegression()
linreg_model.fit(train_X, train_y)
linreg_model_predictions = svm_model.predict(train_X)
linreg_model_mae = mean_absolute_error(svm_model_predictions, train_y)
   
 # grid search cross validation

param_grid = {
    'fit_intercept': [True, False],
    'normalize': [True, False]
}

grid_search.fit(train_X, train_y)
best_params = grid_search.best_params_
best_linreg_model = grid_search.best_estimator_

print("Best Hyperparameters linear regression:", best_params)