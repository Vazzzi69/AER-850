# AER850 Project 1
# Vasi Sivakumar
#501024572

#2.1 Data Processing 

import pandas as pd

df = pd.read_csv("Project 1 Data.csv")
dfinfo = df.info()

print(dfinfo)  # check to see if data is indeed in 

# 2.2 Data Visualization

import matplotlib.pyplot as plt
import numpy as np

    # Sorting data
    
xc = df['X']
yc = df['Y']
zc = df['Z']
step = df['Step']
 

    # Scatter plot 


fig1= plt.figure()

unique_steps = step.unique()
colors = plt.cm.jet(np.linspace(0, 1, len(unique_steps)))

ax1= fig1.add_subplot(111, projection='3d')

for i, step_value in enumerate(unique_steps):
    step_mask = (step == step_value)
    ax1.scatter(xc[step_mask], yc[step_mask], zc[step_mask], c=[colors[i]], label=f'Step {step_value}')

ax1.set_xlabel('X')
ax1.set_ylabel('Y')    
ax1.set_zlabel('Z')
ax1.legend(loc=2,fontsize='small', frameon=False,bbox_to_anchor=(1, 0.5))
ax1.set_title('Coordinates visualization Plot for Maintenance Steps')

plt.show()


#2.3 correlation analysis 

import seaborn as sns 

crmtrx =  df.drop(columns = "Step")
corr_matrix = crmtrx.corr()
sns.heatmap(corr_matrix)

#2.4 Classification Model Development/Engineering


# im going to check the data to see any data bias per step CODE THIS OUT LATER


# strat = df.groupby('Step')

# for step, group in strat:
#     print(f"Step {step}:")
#     num_data_points = len(group)
#     print(f"Number of data points: {num_data_points}")
    
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

    
# random forest

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(train_X, train_y)
rf_model_predictions = rf_model.predict(test_X)

# grid search cross validation

from sklearn.model_selection import GridSearchCV
 
param_grid ={
    
    'n_estimators': [100, 200, 300],  
    'max_depth': [None, 10, 20],     
    'min_samples_split': [2, 5, 10] 
    
    }

grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(train_X, train_y)
best_params = grid_search.best_params_
best_rf_model = grid_search.best_estimator_
 
print("\nBest Hyperparameters RF:", best_params)
 
# support vector machines 

from sklearn.svm import SVC

svm_model = SVC(C=1.0, kernel='rbf', gamma='scale')
svm_model.fit(train_X, train_y)
svm_model_predictions = svm_model.predict(test_X)


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
 
print("\nBest Hyperparameters SVM:", best_params)
 
#linear regressin
    
from sklearn.linear_model import LinearRegression


linreg_model = LinearRegression()
linreg_model.fit(train_X, train_y)
linreg_model_predictions = svm_model.predict(test_X)
   
 # grid search cross validation

param_grid = {
    'fit_intercept': [True, False],
    'normalize': [True, False]
}

grid_search.fit(train_X, train_y)
best_params = grid_search.best_params_
best_linreg_model = grid_search.best_estimator_

print("\nBest Hyperparameters linear regression:", best_params)


#2.5 Model performance analysis

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


    #RFC model
accuracy_rf = accuracy_score(test_y, rf_model_predictions)
precision_rf = precision_score(test_y, rf_model_predictions, average='weighted',zero_division=0)
f1_rf = f1_score(test_y, rf_model_predictions, average='weighted')

print("\n\n Rain Forest Classifier MODEL INFO:")
print("\naccuracy:", accuracy_rf )
print("\nprecision_rf:", precision_rf )
print("\nf1_rf:", f1_rf )

    #SVM model

accuracy_svm = accuracy_score(test_y, svm_model_predictions)
precision_svm = precision_score(test_y, svm_model_predictions, average='weighted',zero_division=0)
f1_svm = f1_score(test_y, svm_model_predictions, average='weighted')
print("\n\n Support Vector Machines MODEL INFO:")
print("\naccuracy:", accuracy_svm )
print("\nprecision_rf:", precision_svm )
print("\nf1_rf:", f1_svm )


    #LR model

accuracy_lm = accuracy_score(test_y, linreg_model_predictions)
precision_lm = precision_score(test_y, linreg_model_predictions, average='weighted',zero_division=0)
f1_lm = f1_score(test_y, linreg_model_predictions, average='weighted')
print("\n\n Linear Regression MODEL INFO:")
print("\naccuracy:", accuracy_lm)
print("\nprecision_rf:", precision_lm )
print("\nf1_rf:", f1_lm )

# Based of the scores from accuracy, presicion and F1, Random forest has 
# the highest score across the board. Score is ranged from 0 to 1, where 0 means
# least while 1 means most in terms of all categories. 



    # Confusion matrix

from sklearn.metrics import  confusion_matrix

conf_matrix = confusion_matrix(test_y, rf_model_predictions)

step_labels = ['1', '2', '3','4','5','6','7', '8', '9','10','11','12','13'] 

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=step_labels, yticklabels=step_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Rainforest Classifier Model')
plt.show()



# 2.6 Train the model with Rainforest classifier 


import joblib

joblib.dump(best_rf_model, 'final_maintenace_model.joblib')
final_model = joblib.load('final_maintenace_model.joblib')

test_coordinates = [
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0, 3.0625, 1.93],
    [9.4, 3, 1.8],
    [9.4, 3, 1.3]
]

predicted_steps = final_model.predict(test_coordinates)

print ("\n The predicted maintenace steps for given coordinates are:\n", predicted_steps)



