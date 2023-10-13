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
    
x = df['X']
y = df['Y']
z = df['Z']
step = df['Step']



    # Scatter plot 
    
fig1= plt.figure()

ax1= fig1.add_subplot(111, projection='3d')
ax1.scatter(x[step], y[step], z[step])
ax1.set_xlabel('X')
ax1.set_ylabel('Y')    
ax1.set_zlabel('Z')
ax1.set_title('Coordinates plot for maintinance steps') # check spelling 
 
plt.show()

