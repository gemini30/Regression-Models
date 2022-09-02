import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the data set
dataset  = pd.read_csv('Position_Salaries.csv')
# Creating matrix of features
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values


#Splitting dataset into training set and test set
'''from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)'''

# Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc_x = StandardScaler()
# sc_y = StandardScaler()
# x = sc_x.fit_transform(x)
# y = sc_y.fit_transform(y.reshape(-1,1))


# Fitting Random Forest regression to data set
# Create regressor
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300,random_state=0) 
regressor.fit(x,y)

# Predicting a new result with Random Forest regression
y_pred = regressor.predict([[6.5]])


# Visualising the Random Forest regression results
x_grid = np.arange(min(x),max(x),0.01)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x, y,color = 'red',edgecolors='black')
plt.plot(x_grid,regressor.predict(x_grid),color='royalblue')
plt.title('Truth or Bluff (Random Forest regression)')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()

