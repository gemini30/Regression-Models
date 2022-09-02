import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the data set
dataset  = pd.read_csv('position_salaries.csv')
# Creating matrix of features
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values


#Splitting dataset into training set and test set
# from sklearn.model_selection import train_test_split 
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# Feature Scaling
''' from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test) '''

# Fitting linear regression to data set
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

# Fitting polynomial regression to data set
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)


# Visualising linear regression model
plt.scatter(x, y,color = 'red',edgecolors='black')
plt.plot(x,lin_reg.predict(x),color='royalblue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()


# Visualising ploynomial regression model
x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x, y,color = 'red',edgecolors='black')
plt.plot(x_grid,lin_reg2.predict(poly_reg.fit_transform(x_grid)),color='royalblue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()


# Predicting a new result with linear regression
lin_reg.predict([[6.5]])

# Predicting a new result with polynomial regression
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))




