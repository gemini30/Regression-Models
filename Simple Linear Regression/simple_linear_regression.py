# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the data set
dataset  = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#Splitting dataset into training set and test set
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)

# Feature Scaling
''' from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test) '''

# Fitting simple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

# Predicting test set results
y_pred = regressor.predict(x_test)

# Visualising the training set values
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train),color='royalblue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the test set values
plt.scatter(x_test, y_test, color = 'brown')
plt.plot(x_train, regressor.predict(x_train),color='hotpink')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

