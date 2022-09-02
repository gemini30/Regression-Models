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
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y.reshape(-1,1))


# Fitting SVR to data set
from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf')
svr_reg.fit(x,y)

# Predicting a new result with polynomial regression
y_pred = sc_y.inverse_transform(svr_reg.predict( sc_x.transform(np.array([[6.5]]))))


# Visualising SVR results
plt.scatter(x, y,color = 'red',edgecolors='black')
plt.plot(x,svr_reg.predict(x),color='royalblue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()
