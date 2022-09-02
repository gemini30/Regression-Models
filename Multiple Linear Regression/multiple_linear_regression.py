import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the data set
dataset  = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
#Encode Country Column
# labelencoder_X = LabelEncoder()
# x[:,0] = labelencoder_X.fit_transform(x[:,0])
ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = 'passthrough')
x = ct.fit_transform(x)

# Avoiding dummy variable trap
x= x[:,1:]

#Splitting dataset into training set and test set
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# Feature Scaling
''' from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test) '''

#Fitting mulitple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

# Predicting Test set results
y_pred = regressor.predict(x_test)

# Building optimal model using backward elimination
import statsmodels.api as sm
x = np.append(arr=np.ones((50,1)).astype(int),values = x,axis = 1)
x_opt = np.array(x[:, [0, 1, 2, 3, 4, 5]], dtype=float)
regressor_OLS = sm.OLS(endog=y,exog = x_opt).fit()
regressor_OLS.summary()

# Remove predictor with highest p-val
x_opt = np.array(x[:, [0, 1, 3, 4, 5]], dtype=float)
regressor_OLS = sm.OLS(endog=y,exog = x_opt).fit()
regressor_OLS.summary()

x_opt = np.array(x[:, [0, 3, 4, 5]], dtype=float)
regressor_OLS = sm.OLS(endog=y,exog = x_opt).fit()
regressor_OLS.summary()

x_opt = np.array(x[:, [0, 3, 5]], dtype=float)
regressor_OLS = sm.OLS(endog=y,exog = x_opt).fit()
regressor_OLS.summary()


x_opt = np.array(x[:, [0, 3]], dtype=float)
regressor_OLS = sm.OLS(endog=y,exog = x_opt).fit()
regressor_OLS.summary()











