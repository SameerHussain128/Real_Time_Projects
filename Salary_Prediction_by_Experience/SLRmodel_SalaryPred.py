
#import librabries

import numpy as np   # linear algebra
import pandas as pd  # data preprocessing 
import matplotlib.pyplot as plt 

salary = pd.read_csv(r"D:\Data Science 6pm\2 - January\25th\SIMPLE LINEAR REGRESSION\Salary_Data.csv")

x=salary.iloc[:,:-1].values
y=salary.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.5, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()  # parameter tunning
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
y_pred

# New Data Prediction 
regressor.predict([[1]])

regressor.predict([[25]])

regressor.predict([[50]])


#Visualization of Training Set

plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='green')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualization of Testing Set

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='green')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
