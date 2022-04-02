# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 15:22:06 2022

@author: hosam
"""

#Task 1 Prediction using Supervised ML

#Predict the percentage of a student based on study hours
#This is a simple Linear Regression Task which only takes 2 variables
#What will be the predicted score if the student studied for 9.25 hrs/day?!

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('Data.txt')

#Describing the dataset
dataset.describe()
#The dataset consists of 25 records with an average of 5 hours of study time per student and a 51.48% average Score 
#Max hours studied are 9.2*** while we want to predict on 9.25?!
#From what i know it isnt good to use linear regression on values that are higher than the max values in the training set?

dataset.info()
#There are no null values in the dataset


#Using Seaborn Scatter plot to viusalize the relationship between Hours and Scores
sns.scatterplot(data=dataset, x='Hours', y='Scores')
plt.title('Hours Studied vs Scores')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

#From the scatter plot theres a linear relationship between the hours studied and the scores
dataset.corr() #97.6% correlation between hours and Scores


#Splitting the data into Feaures and Labels(Target)
X = dataset['Hours']
y = dataset['Scores']

#The Linear Regression object needs the input to be in an array shape of (-1, 1)or(1,-1) also the .reshape doesnt work
#on series and the .values changes the X and y from series to an array
X_clean = X.values.reshape(-1, 1)
y_clean = y.values.reshape(-1, 1)

#Splitting the data into train/test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=0)

#Building the model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)


#Visualising the Training results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, lr.predict(X_train), color = 'blue')
plt.title('Training Results')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

#Visualising the test results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, lr.predict(X_test), color = 'blue')
plt.title('Test Results')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()


#Prediction accuracy on the train data
y_pred_train = lr.predict(X_train)
from sklearn.metrics import r2_score, mean_absolute_error
print('Train R2 Score:',  '{:.2f}%'.format(r2_score(y_train, y_pred_train)*100))
print('Train MAE:',mean_absolute_error(y_train, y_pred_train))

#Prediction accuracy on the test data
y_pred = lr.predict(X_test)
from sklearn.metrics import r2_score, mean_absolute_error
print('Test R2 Score:',  '{:.2f}%'.format(r2_score(y_test, y_pred)*100))
print('Test MAE:',mean_absolute_error(y_test, y_pred))


#What will be the predicted score if the student studied for 9.25 hrs/day?!
Prediction = lr.predict(np.array([9.25]).reshape(-1,1))
print('If a student studies for 9.25 hrs/day he would probably get a score arround', '{:.2f}%'.format(Prediction[0,0]))