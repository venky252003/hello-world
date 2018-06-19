# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 18:20:51 2018

@author: Venky
"""

import sys as sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()



def trainData(x, y):        
    return train_test_split(x, y, test_size = 1/3, random_state = 0)
    

def regression(xTrain, yTrain):        
    regressor.fit(xTrain, yTrain)    
    return regressor
    

def plot(X_train, y_train):
    plt.scatter(X_train, y_train, color = 'red')
    plt.plot(X_train, regressor.predict(X_train), color = 'blue')
    plt.title('Salary vs Experience (Training set)')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.show()
    
def predit(test):
    y_pred = regressor.predict(test)
    print(y_pred)
    
def main(csvPath):
    try:
        print('Started')
        dataset = pd.read_csv(csvPath) #'Salary_Data.csv'
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, 1].values
        X_train, X_test, y_train, y_test = trainData(X, y)
        regression(X_train, y_train)
        plot(X_train, y_train)        
    except ValueError as e:
        print(e)
    finally:
        print('Completed')
    
if __name__ == '__main__':
    main(sys.argv[1])
    


