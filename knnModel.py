# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 19:42:28 2018

@author: Venky
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier

from sklearn.grid_search import GridSearchCV

from sklearn.svm import SVC

from sklearn.model_selection import  cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def LoadData():
    dataset = pd.read_csv('Social_Network_Ads.csv')
    X = dataset.iloc[:,[2,3]].values
    y = dataset.iloc[:, 4].values
    return X,y

def Train(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state = 0)
    return X_train, X_test, y_train, y_test

def Tranform(X_train, X_test):
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test  = sc_X.fit_transform(X_test)
    return X_train, X_test

def Classfier(cls, X_train, y_train):    
    cls.fit(X_train, y_train)
    return cls 

def predict(x, cls):
    #Predicting 
    y_pred = cls.predict(x)
    return y_pred

def confusion(test, pred):
    #Confusion matrix
    cm = confusion_matrix(test, pred)
    print(cm)

#Visual the training
def DrawChart(cls, x_set, y_set, title=''):
    X1, X2 = np.meshgrid(np.arange(start = x_set[:,0].min() - 1, stop = x_set[:,0].max() + 1, step = 0.01),
                         np.arange(start = x_set[:,1].min() - 1, stop = x_set[:,1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, cls.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap=ListedColormap(('red','green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i,j in enumerate(np.unique(y_set)):
        plt.scatter(x_set[y_set==j, 0], x_set[y_set == j, 1], 
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title(title)
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()
    
def print_score(clf, X_train, y_train, X_test, y_test, train = True):
      if train:
          print('Train Result:\n')
          print('Accuracy Score: {0:.4f}\n'.format(accuracy_score(y_train, clf.predict(X_train))))
          print('Classfication Report:\n {}\n'.format(classification_report(y_train, clf.predict(X_train))))
          print('Confusion Matrix: \n {} \n'.format(confusion_matrix(y_train, clf.predict(X_train))))
          
          res = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
          print('Average Accuray: \t {0:.4f}'.format(np.mean(res)))
          print('Accuracy SD: \t\t {0:.4f}'.format(np.std(res)))
        
      elif train == False:            
          print('Test Result:\n')
          print('Accuracy Score: {0:.4f}\n'.format(accuracy_score(y_test, clf.predict(X_test))))
          print('Classfication Report:\n {}\n'.format(classification_report(y_test, clf.predict(X_test))))
          print('Confusion Matrix: \n {} \n'.format(confusion_matrix(y_test, clf.predict(X_test))))

    
def main():
    X, y = LoadData()
    X_train, X_test, y_train, y_test = Train(X,y)
    X_train, X_test = Tranform(X_train, X_test)    
    #classfier = SVC(kernel = 'linear', random_state = 0)    
    #classfier = KNeighborsClassifier(n_neighbors=9, metric='minkowski', p = 2)
    classfier = Classfier(classfier, X_train, y_train)    
    y_pred = predict(X_test, classfier)
    confusion(y_test, y_pred)
    DrawChart(classfier, X_train, y_train, title='Logistic Regression (Train)')
    #DrawChart(classfier, X_test, y_test, title='Logistic Regression (Test)')
    
   
    params = {'n_neighbors' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    grid_search = GridSearchCV(classfier, params, n_jobs=1, verbose=1)
    grid_search.fit(X_train, y_train)
    grid_search.best_estimator_
    print_score(grid_search, X_train, y_train, X_test, y_test, train=True)
    print('Best Neighbour {0}'.format(grid_search.best_params_))
    



  


