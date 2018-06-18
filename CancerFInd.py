# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 05:59:27 2018

@author: Venky
"""

import numpy as np
import pandas as pd

"""
   Attribute Information: (class attribute has been moved to last column)

   #  Attribute                     Domain
   -- -----------------------------------------
   1. Sample code number            id number
   2. Clump Thickness               1 - 10
   3. Uniformity of Cell Size       1 - 10
   4. Uniformity of Cell Shape      1 - 10
   5. Marginal Adhesion             1 - 10
   6. Single Epithelial Cell Size   1 - 10
   7. Bare Nuclei                   1 - 10
   8. Bland Chromatin               1 - 10
   9. Normal Nucleoli               1 - 10
  10. Mitoses                       1 - 10
  11. Class:                        (2 for benign, 4 for malignant) """
  
  col = ['id', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 
         'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
         'Normal Nucleoli', 'Mitoses', 'Class']
  dataset = pd.read_csv('breast-cancer-wisconsin.data.csv', names=col, header=None)
  
  dataset['Bare Nuclei'].value_counts()     #Show value in counts
  
  dataset['Bare Nuclei'].replace('?', np.NAN, inplace=True) #replace ? t NAN
  dataset = dataset.dropna()          #drop NAN
  
  dataset['Class'].value_counts()
  
  dataset['Class'] = dataset['Class']/2-1       #replace 2 to 0 and 4 to 1
  
  X = dataset.drop(['id', 'Class'], axis = 1)   #X train COlumn values
  x_col = X.columns
  
  y = dataset['Class']                          #Y Perdit values
  
  from sklearn.preprocessing import StandardScaler
  X = StandardScaler().fit_transform(X.values)      #Scalar value
  
  from sklearn.model_selection import train_test_split
  dataset1 = pd.DataFrame(X, columns = x_col)
  
  X_train, X_test, y_train, y_test = train_test_split(dataset1, y, train_size=0.8, random_state=5)
  
  from sklearn.preprocessing import MinMaxScaler
  pd.DataFrame(MinMaxScaler().fit_transform(dataset.drop(['id', 'Class'], axis=1).values), columns=x_col).head()

  from sklearn.neighbors import KNeighborsClassifier
  knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
  knn.fit(X_train, y_train)
  
  from sklearn.model_selection import  cross_val_predict, cross_val_score
  from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
  
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

  
  print_score(knn, X_train, y_train, X_test, y_test, train=False)
  
  params = {'n_neighbors' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
  
  from sklearn.grid_search import GridSearchCV
  grid_search = GridSearchCV(knn, params, n_jobs=1, verbose=1)
  grid_search.fit(X_train, y_train)
  grid_search.best_estimator_
  print_score(grid_search, X_train, y_train, X_test, y_test, train=True)
  grid_search.best_params_
  
  
  
