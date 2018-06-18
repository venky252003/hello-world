# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 22:14:36 2018

@author: Venky
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
import matplotlib.pyplot as plt
#%matplotlib inline
sns.set_style('whitegrid')

df = sns.load_dataset('iris')
X_train = df[['petal_length', 'petal_width']]

species_to_num = {'setosa': 0, 'versicolor': 1, 'virginica': 2 }
df['species'] = df['species'].map(species_to_num)
y_train = df['species']

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors= 2)
knn.fit(X_train, y_train)

xv = X_train.values.reshape(-1,1)
h = 0.02
x_min, x_max = xv.min(), xv.max() + 1
y_min, y_max = y_train.min(), y_train.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
fig = plt.figure(figsize=(8,5))
ax = plt.contourf(xx, yy, z, cmap = 'afmhot', alpha=0.3)
plt.scatter(X_train.values[:, 0], X_train.values[:, 1], c=y_train, s=40, alpha=0.9, edgecolors='k')

#knn.predict([6.7, 2])

