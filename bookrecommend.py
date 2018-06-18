# -*- coding: utf-8 -*-
"""
Created on Thu May 17 11:37:40 2018

@author: Venky
"""
import pandas as pd
datafile = 'data/BX-Book-Ratings.csv'
#with open(datafile, 'r', newline='', encoding='utf-8') as csvfile:
      #data=csvfile.read_csv(datafile, sep=';', names=['User-ID','ISBN','Book-Rating'])

data=pd.read_csv(datafile, sep=';', encoding='ISO-8859-1', header=0, names=['User-ID','ISBN','Book-Rating'])