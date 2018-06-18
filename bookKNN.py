# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 00:07:22 2018

@author: Venky
"""

import pandas as pd
import numpy as np

#Load Data
book = pd.read_csv('data\BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
book.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher',
       'imageUrlS', 'imageUrlM', 'imageUrlL']

user = pd.read_csv('data\BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
user.columns = ['userID', 'location', 'age']

rating = pd.read_csv('data\BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
rating.columns = ['userID', 'ISBN', 'rating']

#Combine books
combine_book_rating = pd.merge(rating, book, on='ISBN')
columns =  ['yearOfPublication', 'publisher','imageUrlS', 'imageUrlM', 'imageUrlL']
combine_book_rating = combine_book_rating.drop(columns, axis=1)

combine_book_rating = combine_book_rating.dropna(axis=0, subset = ['bookTitle'])

#Get Total Rating based on books
book_rating_count = (combine_book_rating.groupby(by=['bookTitle'])['rating'].count().
                     reset_index().rename(columns={'rating': 'totalRatingCount'})
                     [['bookTitle', 'totalRatingCount']])

book_rating_count.head()

#merge Rating to Combine book and rating data
rating_with_totalRatingCount = combine_book_rating.merge(book_rating_count, left_on='bookTitle', 
                                                         right_on='bookTitle')

pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(book_rating_count['totalRatingCount'].describe())
print(book_rating_count['totalRatingCount'].quantile(np.arange(.9,1,.01)))

#get All rating has more then 50
popularty_threshold = 50
rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularty_threshold')
rating_popular_book.head()


#Combine user and rating book data
combined = rating_popular_book.merge(user, left_on='userID', right_on='userID', how='left')

#select only usa and canada user
us_canada_user_rating = combined[combined['location'].str.contains('usa | canada')]
#drop age
us_canada_user_rating = us_canada_user_rating.drop('age', axis=1)

#KNN Aglorthim
us_canada_user_rating_pivot = us_canada_user_rating.pivot(index='bookTitle', columns='userID'
                                                          , values = 'rating').fillna(0)
