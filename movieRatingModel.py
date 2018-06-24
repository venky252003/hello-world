# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 15:08:03 2018

@author: Venky
"""

import numpy as np
import pandas as pd

dataFile = pd.read_csv('data/u.data', sep="\t", header=None, 
                       names=['userId', 'itemId', 'rating', 'timestamp'])
movieFile = pd.read_csv('data/u.item', sep="|", header=None,  error_bad_lines=False, encoding="latin-1", index_col=False,
                       names=['itemId', 'title'], usecols=[0,1])

data = pd.merge(dataFile, movieFile, left_on="itemId", right_on="itemId")

data = pd.DataFrame.sort_values(data, ['userId', 'itemId'],ascending=[0,1])

numUsers = max(data.userId)
numMovies = max(data.itemId)

moviePerUser = data.userId.value_counts()
userPerMovie = data.itemId.value_counts()

def favouriteMovie(activeUser, N):
    topMovies = pd.DataFrame.sort_values(data[data.userId == activeUser], ['rating'], ascending=[0])[:N]
    return list(topMovies.title)

userItemRatingMatrix = pd.pivot_table(data, values='rating', 
                                      index=['userId'], columns=['itemId'])

from scipy.spatial.distance import correlation

def similarity(user1, user2):
    user1 = np.array(user1) - np.nanmean(user1)
    user2 = np.array(user2) - np.nanmean(user2)
    commonItemId=[i for i in range(len(user1)) if user1[i]>0 and user2[i]>0]
    
    if len(commonItemId) ==0:
        return 0
    else:
        user1 = np.array([user1[i] for i in commonItemId])
        user2 = np.array([user2[i] for i in commonItemId])
        
    return correlation(user1, user2)
    
def nearestNeighbourRating(activeUser, K):
    similartyMatrix = pd.DataFrame(index = userItemRatingMatrix.index, columns=['similarity'])

    for i in userItemRatingMatrix.index:        
        similartyMatrix.loc[i] = similarity(userItemRatingMatrix.loc[activeUser], 
                                            userItemRatingMatrix.loc[i])
        
    similartyMatrix = pd.DataFrame.sort_values(similartyMatrix, ['similarity'], ascending=[0])
    nearestNeighbour = similartyMatrix[:K]
    neighbourItemRating = userItemRatingMatrix.loc[nearestNeighbour.index]
    predictItemrRating = pd.DataFrame(index=userItemRatingMatrix.columns, columns=['Rating'])
    
    for i in userItemRatingMatrix.columns:
        predictedRating= np.nanmean(userItemRatingMatrix.loc[activeUser])
        for j in userItemRatingMatrix.index:
            if userItemRatingMatrix.loc[j,i]>0:
                predictedRating += (userItemRatingMatrix.loc[j,i] - np.nanmean(userItemRatingMatrix.loc[j])) * nearestNeighbour.loc[j, 'similarity']
                
        predictItemrRating.loc[i,'Rating'] = predictedRating
        
    return predictItemrRating

def topNRecommendtion(activeUser, N):
    predicitItemRating = nearestNeighbourRating(activeUser, N)
    moviesAlreadyWatched = list(userItemRatingMatrix.loc[activeUser].loc[userItemRatingMatrix.loc[activeUser] > 0].index)
    predicitItemRating = predicitItemRating.drop(moviesAlreadyWatched)
    topRecommendations = pd.DataFrame.sort_values(predicitItemRating, ['Rating'], ascending=[0])[:N]
    topRecommendationTitle = (movieFile.loc[movieFile.itemId.isin(topRecommendations.index)])
    
    return list(topRecommendationTitle)

user = 5
favouriteMovie(user, 5)
topNRecommendtion(user, 5)
    
                
                