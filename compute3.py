# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 19:52:21 2017

@author: Spider
"""

import pandas as pd
from sklearn.externals import joblib

def recommend_mv(user_id, top_cnt):
    R_hat = joblib.load('data/R_hat-ML-ALS2.pkl')
    R = joblib.load('data/R-ML-ALS2.pkl')
    movies = joblib.load('data/movies.pkl')
    
    #user_id = 6
    #top_cnt = 10
    user_id = user_id - 1
    
    try:
        ratings = pd.DataFrame(data=R.loc[user_id,R.loc[user_id,:] > 0])
        ratings['Prediction'] = R_hat.loc[user_id,R.loc[user_id,:] > 0]
        ratings.reset_index(level=0, inplace=True)
        ratings['index'] = ratings['index'] + 1
        ratings.columns = ['item_id', 'Actual Rating', 'Predicted Rating']
        #ratings
        
        
        predictions = R_hat.loc[user_id,R.loc[user_id,:] == 0] # Predictions for movies that the user 17 hasn't rated yet
        topN = predictions.sort_values(ascending=False).head(n=top_cnt)
        recommendations = pd.DataFrame(data=topN)
        recommendations.reset_index(level=0, inplace=True)
        recommendations.columns = ['item_id', 'Predicted Rating']
        #recommendations
    
    
        movies['item_id'] = pd.to_numeric(movies['item_id'], errors='coerce')
        final_rec = pd.merge(recommendations, movies, on='item_id', how='left')
        #del final_rec['genre']
        final_rec.index += 1
        final_rec = final_rec[['item_id', 'title', 'Predicted Rating']]
        final_rec = final_rec.rename(columns={'item_id': 'ID produktu', 'title': 'Tytuł', 'Predicted Rating': 'Przewidywana ocena'})
        
        return final_rec
    except (KeyError, NameError, ValueError):
        return print("Podaj poprawny ID użytkownika i ilość rekomendacji")


def recommend_bx(user_id, top_cnt):
    R_hat = joblib.load('data/R_hat-BX-ALS.pkl')
    R = joblib.load('data/R-BX-ALS.pkl')
    books = joblib.load('data/books.pkl')
    
    #user_id = 6
    #top_cnt = 10
    user_id = user_id - 1
    
    try:
        ratings = pd.DataFrame(data=R.loc[user_id,R.loc[user_id,:] > 0])
        ratings['Prediction'] = R_hat.loc[user_id,R.loc[user_id,:] > 0]
        ratings.reset_index(level=0, inplace=True)
        ratings['index'] = ratings['index'] + 1
        ratings.columns = ['item_id', 'Actual Rating', 'Predicted Rating']
        #ratings
        
        
        predictions = R_hat.loc[user_id,R.loc[user_id,:] == 0] # Predictions for movies that the user 17 hasn't rated yet
        topN = predictions.sort_values(ascending=False).head(n=top_cnt)
        recommendations = pd.DataFrame(data=topN)
        recommendations.reset_index(level=0, inplace=True)
        recommendations.columns = ['item_id', 'Predicted Rating']
        #recommendations
    
    
        books['item_id'] = pd.to_numeric(books['item_id'], errors='coerce')
        final_rec = pd.merge(recommendations, books, on='item_id', how='left')
        #del final_rec['genre']
        final_rec.index += 1
        final_rec = final_rec[['item_id', 'title', 'Predicted Rating']]
        final_rec = final_rec.rename(columns={'item_id': 'ID produktu', 'title': 'Tytuł', 'Predicted Rating': 'Przewidywana ocena'})
        
        return final_rec
    except (KeyError, NameError, ValueError):
        return print("Podaj poprawny ID użytkownika i ilość rekomendacji")