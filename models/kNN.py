# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 18:10:19 2017

@author: Spider
"""

import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
from math import sqrt
import copy
import operator


# MovieLens
header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('C:/Users/Spider/Desktop/Magisterka/MovieLens/ml-100k/u.data', sep='\t', names=header)

header = ['item_id', 'title', 'url']
movies = pd.read_csv('C:/Users/Spider/Desktop/Magisterka/MovieLens/ml-100k/movies.csv', sep='::', names=header, engine='python')


# BookCrossing
header = ['user_id', 'item_id', 'rating']
df = pd.read_csv('C:/Users/Spider/Desktop/Magisterka/Book-Crossing/Zmodyfikowane/bx_book_ratings.csv', sep='::', names=header, engine='python')

header = ['item_id', 'title', 'author', 'year of publication', 'publisher']
books = pd.read_csv('C:/Users/Spider/Desktop/Magisterka/Book-Crossing/Zmodyfikowane/bx_books.csv', sep='::', names=header, engine='python')


n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]

print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))


sparsity = len(df) / (n_users * n_items)
sparsity *= 100
print('Sparsity: {:4.2f}%'.format(sparsity))


train_data, test_data = cv.train_test_split(df,test_size=0.25)

train_data = pd.DataFrame(train_data)
test_data = pd.DataFrame(test_data)
all_data = pd.DataFrame(df)


R = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    R[line[1]-1, line[2]-1] = line[3]  

T = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    T[line[1]-1, line[2]-1] = line[3]
    
A = np.zeros((n_users, n_items))
for line in all_data.itertuples():
    A[line[1]-1, line[2]-1] = line[3]


I = R.copy()
I[I > 0] = 1
I[I == 0] = 0

I2 = T.copy()
I2[I2 > 0] = 1
I2[I2 == 0] = 0


def read_dataframe(dataframe):
    ratings = {}

    for index, row in dataframe.iterrows():
        userId = row['user_id']
        itemId = row['item_id']
        rating = row['rating']
        
        if userId not in ratings:
            ratings[userId] = {}
        ratings[userId][itemId] = float(rating)

    return ratings


test_ratings = read_dataframe(test_data)
training_ratings = read_dataframe(train_data)


class CosineDistancer:
    def __init__(self):
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def combine_users(self, user1, user2):
        user1 = copy.copy(user1)
        user2 = copy.copy(user2)

        all_keys = set(user1.keys()) | set(user2.keys())

        for key in all_keys:
            if key not in user1:
                user1[key] = None
            if key not in user2:
                user2[key] = None

        u1 = []
        u2 = []
        for key in sorted(user1.keys()):
            if user1[key] is not None and user2[key] is not None:
                u1.append(user1[key])
                u2.append(user2[key])

        return u1, u2

    def cosine_distance(self, user1, user2):
        cache_key = '%s:%s' % (user1, user2)
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        else:
            self.cache_misses += 1

        u1, u2 = self.combine_users(user1, user2)

        top = 0
        user1bottom = 0
        user2bottom = 0

        for i in range(0, len(u1)):
            if u1[i] is not None and u2[i] is not None:
                top += u1[i] * u2[i]
                user1bottom += u1[i] * u1[i]
                user2bottom += u2[i] * u2[i]

        bottom = sqrt(user1bottom) * sqrt(user2bottom)
        if bottom == 0:
            self.cache[cache_key] = 0
        else:
            self.cache[cache_key] = top / bottom

        return self.cache[cache_key]

def get_users_with_item(ratings, itemId):
    users = []
    for userId in ratings:
        if itemId in ratings[userId]:
            users.append(userId)

    return users


def calculate_rating_avg(distancer, users, userId, itemId, max_n):
    dist = {}
    for related_user in get_users_with_item(users, itemId):
        if related_user == userId:
            continue

        dist[related_user] = distancer.cosine_distance(users[userId], users[related_user])

    sorted_dist = sorted(dist.items(), key=operator.itemgetter(1), reverse=True)
    total = 0.0
    n = 0
    for i in range(0, max_n):
        try:
            total += users[sorted_dist[i][0]][itemId]
        except:
            break
        n += 1

    if n == 0:
        return 2.5

    return total / n

T_hat = np.zeros((n_users, n_items))
distancer = CosineDistancer()

for user in range(n_users):
    for item in range(n_items):
        T_hat[user][item] = calculate_rating_avg(distancer, test_ratings, user+1,
            item+1, 20)


T_hat = pd.DataFrame(T_hat)
A = pd.DataFrame(A)