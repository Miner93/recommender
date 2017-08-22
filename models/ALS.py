import numpy as np
import pandas as pd
from sklearn import cross_validation as cv


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


R = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    R[line[1]-1, line[2]-1] = line[3]  

T = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    T[line[1]-1, line[2]-1] = line[3]


I = R.copy()
I[I > 0] = 1
I[I == 0] = 0

I2 = T.copy()
I2[I2 > 0] = 1
I2[I2 == 0] = 0


def rmse(I,R,Q,P):
    return np.sqrt(np.sum((I * (R - np.dot(P.T,Q)))**2)/len(R[R > 0]))


lmbda = 0.1
k = 20
m, n = R.shape
n_epochs = 15

P = 3 * np.random.rand(k,m)
Q = 3 * np.random.rand(k,n)
Q[0,:] = R[R != 0].mean(axis=0)
E = np.eye(k)


train_errors = []
test_errors = []


def als_step(Ii, Q, Ri, lmbda, E):
    nui = np.count_nonzero(Ii)
    if (nui == 0): nui = 1

    Ai = np.dot(Q, np.multiply(Q.T, Ii)) + lmbda * nui * E
    Vi = np.dot(Q, np.multiply(Ri, Ii))
    return np.linalg.solve(Ai,Vi).T[0]


def als():
    for epoch in range(n_epochs):
        for i, Ii in enumerate(I):
            P[:,i] = als_step(I[i:i+1,:].T, Q, R[i:i+1,:].T, lmbda, E)
    
        for j, Ij in enumerate(I.T):
            Q[:,j] = als_step(I[:,j:j+1], P, R[:,j:j+1], lmbda, E)
    
        train_rmse = rmse(I,R,Q,P)
        test_rmse = rmse(I2,T,Q,P)
        train_errors.append(train_rmse)
        test_errors.append(test_rmse)
        
        print("[Epoch %d/%d] train error: %f, test error: %f" \
        %(epoch+1, n_epochs, train_rmse, test_rmse))
        
    print("Algorithm converged")

als()


R_hat = pd.DataFrame(np.dot(P.T,Q))
R = pd.DataFrame(R)