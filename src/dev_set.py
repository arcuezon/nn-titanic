import pandas as pd
import numpy as np
import cupy as cp

#from data_read import *
from L_NN_model import *

testno = 50

# Reading data
print("Read data...")
X, Y, stdval_cache = read_data()
X_test = X[:, 0:testno]
X = X[:, testno:]

Y_test = Y[0:testno, :]
Y = Y[testno:, :]

assert(X.shape[1] == Y.shape[0])  # Check if dims are correct

params = np.load('test.npy', allow_pickle='TRUE').item() #Load trained params

Y_hat, caches = L_forward(X_test, params) #Forward

Y_hat = Y_hat > 0.5
Y_test = Y_test.T == 1
Y_hat = (Y_test == Y_hat)
m = Y_hat.shape[1]
print(np.sum(Y_hat) / m)