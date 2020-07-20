import pandas as pd
import numpy as np
import cupy as cp

#from data_read import *
from L_NN_model import *

save_params = True
load_params = False

grads_momentum = True
rate_decay = True

testno = 50

# Reading data
print("Read data...")
X, Y, stdval_cache = read_data()
X_test = X[:, 0:testno]
X = X[:, testno: ]

Y_test = Y[0:testno, :]
Y = Y[testno:, :]

assert(X.shape[1] == Y.shape[0])  # Check if dims are correct

# Choose dimension
if(load_params == False):
    print("Initializing Parameters...")
    #layer_dims = {"n_h": 2, "n_1": 5, "n_2": 3, "y": Y.shape[1], "n_0": X.shape[0]}
    layer_dims = {"n_h" : 1, "n_1" : 4, "y" : Y.shape[1], "n_0" : X.shape[0]}

    # Initialize parameters
    params = {}
    params = init_param(layer_dims)

else:
    params = np.load('my_file.npy',allow_pickle='TRUE').item()

if(grads_momentum):
    vgrads = {}
    L = len(params) // 2

    for l in range(1, L + 1):
        vgrads["dW" + str(l)] = 0
        vgrads["db" + str(l)] = 0

for epoch in range(1, 800):
    # forward prop
    Y_hat, caches = L_forward(X, params)

    # Compute Cost
    J = compute_cost(Y_hat, Y)

    # Back prop
    grads = L_backward(Y_hat, Y, caches)

    #Learning rate decay
    if(rate_decay):
        alpha = 1.9
        decay_rate = 0.01
        #learning_rate = np.divide(decay_rate, np.sqrt(epoch)) * alpha
        learning_rate = np.divide(1, 1 + decay_rate * epoch) * alpha

    else:
        learning_rate = 0.5

    # Update params
    #Momentum
    if(grads_momentum):
        beta = 0.9
        L = len(caches)
        for l in range(1, L + 1):
            vgrads["dW" + str(l)] = beta * vgrads["dW" + str(l)] + (1 - beta) * grads["dW" + str(l)]
            vgrads["db" + str(l)] = beta * vgrads["db" + str(l)] + (1 - beta) * grads["db" + str(l)]

        params = update_params(params, vgrads, learning_rate=learning_rate)
    #No Momentum
    else:
        params = update_params(params, grads, learning_rate=learning_rate)

    if(epoch % 10 == 0):
        print("Epoch:", epoch, end='')
        print(" Cost =", J)
        print("Learning rate =", learning_rate)

print("J =", J)

Y_hat = Y_hat > 0.5
Y = Y.T == 1
Y_hat = (Y == Y_hat)
m = Y_hat.shape[1]
print(np.sum(Y_hat) / m)

if(save_params):
    print("Saving parameters to file...")
    # Save
    np.save('test.npy', params) 
