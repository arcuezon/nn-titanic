import pandas as pd
import numpy as np
import cupy as cp

#from data_read import *
from L_NN_model import *

save_params = True

# Reading data
print("Read data...")
X, Y, stdval_cache = read_data()
assert(X.shape[1] == Y.shape[0])  # Check if dims are correct

# Choose dimension
print("Initializing Parameters...")
#layer_dims = {"n_h": 2, "n_1": 5, "n_2": 3, "y": Y.shape[1], "n_0": X.shape[0]}
layer_dims = {"n_h" : 1, "n_1" : 5, "y" : Y.shape[1], "n_0" : X.shape[0]}

# Initialize parameters
params = {}
params = init_param(layer_dims)

for epoch in range(1, 500000):
    # forward prop
    Y_hat, caches = L_forward(X, params)

    # Compute Cost
    J = compute_cost(Y_hat, Y)

    # Back prop
    grads = L_backward(Y_hat, Y, caches)

    alpha = 2
    learning_rate = np.divide(1, np.sqrt(epoch)) * alpha
    # Update params
    params = update_params(params, grads, learning_rate=learning_rate)

    if(epoch % 1000 == 0):
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
    np.save('my_file.npy', params) 

""" # Load
read_dictionary = np.load('my_file.npy',allow_pickle='TRUE').item()
print(read_dictionary['hello']) # displays "world" """