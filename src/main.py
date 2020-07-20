import pandas as pd
import numpy as np

from data_read import *
from init_net import *
from grad_descent_forward import *
from grad_descent_back import *

#Reading data
print("Read data...")
X, Y, stdval_cache = read_data()
assert(X.shape[1] == Y.shape[0]) #Check if dims are correct

#Choose dimension
print("Initializing Parameters...")
layer_dims = {"n_h" : 2, "n_1" : 5, "n_2" : 3, "y" : Y.shape[1], "n_0" : X.shape[0]}

#Initialize parameters
params = {}
params = init_param(layer_dims)
print(len(params))
#check_params(params)

#forward prop
Y_hat, caches = L_forward(X, params)

#Compute Cost
J = compute_cost(Y_hat, Y)
print(J)

#Back prop
linear_activation_backward(Y_hat, Y, caches)

