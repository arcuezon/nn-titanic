#from grad_descent_forward import sigmoid
import numpy as np
from grad_descent_forward import sigmoid

def compute_cost(Y_hat, Y):
    m = Y.shape[0]

    J = np.sum(Y * np.log(Y_hat) - (1 - Y) * np.log(1 - Y_hat)) / -m

    return J

def sigmoid_backward(dA):
    return sigmoid(dA) * (1 - sigmoid(dA))

def relu_backward(dA):
    return max(0, dA)

def linear_backward(dZ, cache):

    A_prev, W, b = cache
    
    m = A_prev.shape[-1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis = 1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def activation_backwards(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == 'relu':
        dZ

def linear_activation_backward(Y_hat, Y, caches):
    grads = {}
    L = len(caches) 

    print(L)

    #Layer L
    grads["dA" + str(L)] = - (Y/Y_hat) + np.divide(1 - Y, 1 - Y_hat)

