# from grad_descent_forward import sigmoid
import numpy as np
from grad_descent_forward import sigmoid


def compute_cost(Y_hat, Y):
    Y = Y.T
    m = Y.shape[1]

    assert(Y_hat.shape == Y.shape)

    J = np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat)) / -m

    return J


def sigmoid_backward(dA, activation_cache):
    Z = activation_cache

    dZ = (sigmoid(Z) * (1 - sigmoid(Z))) * dA
    return dZ


def relu_backward(dA, activation_cache):
    Z = activation_cache

    Z = Z > 0

    dZ = Z * dA
    return dZ


def linear_backward(dZ, cache):

    A_prev, W, b = cache

    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def activation_backwards(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_backward(Y_hat, Y, caches):
    grads = {}
    L = len(caches)

    #Reshape Y to (1, m) like Y_hat.shape
    Y = Y.T
    assert(Y.shape == Y_hat.shape)

    # Layer L
    grads["dA" + str(L)] = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))
    current_cache = caches[L - 1] #L - 1 indexing
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = \
        activation_backwards(grads["dA" + str(L)], current_cache, activation = "sigmoid")

    # For layers 1 to L-1
    for l in range(L - 1, 0, -1):
        #print("Backprop Layer", l)
        current_cache = caches[l - 1] #Indexing
        grads["dA" + str(l - 1)], grads["dW" + str(l)], grads["db" + str(l)] = \
            activation_backwards(grads["dA" + str(l)], current_cache, activation = "relu")

    return grads

def update_params(params, grads, learning_rate = 0.01):
    L = len(params) // 2

    for l in range(1, L + 1):
        params["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        params["b" + str(l)] -= learning_rate * grads["db" + str(l)]
    

    return params

