import numpy as np


def sigmoid(z):
    g = np.divide(1, (1 + np.exp(-z)))
    return g


def relu(z):
    g = np.maximum(0, z)
    return g


def linear_forward(A_prev, W, b):
    Z = np.dot(W, A_prev) + b

    linear_cache = (A_prev, W, b)
    return Z, linear_cache


def activation(Z, activation="sigmoid"):

    if activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "relu":
        A = relu(Z)

    activation_cache = (Z)

    return A, activation_cache


def L_forward(X, params):

    n_h = len(params) // 2
    caches = []

    A_prev = X

    for l in range(1, n_h):
        W = params["W" + str(l)]
        b = params["b" + str(l)]

        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = activation(Z, activation="relu")
        A_prev = A
        caches.append((linear_cache, activation_cache))
        #print("Layer " + str(l + 1))
    W = params["W" + str(l + 1)]
    b = params["b" + str(l + 1)]
    Z, linear_cache = linear_forward(A_prev, W, b)
    Y, activation_cache = activation(Z, activation="sigmoid")
    caches.append((linear_cache, activation_cache))

    return Y, caches
