import numpy as np
import pandas as pd

def std_data(X):
    mu = np.mean(X, axis=1, keepdims=True)
    sigma = np.std(X, axis=1, keepdims=True)

    X = (X - mu) / sigma

    stdval_cache = {"mu": mu, "sigma": sigma}

    return X, stdval_cache


def read_data():
    in_data = pd.read_csv("../data/train.csv")

    in_data.fillna('0', inplace=True)

    gender = {'male': 1, 'female': 2}
    in_data.Sex = [gender[item] for item in in_data.Sex]

    port = {'C': 1, 'Q': 2, 'S': 3, '0': 0}
    in_data.Embarked = [port[item] for item in in_data.Embarked]

    drop_col = ['PassengerId', 'Cabin', 'Name', 'Ticket']
    in_data.drop(drop_col, inplace=True, axis=1)

    in_data.Age = pd.to_numeric(in_data.Age)

    data = in_data.to_numpy()

    Y = data[:, 0]
    X = np.delete(data, 0, axis=1)

    X = X.T
    Y = Y.reshape((Y.shape[0], 1))

    X, stdval_cache = std_data(X)

    return X, Y, stdval_cache

### INITIALIZE NETWORK ###

def init_param(layer_dims):
    params = {}

    # Set hidden layers and units
    n_h = layer_dims["n_h"]
    assert(len(layer_dims) == 3 + n_h)
    print("# of hidden layers = " + str(n_h))

    for l in range(1, n_h + 1):
        n = layer_dims["n_" + str(l)]
        params["W" + str(l)] = np.random.randn(n, layer_dims["n_" + str(l - 1)]) * 0.01
        params["b" + str(l)] = np.zeros((n, 1))

    L = n_h + 1
    params["W" + str(L)] = np.random.randn(layer_dims["y"], layer_dims["n_" + str(L - 1)]) * 0.01
    params["b" + str(L)] = np.zeros((layer_dims["y"], 1))

    return params

def check_params(params):
    L = len(params) // 2

    for l in range(1, L + 1):
        curW = "W" + str(l)
        curb = "b" + str(l)
        print(curW + " shape: " + str(params[curW].shape))
        print(curb + " shape: " + str(params[curb].shape))

### FORWARD PROPAGATION ###

def sigmoid(z):
    g = np.divide(1, (1 + np.exp(-z)))
    return g


def relu(z):
    g = np.maximum(0, z)
    return g


def linear_forward(A_prev, W, b):
    Z = np.add(np.dot(W, A_prev), + b)

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

### COST FUNCTION ###

def compute_cost(Y_hat, Y):
    Y = Y.T
    m = Y.shape[1]

    assert(Y_hat.shape == Y.shape)

    J = np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat)) / -m

    return J


### BACKWARD PROPAGATION ###

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

