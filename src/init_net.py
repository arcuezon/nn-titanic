# Function to initialize parameters W, b with randn values
import numpy as np

def init_param(layer_dims):
    params = {}

    # Set hidden layers and units
    n_h = layer_dims["n_h"]
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
    
    for l in range(1, 4):
        curW = "W" + str(l)
        curb = "b" + str(l)
        print(curW + " shape: " + str(params[curW].shape))
        print(curb + " shape: " + str(params[curb].shape))