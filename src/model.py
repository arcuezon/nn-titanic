import pandas as pd
import numpy as np

from L_NN_model import *


def L_layer_NN(X, Y, layer_dims={}, save_params=True, load_params=False,
               optimizer='', rate_decay=False, learning_rate=0.01,
               epochs=1000, decay_rate=0.1, beta1=0.9, beta2=0.999):

    assert(X.shape[1] == Y.shape[0])  # Check if dims are correct

    # Choose dimension
    if(load_params == False):
        print("Initializing Parameters...")
        #layer_dims = {"n_h": 2, "n_1": 5, "n_2": 3, "y": Y.shape[1], "n_0": X.shape[0]}
        # layer_dims = {"n_h": 1, "n_1": 4,
        #              "y": Y.shape[1], "n_0": X.shape[0]}

        # Initialize parameters
        params = {}
        params = init_param(layer_dims)

    else:
        params = np.load('my_file.npy', allow_pickle='TRUE').item()

    # Intitialize update dicts
    if(optimizer == 'Momentum' or optimizer == 'Adam'):  # Momentum
        vgrads = {}
        L = len(params) // 2

        for l in range(1, L + 1):
            vgrads["dW" + str(l)] = 0
            vgrads["db" + str(l)] = 0
    elif(optimizer == 'RMSprop' or optimizer == 'Adam'):  # RMSprop
        sgrads = {}
        L = len(params) // 2

        for l in range(1, L + 1):
            sgrads["dW" + str(l)] = 0
            sgrads["db" + str(l)] = 0

    for epoch in range(1, epochs):
        # forward prop
        Y_hat, caches = L_forward(X, params)

        # Compute Cost
        J = compute_cost(Y_hat, Y)

        # Back prop
        grads = L_backward(Y_hat, Y, caches)

        # Learning rate decay
        if(rate_decay):
            # learning_rate = np.divide(decay_rate, np.sqrt(epoch)) * alpha
            learning_rate = np.divide(
                1, 1 + decay_rate * epoch) * learning_rate

        ### UPDATE PARAMS ###

        # Momentum
        if(optimizer == 'Momentum'):
            L = len(caches)
            for l in range(1, L + 1):
                vgrads["dW" + str(l)] = beta1 * vgrads["dW" +
                                                       str(l)] + (1 - beta1) * grads["dW" + str(l)]
                vgrads["db" + str(l)] = beta1 * vgrads["db" +
                                                       str(l)] + (1 - beta1) * grads["db" + str(l)]

            params = update_params(params, vgrads, learning_rate=learning_rate)

        # Adam Optimizer
        elif(optimizer == 'Adam'):
            for l in range(1, L + 1):  
                # Momentum update
                vgrads["dW" + str(l)] = beta1 * vgrads["dW" +
                                                       str(l)] + (1 - beta1) * grads["dW" + str(l)]
                vgrads["db" + str(l)] = beta1 * vgrads["db" +
                                                       str(l)] + (1 - beta1) * grads["db" + str(l)]

                #RMSprop update
                sgrads["dW" + str(l)] = beta2 * sgrads["dW" +
                                                       str(l)] + (1 - beta2) * np.square(grads["dW" + str(l)])
                sgrads["db" + str(l)] = beta2 * sgrads["db" +
                                                       str(l)] + (1 - beta2) * np.square(grads["db" + str(l)])

        # No Momentum
        else:
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

    return params
