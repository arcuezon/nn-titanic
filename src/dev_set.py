import pandas as pd
import numpy as np

#from data_read import *
from L_NN_model import *

# Reading data

def test_params(X, Y, params):

    assert(X.shape[1] == Y.shape[0])  # Check if dims are correct

    Y_hat, _ = L_forward(X, params) #Forward

    Y_hat = Y_hat > 0.5
    Y = Y.T == 1
    Y_hat = (Y == Y_hat)
    m = Y_hat.shape[1]
    print("Accuracy", np.sum(Y_hat) / m)