#from grad_descent_forward import sigmoid
import numpy as np

def compute_cost(Y_hat, Y):
    m = Y.shape[0]

    J = np.sum(Y * np.log(Y_hat) - (1 - Y) * np.log(1 - Y_hat)) / -m

    return J

def linear_backward():
    pass