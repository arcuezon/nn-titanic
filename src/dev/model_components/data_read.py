import pandas as pd
import cupy as cp
import numpy as np


def std_data(X):
    mu = cp.mean(X, axis=1, keepdims=True)
    sigma = cp.std(X, axis=1, keepdims=True)

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

    X = cp.asarray(X)
    Y = cp.asarray(Y)

    return X, Y, stdval_cache
