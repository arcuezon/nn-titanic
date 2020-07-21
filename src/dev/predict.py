import pandas as pd
import numpy as np
import cupy as cp

#from data_read import *
from L_NN_model import *

testno = 50


def read_test_data():
    in_data = pd.read_csv("../data/test.csv")

    in_data.fillna('0', inplace=True)

    gender = {'male': 1, 'female': 2}
    in_data.Sex = [gender[item] for item in in_data.Sex]

    port = {'C': 1, 'Q': 2, 'S': 3, '0': 0}
    in_data.Embarked = [port[item] for item in in_data.Embarked]

    drop_col = ['Cabin', 'Name', 'Ticket']
    in_data.drop(drop_col, inplace=True, axis=1)

    in_data.Age = pd.to_numeric(in_data.Age)
    in_data.Fare = pd.to_numeric(in_data.Fare)

    X = in_data.to_numpy()

    PID = X[:, 0]
    X = X[:, 1:].T
    
    return X, PID


# Reading data
print("Read data...")
X, Y, stdval_cache = read_data()
X_test = X[:, 0:testno]
X = X[:, testno:]

Y_test = Y[0:testno, :]
Y = Y[testno:, :]

assert(X.shape[1] == Y.shape[0])  # Check if dims are correct

# Load trained params
params = np.load('test.npy', allow_pickle='TRUE').item()

mu = stdval_cache["mu"]
sigma = stdval_cache["sigma"]

X, PID = read_test_data()

X = (X - mu) / sigma

Y_hat, caches = L_forward(X, params)  # Forward

PID = PID.reshape((PID.shape[0], 1))

Y_hat = Y_hat > 0.5
Y_hat = Y_hat.T

PID = PID.flatten()
Y_hat = Y_hat.flatten()
Y_hat = Y_hat.astype(int)


predictions = {"PassengerId": PID, "Survived": Y_hat}
df = pd.DataFrame(predictions)
print(df.head(10))
df.to_csv(r"../predict.csv")
