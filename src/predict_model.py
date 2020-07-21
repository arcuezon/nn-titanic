import pandas as pd
import numpy as np

#from data_read import *
from L_NN_model import *


def predict_model(X, PID, params, save_to_csv = True):
    Y_hat, _ = L_forward(X, params)  # Forward

    Y_hat = Y_hat > 0.5
    Y_hat = Y_hat.T

    PID = PID.flatten()
    Y_hat = Y_hat.flatten()
    Y_hat = Y_hat.astype(int)


    predictions = {"PassengerId": PID, "Survived": Y_hat}
    df = pd.DataFrame(predictions)
    print(df.head(10))
    if(save_to_csv):
        df.to_csv(r"../predict.csv")

    return predictions
