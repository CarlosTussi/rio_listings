'''
    This module contains the functions responsable only for training and evaluation of the model.
'''

import pandas as pd
import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor

# Metrics
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score



'''
    def model_training(X_train, y_train):

        - Train a HistGradientBoostingRegressor model

    Input:
    ******
        - x_train : list(np.array)
        - y_train: np.array

    Output:
    *******
    model = HistGradientBoostingRegressor       # Trained model

'''
def model_training(X_train: list, y_train: np.array) -> HistGradientBoostingRegressor:

    model = HistGradientBoostingRegressor(random_state = 69)
    model.fit(X_train, y_train)

    return model


'''

    def model_evaluation(pred_model, X_test, y_test):

        - Evaluate the mmodel using regression's statistics:
            MAE, RMSE, R2, MAPE

    Input:
    ******
        - pred_model: HistGradientBoostingRegressor
        - X_test : list(np.array)
        - y_test : np.array

    Output:
    *******
        - result : pd.DataFrame
       
'''

def model_evaluation(pred_model : HistGradientBoostingRegressor, 
                    X_test : list, 
                    y_test: np.array) -> pd.DataFrame:
    
    # Predict using test set
    y_pred = pred_model.predict(X_test)


    # Calclulate the score for the following metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    metrics = [r2, mape, mae, rmse]
    results = pd.DataFrame(metrics, index = ["R2", "MAPE", "MAE", "RMSE"], columns = ["Results"])

    return results