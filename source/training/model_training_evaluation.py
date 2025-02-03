import pandas as pd
import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor

# Metrics
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score



'''
    * *

    Input:
    -----
        

    Output:
    -------
       
'''
def model_training(X_train, y_train):
    model = HistGradientBoostingRegressor(random_state = 69)
    model.fit(X_train, y_train)

    return model


'''
    * *

    Input:
    -----
        

    Output:
    -------
       
'''

def model_evaluation(model, X_test, y_test):
    
    y_pred = model.predict(X_test)


    # Calclulate the score for the following metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    metrics = [r2, mape, mae, rmse]
    results = pd.DataFrame(metrics, index = ["R2", "MAPE", "MAE", "RMSE"], columns = ["Results"])

    return results