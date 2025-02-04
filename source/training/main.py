'''
How to run (Windows):

** From the 'rio_listings' folder **

    python -m training.main

'''

import pandas as pd
import numpy as np

import joblib

import training.data_preprocess as pre
import training.model_training_evaluation as mdl

from source.config import * 


from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    # Read the data
    df = pd.read_csv(TRAINING_DATA_PATH, sep = ",")

    # Preprocess Data Piepline
    X, y = pre.preprocess(df, target = "price")

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 70)

    # Model Training
    price_pred_model = mdl.model_training(X_train, y_train)


    # Model Evaluation
    results = mdl.model_evaluation(price_pred_model, X_test, y_test)

    print(results)

    # Save Model
    joblib.dump(price_pred_model, MAIN_PRED_MODEL_PATH)