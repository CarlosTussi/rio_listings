import pandas as pd
import numpy as np

import joblib

import data.preprocess as pre
import models.model as mdl


from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    # Read the data
    df = pd.read_csv("../Data/listings.csv", sep = ",")

    # Preprocess Data Piepline
    X, y = pre.preprocess(df, target = "price")

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 70)

    # Model Training
    model = mdl.model_training(X_train, y_train)


    # Model Evaluation
    results = mdl.model_evaluation(model, X_test, y_test)

    print(results)

    # Save Model
    joblib.dump(model, 'model.joblib')