import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline


from source.pipelines.pipelines import preprocess_pipeline
from source.config import *


'''
    def pipeline_preparation(df : pd.DataFrame, target : str) -> tuple

        - This function preprocess the target feature only and separate it from the training dataset.

    Input:
    *****
        df : pd.DataFrame
        target : str

    Output:
    ******
        tuple: (pd.DataFrame, np.Series)

'''
def pipeline_preparation(df : pd.DataFrame, target : str) -> tuple:


    # Dropping 'na' for prices
    df = df.dropna(subset = target)

    # Convert prices to numerical value
    df.loc[:,target] = df[target].apply(lambda x: float(x[1:].replace(",","")) if pd.notna(x) else x)

    # Cap price outliers
    df.loc[:,target] = df[target].apply(lambda x : x if x < PRICE_CAP_FT else PRICE_CAP_FT)

    # Drop prices that are zero
    df = df.loc[df.price > 0, :]

    # Separte target feature from the dataset
    X = df.drop(target, axis = 1)
    y = df[target]

    return (X, y)


'''
    def preprocess(df : pd.DataFrame, target: str) -> tuple:
    
        - This functions receives the target feature and the input dataset and transforms the data to be fit to a model.

    Input:
    ******
       - df: pd.DataFrame
       - target: str

    Output:
    *******
       - tuple: (pd.DataFrame, np.Series)

'''
def preprocess(df : pd.DataFrame, target: str) -> tuple:

    # Filter target value and separate from dataset
    X, y = pipeline_preparation(df, target)

    # Exercutes data preprocessing pipeline
    X_t = preprocess_pipeline.fit_transform(X)

    return X_t, y