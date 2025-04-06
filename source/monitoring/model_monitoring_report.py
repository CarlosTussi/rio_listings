'''

    * This module is responsible for analysing two different datasets and generating a report to assist in data drift detection.

    * Two data sets are required: a reference dataset and a new dataset that we want to compare with.
        - Reference dataset: Automatically uses the training dataset that was use for the current trained model.
        - New dataset: programatically fetched from a given URL provided as an input for this module.

    * How to run:
        * From inside rio_listing folder execute:

        python -m  source.monitoring.main [URL WITH THE NEW DATASET]

    * Improtant: The new dataset needs to conform with the training dataset, that is, it must have the same features as the original dataset.
                - For more details, check this project README at https://github.com/CarlosTussi/rio_listings



'''


import pandas as pd
import numpy as np
import sys
import requests
import os
import io
import gzip
import webbrowser


from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import (
    RegressionQualityMetric,
    RegressionPredictedVsActualScatter,
    RegressionErrorPlot,
    RegressionAbsPercentageErrorPlot,
    RegressionErrorDistribution,
    RegressionErrorNormality,
    RegressionTopErrorMetric,
    RegressionErrorBiasTable,
)

import source.training.data_preprocess as pre
import joblib


from source.config import * 
from source.monitoring.process_data_monitoring import *

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, PROJECT_ROOT) 

'''
    def setup_drif_report(X_ref, y_ref, X_new, y_new, prediction_ref, prediction_new):
    
        - This function generates an HTML report comparing the performance 
        of the two datasets (reference and new/current dataset) with the current model.
    Input:
    ******
       - X_ref: Reference dataset
       - y_ref: Reference target values
       - X_new: New dataset
       - y_new: New target values
       - prediction_ref: Prediction values for the reference dataset
       - prediction_new: Predicition values for the new dataset

    Output:
    *******
       - None

'''
def setup_drif_report(X_ref, y_ref, X_new, y_new, prediction_ref, prediction_new):
    ###################
    # SETUP EVIDENTLY # 
    ###################
    # Prepare the reference data and current (new) data for comparison with Evidently
    # Reference Data
    reference_df = X_ref.copy()
    reference_df["price"] = y_ref
    reference_df["prediction"] = prediction_ref

    # Current (new) Data
    new_df = X_new.copy()
    new_df["price"] = y_new
    new_df["prediction"] = prediction_new

    # Prepare the mapping
    column_mapping = ColumnMapping()
    column_mapping.target = "price"
    # Defining which feature contains the predicted values for both reference and current data
    column_mapping.prediction = "prediction"
    column_mapping.numerical_features = list(X_ref._get_numeric_data().columns)
    column_mapping.categorical_features = [] # Considering all features numerical after transformation. (For testing purposes)



    #######################
    # GENERATE THE REPORT #
    #######################
    regression_performance = Report(metrics=[
                                            RegressionQualityMetric(),
                                            RegressionPredictedVsActualScatter(),
                                            RegressionErrorPlot(),
                                            RegressionAbsPercentageErrorPlot(),
                                            RegressionErrorDistribution(),
                                            RegressionErrorNormality(),
                                            RegressionTopErrorMetric(),
                                            RegressionErrorBiasTable(),
                                        ]) #Not a time series dataset
    regression_performance.run(current_data=new_df, 
                            reference_data=reference_df,   
                            column_mapping=column_mapping)

    # Save HTML file and display in browser
    output_file = "source/monitoring/regression_performance_report.html"
    regression_performance.save_html(output_file)   
    webbrowser.open(('file://' + os.path.realpath(output_file)))    
    

'''
    def retrieve_datasets(reference_dataset_location, new_dataset_location):
    
        - This function retrieves the two datasets needed: the training dataset (localy) and the new dataset (fetched remotely)

    Input:
    ******
       - reference_dataset_location : str
       - new_dataset_location : str             #URL with remote dataset.

    Output:
    *******
       - reference_raw_data : DataFrame     # Not processed
       - new_raw_data : DataFrame           # Not processed


'''
def retrieve_datasets(reference_dataset_location, new_dataset_location):

    # Retrieve reference data. (Training Data)
    reference_raw_data = pd.read_csv(reference_dataset_location, sep = ",")


    # Download the new data
    new_data_zip = requests.get(new_dataset_location).content
    with gzip.open(io.BytesIO(new_data_zip)) as csv_file:
        new_raw_data = pd.read_csv(csv_file)

    return reference_raw_data, new_raw_data


    
'''
    def process_data(reference_raw_data, new_raw_data):
    
        - This function processes both datasets conforming and using the same pipelines defined during training.

    Input:
    ******
       - reference_raw_data : DataFrame     # Not processed
       - new_raw_data                       # Not processed

    Output:
    *******
       - X_ref : DataFrame              # Proccessed Reference Dataset Input Features
       - y_ref : DataFrame              # Proccessed Reference Dataset Target Feature   
       - X_new : DataFrame              # Proccessed New Dataset Input Features
       - y_new : DataFrame              # Proccessed New Dataset Target Feature


'''
def process_data(reference_raw_data, new_raw_data, geo_cluster_pred_model, scaler_transf_model):
    X_ref, y_ref = preprocess(reference_raw_data, "price", geo_cluster_pred_model, scaler_transf_model) 
    X_new, y_new = preprocess(new_raw_data, "price", geo_cluster_pred_model, scaler_transf_model) 

    return X_ref, y_ref, X_new, y_new

'''
    def model_monitoring_report(data_url_input):
    
        - This function orchestrates the model monitoring report, loading the trained models and callign all the necessary functions.

    Input:
    ******
       - data_url_input : str

    Output:
    *******
       - None


'''
def model_monitoring_report(data_url_input):
    
    ###############
    # LOAD MODELS #
    ###############
    # Define the absolute path for the model
    price_pred_model = joblib.load(MAIN_PRED_MODEL_PATH)
    # Cluster model for coordinates
    geo_cluster_pred_model = joblib.load(GEO_CLUSTER_MODEL_PATH)
    # Data Normalisation model
    scaler_transf_model = joblib.load(SCALER_MODEL_PATH)
    


    #########################
    # RETRIEVE RAW DATASETS #
    #########################
    print("Retrieving datasets...")
    reference_raw_data, new_raw_data = retrieve_datasets(TRAINING_DATA_PATH, data_url_input)
    print("Datasets retrieved!")

    #######################
    # PREPROCESS DATASETS #
    #######################
    X_ref, y_ref, X_new, y_new = process_data(reference_raw_data, new_raw_data,geo_cluster_pred_model,scaler_transf_model)


    #####################################
    # PREDICT WITH REF AND NEW DATASETS #
    #####################################
    # Predict reference dataset
    prediction_ref = price_pred_model.predict(X_ref)
    # Predict new dataset
    prediction_new = price_pred_model.predict(X_new)

    #########################
    # GENERATE DRIFT REPORT #
    #########################
    setup_drif_report(X_ref, y_ref, X_new, y_new, prediction_ref, prediction_new)