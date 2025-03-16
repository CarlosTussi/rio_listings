'''
    *  Starting point for the application *

    
    * How to run (Windows):   
     (!) From the 'rio_listings' folder.

        $env:PYTHONPATH = (Get-Location)
        streamlit run .\source\app\main.py

'''
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, PROJECT_ROOT) 

import joblib
import time
from source.app.gui import *
from source.config import *




if __name__ == "__main__":
    
    ###############
    # LOAD MODELS #
    ###############
    # Main price prediction model

    # Define the absolute path for the model
    price_pred_model = joblib.load(MAIN_PRED_MODEL_PATH)

    # Cluster model for coordinates
    geo_cluster_pred_model = joblib.load(GEO_CLUSTER_MODEL_PATH)

    # Data Normalisation model
    scaler_transf_model = joblib.load(SCALER_MODEL_PATH)

    price_model_name = type(price_pred_model).__name__
    price_model_last_modified = time.ctime(os.path.getmtime(MAIN_PRED_MODEL_PATH))

    #######
    # GUI #
    #######
    gui(price_pred_model, 
        geo_cluster_pred_model, 
        scaler_transf_model, 
        price_model_name, 
        price_model_last_modified)