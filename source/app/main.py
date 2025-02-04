'''
    *  Starting point for the application *

    
    * How to run (Windows):   
     (!) From the 'rio_listings' folder.

        $env:PYTHONPATH = (Get-Location)
        streamlit run .\source\app\main.py

'''

import joblib
from source.app.gui import *
from source.config import *

if __name__ == "__main__":
    
    ###############
    # LOAD MODELS #
    ###############
    # Main price prediction model
    price_pred_model = joblib.load(MAIN_PRED_MODEL_PATH)

    # Cluster model for coordinates
    geo_cluster_pred_model = joblib.load(GEO_CLUSTER_MODEL_PATH)

    # Data Normalisation model
    scaler_transf_model = joblib.load(SCALER_MODEL_PATH)

    #######
    # GUI #
    #######
    gui(price_pred_model, geo_cluster_pred_model, scaler_transf_model)