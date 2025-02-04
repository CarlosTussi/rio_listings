'''
How to run (Windows):

    $env:PYTHONPATH = (Get-Location)


** From the 'rio_listings' folder **

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
    model = joblib.load(MAIN_PRED_MODEL_PATH)

    # Cluster model for coordinates
    geo_cluster_model = joblib.load(GEO_CLUSTER_MODEL_PATH)

    # Data Normalisation model
    normaliser_model = joblib.load(NORM_MODEL_PATH)

    #######
    # GUI #
    #######
    gui(model, geo_cluster_model, normaliser_model)