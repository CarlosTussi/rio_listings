'''
How to run (Windows):

streamlit run .\app\main.py

'''



import joblib
from app.gui import *

if __name__ == "__main__":
    
    ###############
    # LOAD MODELS #
    ###############
    # Main price prediction model
    model = joblib.load("../models/price_model.joblib")

    # Cluster model for coordinates
    geo_cluster_model = joblib.load("../models/geo_kmeans_model.joblib")

    # Data Normalisation model
    normaliser_model = joblib.load("../models/normaliser_model.joblib")

    #######
    # GUI #
    #######
    gui(model, geo_cluster_model, normaliser_model)