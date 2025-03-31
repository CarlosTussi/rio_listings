from source.pipelines import pipelines
from sklearn.pipeline import Pipeline


'''
    def preprocess_data(X, geo_cluster_pred_model, scaler_transf_model)
    -------------------------------------------------------------------

    - Runs the preprocessing input data pipeline for the app.

    
    - Input:
      ******
        - X : pd.DataFrame
        - geo_cluster_pred_model: KMeans()       # Trained model
        - scaler_transf_model: MaxMinScaler()    # Trained model

    - Output:
      *******
        - X: list(np.array)

'''
def preprocess_data(X, geo_cluster_pred_model, scaler_transf_model):
    # Pre-process data pipeline
        preprocess_data = Pipeline([
            ('cat_feat_extraction_pipeline_app', pipelines.cat_feat_extraction_pipeline_app),  
            ('geo_cluster_app', pipelines.GeoClusterPredict(geo_cluster_pred_model)),
            ('scaler_app', pipelines.CustomTrainedMinMaxScalerTransformer(scaler_transf_model)),
        ])

        X = preprocess_data.fit_transform(X)

        return X


'''
    def process_and_predict(df_model_input, price_pred_model, geo_cluster_pred_model, scaler_transf_model)
    ------------------------------------------------------------------------------------------------------

    - Centralizes the preprocessing of input data and prediction.

    - Input:
      ******
        - df_model_input
        - price_pred_model: HistGradientBoostingRegressor               # Trained model
        - geo_cluster_pred_model: KMeans()                              # Trained model
        - scaler_transf_model: MaxMinScaler()                           # Trained model

    - Output:
      ******
        - predicted_price[0]: float

'''
def process_and_predict(df_model_input, price_pred_model, geo_cluster_pred_model, scaler_transf_model):
        ###################
        # Preprocess Data #
        ###################
        X = preprocess_data(df_model_input, geo_cluster_pred_model, scaler_transf_model)

        ####################
        # Model Prediction #
        ####################
        predicted_price = price_pred_model.predict(X)

        # Returns the final predicted price
        return predicted_price[0]