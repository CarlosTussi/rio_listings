from source.pipelines import pipelines
from sklearn.pipeline import Pipeline


'''

'''
def preprocess_data(X, geo_cluster_model, normaliser_model):
    # Pre-process data pipeline
        preprocess_data = Pipeline([
            ('cat_feat_extraction_pipeline_app', pipelines.cat_feat_extraction_pipeline_app),  
            ('geo_cluster_app', pipelines.GeoClusterPredict(geo_cluster_model)),
            ('normaliser_app', pipelines.CustomMinMaxScalerAppTransformer(normaliser_model)),
        ])

        X = preprocess_data.fit_transform(X)

        return X


'''

'''
def process_and_predict(df_model_input, model, geo_cluster_model, normaliser_model):
        ###################
        # Preprocess Data #
        ###################
        X = preprocess_data(df_model_input, geo_cluster_model, normaliser_model)

        ####################
        # Model Prediction #
        ####################
        predicted_price = model.predict(X)

        return predicted_price[0]