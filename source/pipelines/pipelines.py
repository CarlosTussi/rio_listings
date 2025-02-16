from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


from source.pipelines.custom_transformers import *
from source.config import *

'''
    ----------------------------
    num_feat_extraction_pipeline
    ----------------------------

    * Feature extraction for numerical features*

'''
num_feat_extraction_pipeline = Pipeline([
    ("clusterlocation", ClusterGeolocationTransformer(clusters = NUMBER_OF_CLUSTERS_FT, init = "random", n_init = 15, max_iter = 1000)),
])


'''
    ---------------------------------
    num_feat_extraction_pipeline_app
    --------------------------------

    * Feature extraction for numerical features for the application (!) *

    OBS: Same as the original 'num_feat_extraction_pipeline' for the moment for now. Created for consistency purposes. 


'''
num_feat_extraction_pipeline_app = Pipeline([
    ("clusterlocation", ClusterGeolocationTransformer(clusters = NUMBER_OF_CLUSTERS_FT, init = "random", n_init = 15, max_iter = 1000)),
])


'''
    ---------------------------------
    cat_feat_extraction_pipeline_app
    ---------------------------------

    * Feature extraction for categorical features for the application (!) *

    1) Text Preprocess
    2) Feature Extraction relevant for the application (!)

'''
cat_feat_extraction_pipeline_app = Pipeline([
    #Removes capital letters, ponctuation, etc...
    ("preprocesstext", PreprocessCorpus("description")),
     #Creates a new feature based on lux words
    ("extractluxdescription", ContainWordsTransformer(new_feature_name = "contains_lux_description", 
                                               corpus_target = "description",
                                               words = DESCRIPTION_LUXWORDS_FT)), 
])



'''
    ----------------------------
    cat_feat_extraction_pipeline
    ----------------------------

    * Feature extraction for categorical features *

    1) Text Preprocess
    2) Features Extractions


'''
cat_feat_extraction_pipeline = Pipeline([
    #Removes capital letters, ponctuation, etc...
    ("preprocesstext", PreprocessCorpus("description")),
     #Creates a new feature based on lux words (expensive properties)
    ("extractluxdescription", ContainWordsTransformer(new_feature_name = "contains_lux_description", 
                                               corpus_target = "description",
                                               words = DESCRIPTION_LUXWORDS_FT)), 
    # Extract amenities features
    ("extractamenities", ExtractAmenitiesTransformer(AMENITIES_REGEX_FT)),
    # Extract bathroom information feature
    ("extractbathroom", ExtractBathroom()),
    # Extraact score information features
    ("extractscore", ExtractScore())
])



'''
    ----------------------
    data_cleaning_pipeline
    ----------------------

    1) Dealing with NAs
    2) Removing Outliers
    3) Encoding data


'''
data_cleaning_pipeline = Pipeline([   
      
    #Imputers
    ("num_imputer", NumImputer(value = NUM_VAL_IMP)),
    ("cat_imputer", CatImputer(
                            features_replace = CAT_VAL_IMP)),
    #Outliers
    ("num_outliers", OutlierRemover(
                                features_limit = [
                                            ("minimum_nights_avg_ntm", MIN_NIGHTS_LIM_FT),
                                            ("beds", BEDS_LIM_FT),
                                            ("bedrooms", BEDROOMS_LIM_FT),
                                            ("bathrooms", BATHROOM_LIM_FT),
                                            ("accommodates", ACCOMM_LIM_FT),
                                            ("number_of_reviews_ltm", NIGHTS_LTM_LIM_FT),
                                            ("reviews_per_month", REVIEWS_P_MONTH_LIM_FT),
                                                ],
                                mode = "cap",
                                operator = "lt" #operation: less then (lt)
                                
                                )),
    
     ("cat_outliers", OutlierRemover(
                                features_limit = [("room_type", "Hotel room", "Private room")], #Replace 'Hotel room' by 'Private room'
                                mode = "replace",
                                operator = "eq",
    )),
    
    
    #Encoding Feature
    ("cat_encoding", FeatureEncoding(["room_type"]))
])


'''
    -------------------
    preprocess_pipeline
    --------------------

    * Main data preparation pipeline  *

    1) Data cleaning
    2) Feature Extraction
    3) Feature Selection
    4) Scaling

'''
preprocess_pipeline = Pipeline([
        # Outliers, Nas
        ('data_cleaning_pipeline', data_cleaning_pipeline),
        
        # Feature Extraction
        ("num_feature_extraction_pipeline", num_feat_extraction_pipeline),
        ("cat_feat_extraction_pipeline", cat_feat_extraction_pipeline),   
        
        # Feature Selection
        ('columndrop', ColumnDroppersTransformer()), 

        # Scaling
        ("scaler", CustomMinMaxScaler()),    
])