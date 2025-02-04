from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


from source.pipelines.custom_transformers import *
from source.config import *

'''
    *  *


'''
num_feat_extraction_pipeline = Pipeline([
    ("clusterlocation", ClusterGeolocationTransformer(clusters = NUMBER_OF_CLUSTERS_FT, init = "random", n_init = 15, max_iter = 1000)),
])


'''
    *  *


'''
num_feat_extraction_pipeline_app = Pipeline([
    ("clusterlocation", ClusterGeolocationTransformer(clusters = NUMBER_OF_CLUSTERS_FT, init = "random", n_init = 15, max_iter = 1000)),
])


'''
    *  *


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
    *  *


'''
cat_feat_extraction_pipeline = Pipeline([
    #Removes capital letters, ponctuation, etc...
    ("preprocesstext", PreprocessCorpus("description")),
     #Creates a new feature based on lux words
    ("extractluxdescription", ContainWordsTransformer(new_feature_name = "contains_lux_description", 
                                               corpus_target = "description",
                                               words = DESCRIPTION_LUXWORDS_FT)), 
    
    ("extractamenities", ExtractAmenitiesTransformer(AMENITIES_REGEX_FT)),
    
    ("extractbathroom", ExtractBathroom()),
    ("extractscore", ExtractScore())
])



'''
    *  *


'''
data_cleaning_pipeline = Pipeline([   
      
    #Imputers
    ("num_imputer", NumImputer(value = NUM_VAL_IMP)),
    ("cat_imputer", CatImputer(
                            features_limits = CAT_VAL_IMP)),
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
    * Main pipeline  *

'''
preprocess_pipeline = Pipeline([
        # Outliers, Nas
        ('data_cleaning_pipeline', data_cleaning_pipeline),
        
        # Feature Extraction
        ("num_feature_extraction_pipeline", num_feat_extraction_pipeline),
        ("cat_feat_extraction_pipeline", cat_feat_extraction_pipeline),   
        
        # Feature Selection
        ('columndrop', ColumnDroppersTransformer()), 

        #Normalisation
        ("normalisation", CustomMinMaxScaler()),    
])