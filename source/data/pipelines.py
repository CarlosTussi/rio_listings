from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


from data.pipeline_classes import *

'''
    *  *


'''
num_feat_extraction_pipeline = Pipeline([
    ("clusterlocation", ClusterGeolocationTransformer(clusters = 25, init = "random", n_init = 15, max_iter = 1000)),
])


'''
    *  *


'''
num_feat_extraction_pipeline_app = Pipeline([
    ("clusterlocation", ClusterGeolocationTransformer(clusters = 25, init = "random", n_init = 15, max_iter = 1000)),
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
                                               words = ["lux","luxurious","luxury","fancy","garage", 
                                                  "hydromassage", "cellar", "sophistication", 
                                                  "magnificent", "colonial", "rooftop", "triplex", "suite"])), 
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
                                               words = ["lux","luxurious","luxury","fancy","garage", 
                                                  "hydromassage", "cellar", "sophistication", 
                                                  "magnificent", "colonial", "rooftop", "triplex", "suite"])), 
    
    ("extractamenities", ExtractAmenitiesTransformer(
                                                { "parking": ".*parking on premises.*",
                                                  "pool":".*pool.*(?!.*\btable\b).*",
                                                  "washer": ".*washer.*",
                                                  "dishwasher": ".*dishwasher.*",
                                                 "ceiling_fan" : ".*ceiling fan.*",
                                                 "long_term" : ".*long term.*",
                                                 "bbq_grill" : ".*bbq grill.*",
                                                 "outdoor": ".*outdoor.*",
                                                 "hot_tub": ".*hot tub.*",
                                                 "bathtub": ".*bathtub.*",
                                                 "ac": [".*air conditioning.*","\\bac\\b"],
                                                 "seaview" : [".*beach view.*",".*sea view.*",".*ocean view.*"]
                                                }
    )),
    
    ("extractbathroom", ExtractBathroom()),
    ("extractscore", ExtractScore())
])



'''
    *  *


'''
data_cleaning_pipeline = Pipeline([   
      
    #Imputers
    ("num_imputer", NumImputer(value = 0)),
    ("cat_imputer", CatImputer(
                            features_limits = [("bathrooms_text", "Private bath"),
                                               ("description", ""),
                                              ])),
    #Outliers
    ("num_outliers", OutlierRemover(
                                features_limit = [
                                                 ("minimum_nights_avg_ntm", 7),
                                                 ("beds", 8),
                                                 ("bedrooms", 5),
                                                 ("bathrooms", 5),
                                                 ("accommodates", 10),
                                                 ("number_of_reviews_ltm", 25),
                                                 ("reviews_per_month", 4),
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

        # Encoding
        #('data_encoding_pipeline', data_encoding_pipeline),
        
        # Feature Extraction
        ("num_feature_extraction_pipeline", num_feat_extraction_pipeline),
        ("cat_feat_extraction_pipeline", cat_feat_extraction_pipeline),   
        
        # Feature Selection
        ('columndrop', ColumnDroppersTransformer()), 

        #Normalisation
        ("normalisation", MinMaxScaler()),    
])