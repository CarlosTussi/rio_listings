'''
    This file contains all the constants and configs used in the project fro all the features.

'''
########################
#  Features Specifics  #
########################

#Target Feature
PRICE_CAP_FT = 850


#Amenities
AMENITIES_REGEX_FT = { "parking": ".*parking on premises.*",
                       "pool":".*pool.*(?!.*\btable\b).*",
                       "washer": ".*washer.*",
                       "dishwasher": ".*dishwasher.*",
                       "ceiling_fan" : ".*ceiling fan.*",
                       "long_term" : ".*long term.*",
                       "bbq_grill" : ".*bbq grill.*",
                       "outdoor": ".*outdoor.*",
                       "jacuzzi": ".*hot tub.*",
                       "bathtub": ".*bathtub.*",
                       "ac": [".*air conditioning.*","\\bac\\b"],
                       "seaview" : [".*beach view.*",".*sea view.*",".*ocean view.*"]
                     }
# Description
DESCRIPTION_LUXWORDS_FT = ["lux","luxurious","luxury","fancy","garage", 
                        "hydromassage", "cellar", "sophistication", 
                        "magnificent", "colonial", "rooftop", "triplex", "suite"]#,pool]    


# Capacity limits:
# (accommodates)
ACCOMM_LIM_FT = 10 
# (bathrooms)
BATHROOM_LIM_FT = 5
# (beds)
BEDS_LIM_FT = 8
# (bedrooms)
BEDROOMS_LIM_FT = 5
# (number_of_reviews_ltm)
NIGHTS_LTM_LIM_FT = 25
# (reviews_per_month)
REVIEWS_P_MONTH_LIM_FT = 4
# (minimum_nights_avg_ntm)
MIN_NIGHTS_LIM_FT = 5





#####################
#  Model Specifics  #
#####################

# Geolocation
NUMBER_OF_CLUSTERS_FT = 25

# Imputers:
# Numeric value imputers
NUM_VAL_IMP = 0

# Categorical value imputers
CAT_VAL_IMP = [("bathrooms_text", "Private bath"),
               ("description", ""),
              ]


# Not Relevant Features

FEATURES_TO_DROP = ["calendar_updated", "license", "neighbourhood_group_cleansed","neighbourhood_cleansed", "neighbourhood", 
                    "neighborhood_overview", "last_scraped", "source", "first_review", "last_review", "name", 
                    "number_of_reviews_l30d", "number_of_reviews", "availability_30",
                    "minimum_nights", "maximum_nights", "review_scores_accuracy", "review_scores_location", 
                    "review_scores_rating", "review_scores_checkin", "review_scores_cleanliness", "review_scores_communication",
                    "has_availability", "instant_bookable", "calendar_last_scraped", 'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights', 
                    'maximum_maximum_nights', 'maximum_nights_avg_ntm', 'property_type', 'description', 'is_score_empty']

# Extra features not relevant that contain the following patterns
FEATURES_TO_DROP_REGEX = ".*host.*|.*id.*|.*url.*"


#Grid Search Parameters
GRID_SEARCH_PARAM = {
                    'loss': ["poisson", "squared_error"],
                    'max_iter': [150,200,300],
                    'learning_rate': [0.05, 1],
                    'max_leaf_nodes': [71,81,91],
                    'max_features': [0.15,0.2,0.25]
}
#Grid Search Cross-Validation
GRID_SEARCH_CV = 3


################
#  File paths  #
################

import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

# Main prediction model filepath
MAIN_PRED_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "price_pred_model.joblib")

# Geo-cluster kmeans model filepath
GEO_CLUSTER_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "geo_kmeans_pred_model.joblib")

# Scaler model filepath
SCALER_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "scaler_transf_model.joblib")


# Input dataset filepath
TRAINING_DATA_PATH = 'data/listings.csv'


###################
#  GUI Specifics  #
###################

RIO_COORDINATES = [-22.970294234, -43.18559545]
AMENITIES_GUI = [
                ("Parking", "has_parking"), 
                ("Pool", "has_pool"),
                ("Washer","has_washer"),
                ("Dishwasher", "has_dishwasher"),
                ("Ceiling Fan", "has_ceiling_fan"),
                ("Long Term Stay", "has_long_term"),
                ("BBQ Grill", "has_bbq_grill"),
                ("Outdoor Area", "has_outdoor"),
                ("Bathtub", "has_bathtub"),
                ("Jacuzzi", "has_jacuzzi"),
                ("AC", "has_ac"),
                ("Seaview", "has_seaview"),
                ]

DEFAULT_INPUT = {  
                'accommodates' : 1,
                'bathrooms' : 1,
                'bedrooms': 1,
                'beds' : 1,
                'minimum_nights_avg_ntm': 1,
                'availability_60' : 1,
                'availability_90' : 1,
                'availability_365' : 1,
                'number_of_reviews_ltm' : 0,
                'review_scores_value': 0,
                'reviews_per_month': 0,
                'has_parking': 0,
                'has_pool': 0, 
                'has_washer': 0,
                'has_dishwasher': 0, 
                'has_ceiling_fan': 0, 
                'has_long_term': 0, 
                'has_bbq_grill': 0,
                'has_outdoor': 0, 
                'has_jacuzzi': 0, 
                'has_bathtub': 0, 
                'has_ac': 0, 
                'has_seaview': 0,
                'is_bathroom_shared': 0
}