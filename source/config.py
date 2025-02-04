'''
    This file contains all the constants and configs used in the project fro all the features.

'''
########################
#  Features Specifics  #
########################

#Target Feature
PRICE_CAP_FT = 1000


#Amenities
AMENITIES_REGEX_FT = { "parking": ".*parking on premises.*",
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
# Description
DESCRIPTION_LUXWORDS_FT = ["lux","luxurious","luxury","fancy","garage", 
                        "hydromassage", "cellar", "sophistication", 
                        "magnificent", "colonial", "rooftop", "triplex", "suite"]


# Capacity limits:
# (accommodates)
ACCOMM_LIM_FT = 10 
# (bathrooms)
BATHROOM_LIM_FT = 5
# (beds)
BEDS_LIM_FT = 8
# (bedrooms)
BEDROOMS_LIM_FT = 5
# (minimum_nights_avg_ntm)
NIGHTS_LTM_LIM_FT = 25
# (reviews_per_month)
REVIEWS_P_MONTH_LIM_FT = 4
# (minimum_nights_avg_ntm)
MIN_NIGHTS_LIM_FT = 7





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
                    "number_of_reviews_l30d", "number_of_reviews", "availability_30","availability_60","availability_90", 
                    "minimum_nights", "maximum_nights", "review_scores_value", "review_scores_accuracy", 
                    "review_scores_rating", "review_scores_checkin", "review_scores_cleanliness", "review_scores_communication",
                    "has_availability", "instant_bookable", "calendar_last_scraped", 'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights', 
                    'maximum_maximum_nights', 'maximum_nights_avg_ntm', 'property_type']

# Extra features not relevant that contain the following patterns
FEATURES_TO_DROP_REGEX = ".*host.*|.*id.*|.*url.*"





################
#  File paths  #
################

# Input dataset filepath
TRAINING_DATA_PATH = 'data/listings.csv'

# Main prediction model filepath
MAIN_PRED_MODEL_PATH = 'models/price_pred_model.joblib'

# Geo-cluster kmeans model filepath
GEO_CLUSTER_MODEL_PATH = 'models/geo_kmeans_pred_model.joblib'

# Scaler model filepath
SCALER_MODEL_PATH = 'models/scaler_transf_model.joblib'






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
                ("Hot tub", "has_hot_tub"),
                ("AC", "has_ac"),
                ("Seaview", "has_seaview"),
                ]