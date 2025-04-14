import pandas as pd
import numpy as np

# Sample data set with missing values
X_missing = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, 4, 5],
    'C': [1, 2, 3, np.nan, 5],
    'D': [1, 2, 3, 4, 5]
})

# Sample data with corpus to be processed.
X_corpus = pd.DataFrame({
    'description': [
        "Newly decorated penthouse at Copacabana beach. Incredible seaviews. Balcony with jacuzzi/hottub.",
        "SMALL but CoZy 34 square meters!!!",
        "Very close to the ocean =) ",
        "Check my accommodation website for more details: http://www.myhouse.rio-de-janeiro.com.",
        "Please contact directly the owner <br> ...after paying.",
        "     From the amazing Ipanema beach, you are 10min from the bakery.   "
    ]
})

X_processed_corpus = pd.DataFrame({
    'description': [
        "newly decorated penthouse copacabana beach incredible seaviews balcony jacuzzi hottub",
        "small but cozy square meters",
        "very close ocean",
        "check my accommodation website more details http www myhouse rio janeiro com",
        "please contact directly owner after paying",
        "amazing ipanema beach min bakery"
    ]
})

# lux words being considered ["luxurious","fancy", "sophistication", "rooftop"]
X_lux =  pd.DataFrame({ 
    'description': [
        "newly luxurious decorated fancy penthouse copacabana beach incredible seaviews balcony jacuzzi hottub",
        "small but amazing house touch sophistication square meters",
        "rooftop very close ocean",
        "check my accommodation website more details http www myhouse rio janeiro com",
        "please contact directly owner after paying",
        "amazing ipanema beach min bakery"
    ]
})
# 1: if lux word present 
# 0: if lux word not present
X_processed_lux = pd.Series([1,1,1,0,0,0]) 

X_geolocation = pd.DataFrame({
    "latitude": [
        -22.9701, -22.9685, -22.9723, -22.9667, -22.9738,  
        -22.9842, -22.9887, -22.9815, -22.9871, -22.9856, 
        -23.0123, -23.0089, -23.0156, -23.0044, -23.0182   
    ],
    "longitude": [
        -43.1856, -43.1821, -43.1883, -43.1809, -43.1867,  
        -43.2001, -43.1987, -43.2023, -43.1965, -43.2042,  
        -43.3601, -43.3587, -43.3625, -43.3559, -43.3642
    ]
})