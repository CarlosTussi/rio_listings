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