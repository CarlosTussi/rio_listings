import pandas as pd
import numpy as np

import operator as op
import re

# Custom Transformers
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.cluster import KMeans


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


'''
        Class 
        ----------
            *Attributes:
                
               
            
            *:
  
'''
class ColumnDroppersTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def transform(self, X, y=None):
        
        print("Start - ColumnDroppersTransformer")
        
        # Find features related to host, id and url
        re_drop = ".*host.*|.*id.*|.*url.*"
        drop_feat = []
        for feat in X.columns:
            if re.match(re_drop, feat):
                 drop_feat.append(feat)
        # Add extra features that for sure will not ber part of the model
        drop_feat.extend(["calendar_updated", "license", "neighbourhood_group_cleansed","neighbourhood_cleansed", "neighbourhood", "neighborhood_overview", 
                          "last_scraped", "source", "first_review", "last_review", "name", "number_of_reviews_l30d", "number_of_reviews",
                          "availability_30","availability_60","availability_90", "minimum_nights", "maximum_nights", "review_scores_value",  
                         "review_scores_accuracy", "review_scores_rating", "review_scores_checkin", "review_scores_cleanliness", "review_scores_communication",
                         "has_availability", "instant_bookable", "calendar_last_scraped", 'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights', 
                          'maximum_maximum_nights', 'maximum_nights_avg_ntm', 'property_type'])
        
        X = X.drop(drop_feat, axis = 1)
        
        print("End - ColumnDroppersTransformer")
        
        print(X.columns)
        #print(X.room_type.unique())
        return X
    
    def fit(self, X, y=None):
        return self
    


'''
        Class 
        ----------
            *Attributes:
                
               
            
            *:
  
'''
class DropNasTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features
    
    def transform(self, X, y=None):
        
        print("Start - DropNasTransformer")
        
        
        X = X.dropna(subset = self.features)
        
        print(type(X))
        print("End - DropNasTransformer")
        
        
        return X
    
    def fit(self, X):
        
        
        return self
    




'''
        Class 
        ----------
            *Attributes:
                
               
            
            *:
  
'''
class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, features_limit, mode = "remove", operator = "eq"): #st: smaller then
                    
        self.opt_dict = {
        "lt":op.lt,
        "le":op.le,
        "eq":op.eq,
        "ne":op.ne,
        "ge":op.ge,
        "gt":op.gt,
        }    
        
        self.features_limit = features_limit # list of tuples
        self.mode = mode  #mode of operation (cap value or remove value)
        self.operator = operator #lt, eq, gt, ...
     
    
    def transform(self, X, y=None):
        
        print("Start - OutlierRemover")

        
        
        NAME = 0
        LIMIT = 1
        REPLACE_VALUE = 2
        for a_feature in self.features_limit:
            if(self.mode == "remove"): ###!!! check if it works
                X = X.drop(X[self.opt_dict[self.operator](X[a_feature[NAME]], a_feature[LIMIT])].index, axis = 0)
            elif(self.mode == "cap"):
                X[a_feature[NAME]] =  X[a_feature[NAME]].apply(lambda x : x if self.opt_dict[self.operator](x,a_feature[LIMIT]) else a_feature[LIMIT])
            elif(self.mode == "replace"):
                X[a_feature[NAME]] =  X[a_feature[NAME]].apply(lambda x : a_feature[REPLACE_VALUE] if self.opt_dict[self.operator](x,a_feature[LIMIT]) else x)
        

        #print(X.room_type.unique())
        #input()
        
        print("End - OutlierRemover")
        
        
        return X
    
    def fit(self, X):
        return self
        
        


'''
        Class 
        ----------
            *Attributes:
                
               
            
            *:
  
'''
class ClusterGeolocationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, clusters = 8, init = 'k-means++', n_init = 'auto', max_iter = 300):
        
        self.clusters = clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
    
    def transform(self, X, y=None):
        
        print("Start - ClusterGeolocationTransformer")
        
        # Initialize the model
        k_means = KMeans(
            init = self.init,
            n_clusters = self.clusters,
            n_init = self.n_init,
            max_iter = self.max_iter,
            random_state = 69
        )


        #Select the data to be clustered
        df_kmeans = X.loc[:,["latitude", "longitude"]]

        # Fitting
        k_means.fit(df_kmeans)

        #Clusters as a feature
        X['geo_cluster'] = k_means.labels_ 
        #X['geo_cluster'] = X['geo_cluster'].astype("str") # Probably not needed (!) - checkc

        ## Encode here
        
        onehot_encoder_cluster = OneHotEncoder(sparse_output = False, feature_name_combiner='concat')
        series_cluster_onehot = onehot_encoder_cluster.fit_transform(X[["geo_cluster"]])
        encoded_cat_str = [str(x) for x in onehot_encoder_cluster.categories_[0]]
        X[encoded_cat_str] = series_cluster_onehot
        X = X.drop("geo_cluster", axis = 1)
    
        
        #Drop latitude and longitude as they will not be fed into the model
        X = X.drop(["longitude", "latitude"], axis = 1)
        
        
        print("End - ClusterGeolocationTransformer")
        
        return X
    
    def fit(self, X):
        return self




'''
        Class 
        ----------
            *Attributes:
                
               
            
            *:
  
'''
class PreprocessCorpus(BaseEstimator, TransformerMixin):
    def __init__(self, corpus_feature):
        self.corpus_feature = corpus_feature

    
    def transform(self, X, y=None):
        
        print("Start - PreprocessCorpus")
        
        
        def process_corpus(corpus):
            
            corpus = corpus.lower()
            corpus = re.sub(r'[^a-zA-Z ]', ' ', corpus)
            corpus = re.sub(r'\b(the|and|in|with|to|a|of|br|is|from|for|on|this|you|it|has|all|at|de|by|br)\b', ' ', corpus)
            corpus = re.sub(r'br\b', '', corpus)
            return corpus
    
        X[self.corpus_feature] = X[self.corpus_feature].apply(lambda x : process_corpus(x))
        
        print("End - PreprocessCorpus")
        
        return X
    
    def fit(self, X):
        return self    



'''
        Class 
        ----------
            *Attributes:
                
               
            
            *:
  
'''
class ContainWordsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, words, new_feature_name, corpus_target):
        self.words = words
        self.new_feature_name = new_feature_name
        self.corpus_target = corpus_target
    
    def transform(self, X, y=None):
        
        print("Start - ContainWordsTransformer")
        
        
        
        def check_lux_corpus(x):
            
            # Transform list of words in regex
            words_str = ""
            for i in self.words:
                words_str = words_str + i + "|"
            #remove last "|"
            words_str =  words_str[:-1] 
            regex = r'\b('+words_str+r')\b'
            
            
            # Return if description contains lux word or not
            if (re.search(regex, x)):
                return 1
            else:
                return 0
        
        #Create the feature that identify a property as luxurious
        X[self.new_feature_name] = X[self.corpus_target].apply(check_lux_corpus)
        
        # Drop corpus target feature
        X = X.drop(self.corpus_target, axis = 1)
        
        print("End - ContainWordsTransformer")
        
        return X
    
    def fit(self, X):
        return self
    
    


'''
        Class 
        ----------
            *Attributes:
                
               
            
            *:
  
'''
class ExtractAmenitiesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, amenities_dict):
        self.amenities_dict = amenities_dict
    
    def transform(self, X, y=None):
        
        print("Start - ExtractAmenitiesTransformer")
        
        #This function will check the presence of the amenities in the list using regex
        def convert_amenities(amenities, index): 
            for amn_name, amn_re in self.amenities_dict.items():
                if(isinstance(amn_re,list)):
                    # Check each reg from the list of reg for an specific amenity. Ex ("seaview" : [".*beach view.*",".*sea view.*",".*ocean view.*"])
                    for reg in amn_re:
                        if(re.match(reg, str.lower(amenities))):
                            X.loc[index, "has_"+ amn_name] = 1
                else:
                    if(re.match(amn_re, str.lower(amenities))):
                        X.loc[index,"has_"+ amn_name] = 1
        
        # 1) Create and initalize the features from the amenities list
        has_amn_feat = ["has_" + x for x in self.amenities_dict.keys()]
        X[has_amn_feat] = 0
        
        # 2) For each amenities' list, check if they contain the amenities target
        for index, row in X.iterrows():
            convert_amenities(X.loc[index, "amenities"], index)
        
        # 3) Drop 'amenities' feature
        X = X.drop("amenities", axis = 1)
        
        print("End - ExtractAmenitiesTransformer")

        return X
    
    def fit(self, X):
        return self
    
    
   
'''
        Class 
        ----------
            *Attributes:
                
               
            
            *:
  
'''   
class ExtractBathroom(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def transform(self, X, y=None):

        print("Start - ExtractBathroom")
        
        def bathroom_number(a_bath):
            
            is_shared = 0 #default value

            # Case if it is a missing value
            if(pd.isna(a_bath)):
                number_bath = np.nan
                is_shared = np.nan
                
            # Check if bathroom is shared using regex
            else:
                if(re.match(".*shared.*", str.lower(a_bath))):
                        is_shared = 1  


            return is_shared
            
    

        X[["is_bathroom_shared"]] = X["bathrooms_text"].apply(lambda x: pd.Series(bathroom_number(x)))
        
        X = X.drop("bathrooms_text", axis = 1)
        
        print("End - ExtractBathroom")
        
        return X
    
    def fit(self, X):
        return self
    





'''
        Class 
        ----------
            *Attributes:
                
               
            
            *:
  
'''
class ExtractScore(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    
    def transform(self, X, y=None):
        
        print("Start - ExtractScore")
        
        X["is_score_empty"] = X.review_scores_location.apply(lambda x: 1 if x != 0 else 0)
      
        print("End - ExtractScore")
        
        return X
    
    def fit(self, X):
        return self  



'''
        Class 
        ----------
            *Attributes:
                
               
            
            *:
  
'''
class CatImputer(BaseEstimator, TransformerMixin):
    def __init__(self, features_limits):
        self.features_limits = features_limits
    
    def transform(self, X, y=None):
        
        print("Start - CatImputer")
        
        FILL_VALUE = 1
        FEATURE = 0
        X_imp = X
        for feat in self.features_limits:
            imp = SimpleImputer(strategy='constant', fill_value= feat[FILL_VALUE])
            X_imp = imp.fit_transform(X_imp)
        
        print("End - CatImputer")
        
        return pd.DataFrame(X_imp, columns = X.columns)
    
    def fit(self, X):
        return self  
    


'''
        Class 
        ----------
            *Attributes:
                
               
            
            *:
  
'''
class NumImputer(BaseEstimator, TransformerMixin):
    def __init__(self, value):
        self.value = value
    
    def transform(self, X, y=None):
        
        print("Start - FeaturesImputer")
        
        num_ft = X.select_dtypes(exclude = 'object').columns
        for a_ft in num_ft:
            X.loc[:, a_ft] = X[a_ft].fillna(0)
        
        print("End - FeaturesImputer")
    
        return X
    
    def fit(self, X):
        
        
        return self
    
    


'''
        Class 
        ----------
            *Attributes:
                
               
            
            *:
  
'''
class TypeConversionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_to_type_list):
        self.feature_to_type_list = feature_to_type_list
    
    def transform(self, X, y=None):
    
        
        print("Start - TypeConversionTransformer")
        
        for feature, new_type in self.feature_to_type_list:
            X.loc[:,feature] = X[feature].astype(new_type)
        
        
        print("End - TypeConversionTransformer")
        
    
        
        return X
    
    def fit(self, X):
        
        
        return self
    



'''
        Class 
        ----------
            *Attributes:
                
               
            
            *:
  
'''
class FeatureEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, features_list):
        self.features_list = features_list
    
    def transform(self, X, y=None):
    
        
        print("Start - FeatureEncoding")
        
        for feature in self.features_list:    
            ## Encode here

            onehote_encoder = OneHotEncoder(sparse_output = False, feature_name_combiner='concat')
            series_onehot = onehote_encoder.fit_transform(X[[feature]])
            encoded_str = [str(x) for x in onehote_encoder.categories_[0]]
            X[encoded_str] = series_onehot
            X = X.drop(feature, axis = 1)
        
        print("End - FeatureEncoding")
        
    
        return X
    
    def fit(self, X):
        
        
        return self