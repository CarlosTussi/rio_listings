import pandas as pd
import numpy as np

import operator as op
import re

# Custom Transformers
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.cluster import KMeans


from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

from source.config import *

import joblib


'''
        Class 
        ----------
            *Attributes:
                
               
            
            *:
  
'''
class ColumnDroppersTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.drop_feat = []


    def fit(self, X, y=None):

        # Find features related to host, id and url
        re_drop = FEATURES_TO_DROP_REGEX
        for feat in X.columns:
            if re.match(re_drop, feat):
                 self.drop_feat.append(feat)
        # Add extra features that for sure will not ber part of the model
        # This could be defined in the init step, but placed here together if the regex feature selection for better organisation
        self.drop_feat.extend(FEATURES_TO_DROP)

        return self
    
    def transform(self, X, y=None):
        
        print("Start - ColumnDroppersTransformer")
        
        X = X.drop(self.drop_feat, axis = 1)
        
        print("End - ColumnDroppersTransformer")
        
        print(X.columns)
 
        return X
    
    


'''
        Class 
        ----------
            *Attributes:
                
               
            
            *:
  
'''
class DropNasTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features
    
    def fit(self, X):
        return self

    def transform(self, X, y=None):
        
        print("Start - DropNasTransformer")
        
        
        X = X.dropna(subset = self.features)
        
        print(type(X))
        print("End - DropNasTransformer")
        
        
        return X
    




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
     
    def fit(self, X):
        return self
    
    def transform(self, X, y=None):
        
        print("Start - OutlierRemover")

        
        NAME = 0
        LIMIT = 1
        REPLACE_VALUE = 2
        for a_feature in self.features_limit:
            if(self.mode == "cap"):
                X[a_feature[NAME]] =  X[a_feature[NAME]].apply(lambda x : x if self.opt_dict[self.operator](x,a_feature[LIMIT]) else a_feature[LIMIT])
            elif(self.mode == "replace"):
                X[a_feature[NAME]] =  X[a_feature[NAME]].apply(lambda x : a_feature[REPLACE_VALUE] if self.opt_dict[self.operator](x,a_feature[LIMIT]) else x)
        
        
        print("End - OutlierRemover")
        
        
        return X

        
        


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

        # Initialize KMeans model
        self.k_means = KMeans(
            init = self.init,
            n_clusters = self.clusters,
            n_init = self.n_init,
            max_iter = self.max_iter,
            random_state = 69
        )

        # Initialize the encoder
        self.onehot_encoder_cluster = OneHotEncoder(sparse_output = False, feature_name_combiner='concat')

        
    def fit(self, X):

        #Select the data to be clustered
        df_kmeans = X.loc[:,["latitude", "longitude"]]

        # Fitting
        self.k_means.fit(df_kmeans)
        #Clusters as a feature
        X['geo_cluster'] = self.k_means.labels_ 

        # Save the model
        joblib.dump(self.k_means, GEO_CLUSTER_MODEL_PATH)


        # Onde-Hot-Encoder
        self.onehot_encoder_cluster.fit(X[["geo_cluster"]])


        return self



    def transform(self, X, y=None):
        
        print("Start - ClusterGeolocationTransformer")

        ## Encode new feature
        encoded_cat_str = [str(x) for x in self.onehot_encoder_cluster.get_feature_names_out(["geo_cluster"])]
        series_cluster_onehot = self.onehot_encoder_cluster.transform(X[['geo_cluster']])
        X[encoded_cat_str] = series_cluster_onehot
        X = X.drop("geo_cluster", axis = 1)
    
        
        #Drop latitude and longitude as they will not be fed into the model
        X = X.drop(["longitude", "latitude"], axis = 1)
        
        
        print("End - ClusterGeolocationTransformer")
        
        return X



'''
        Class 
        ----------
            *Attributes:
                
               
            
            *:
  
'''
class PreprocessCorpus(BaseEstimator, TransformerMixin):
    def __init__(self, corpus_feature):
        self.corpus_feature = corpus_feature

    def fit(self, X):
        return self
    
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
    

    def fit(self, X):
        return self

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

    
    


'''
        Class 
        ----------
            *Attributes:
                
               
            
            *:
  
'''
class ExtractAmenitiesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, amenities_dict):
        self.amenities_dict = amenities_dict
    

    def fit(self, X):
        return self

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
    
    
   
'''
        Class 
        ----------
            *Attributes:
                
               
            
            *:
  
'''   
class ExtractBathroom(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X):
        return self

    def transform(self, X, y=None):

        print("Start - ExtractBathroom")
        
        def bathroom_number(a_bath):
            
            is_shared = 0 #default value

            # Case if it is a missing value
            if(pd.isna(a_bath)):
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
    
    





'''
        Class 
        ----------
            *Attributes:
                
               
            
            *:
  
'''
class ExtractScore(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X):
        return self  
    
    def transform(self, X, y=None):
        
        print("Start - ExtractScore")
        
        X["is_score_empty"] = X.review_scores_location.apply(lambda x: 1 if x != 0 else 0)
      
        print("End - ExtractScore")
        
        return X
    



'''
        Class 
        ----------
            *Attributes:
                
               
            
            *:
  
'''
class CatImputer(BaseEstimator, TransformerMixin):
    def __init__(self, features_limits):
        self.features_limits = features_limits

        self.imputers = {}

       

    def fit(self, X):


        # For each feature in the list
        for feat_tuple in self.features_limits:
            feature_name =  feat_tuple[0]
            feature_replace_value = feat_tuple[1]

            # Initialize with corresponding imputer value
            imp = SimpleImputer(strategy='constant', fill_value= feature_replace_value)

            # Fit in the corresponding feature
            imp.fit(X[[feature_name]])

            # Update with imputer dictionary
            self.imputers.update({

                    feature_name: imp

            })

        return self  


    def transform(self, X, y=None):
        
        print("Start - CatImputer")
        

        # For each feature to be imputer
        for feat_tuple in self.features_limits:
            
            # Retrieve its name
            feature_name = feat_tuple[0]
            # Retrieve its fitted imputer
            imp = self.imputers[feature_name]
            # Transform
            X[[feature_name]] = imp.transform(X[[feature_name]])
        
        print("End - CatImputer")
        
        return X
    
    


'''
        Class 
        ----------
            *Attributes:
                
               
            
            *:
  
'''
class NumImputer(BaseEstimator, TransformerMixin):
    def __init__(self, value):
        self.value = value

    def fit(self, X):  
        return self
    
    def transform(self, X, y=None):
        
        print("Start - FeaturesImputer")
        
        num_ft = X.select_dtypes(exclude = 'object').columns
        for a_ft in num_ft:
            X.loc[:, a_ft] = X[a_ft].fillna(self.value)
        
        print("End - FeaturesImputer")
    
        return X

    
    


'''
        Class 
        ----------
            *Attributes:
                
               
            
            *:
  
'''
class TypeConversionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_to_type_list):
        self.feature_to_type_list = feature_to_type_list
    
    
    def fit(self, X):
        return self

    def transform(self, X, y=None):
    
        
        print("Start - TypeConversionTransformer")
        
        for feature, new_type in self.feature_to_type_list:
            X.loc[:,feature] = X[feature].astype(new_type)
        
        
        print("End - TypeConversionTransformer")
        
    
        
        return X
    



'''
        Class 
        ----------
            *Attributes:
                
               
            
            *:
  
'''
class FeatureEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, features_list):
        self.features_list = features_list
    
        #One-Hot-Encoder
        self.onehote_encoders = {}
        
    def fit(self, X):

        # For each feature, adds the corresponding encoder and encoded features' name in the dict
        for feature in self.features_list:  

            # Initialize current encoder
            an_encoder = OneHotEncoder(sparse_output = False, feature_name_combiner='concat')
            # Fit
            an_encoder.fit(X[[feature]])
            # Extract column names
            encoded_str = [str(x) for x in an_encoder.get_feature_names_out()]

            # Save current feature encoder in the dict
            self.onehote_encoders.update(
                {
                    feature : {
                        "encoder": an_encoder,
                        "encoded_str": encoded_str 
                    }
                }
            )
            
        return self


    def transform(self, X, y=None):
    
        
        print("Start - FeatureEncoding")

        # For each feature to be encoded
        for feature in self.features_list:
            # Encoder from the corresponding feature
            for key, feature_encoder in self.onehote_encoders.items():    

                # Transform
                series_onehot = feature_encoder["encoder"].transform(X[[feature]])

                # Create new feature in the dataset
                X[feature_encoder["encoded_str"]] = series_onehot

                # Drop original feture
                X = X.drop(feature, axis = 1)
            
        print("End - FeatureEncoding")
        
    
        return X
    

class GeoClusterPredict(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model
    
    def fit(self, X):
        return self

    def transform(self, X, y=None):
        
        print("Start - GeoClusterPredict")
        
        
       # Predict geo-clusyter model

        cluster = self.model.predict(X.loc[:,["latitude", "longitude"]])[0]
        X["geo_cluster"] = cluster

        # One hot-encode representation (manually encoding all 25 categories) and indicate the cluster which current geo-cluster number belongs to
        for i in range(0,25):
            X["geo_cluster_"+str(i)] = 0
        X["geo_cluster_"+str(cluster)] = 1
        X = X.drop(["latitude", "longitude", "geo_cluster"], axis = 1)
    
        print("End - GeoClusterPredict")
        
        
        return X

class CustomMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.minmax = MinMaxScaler()
    
    def fit(self, X):
        self.minmax.fit(X)
        joblib.dump(self.minmax, NORM_MODEL_PATH)
        return self

    def transform(self, X, y=None):
        
        print("Start - CustomMinMaxScaler")
        
        
        X = self.minmax.transform(X)
    
        print("End - CustomMinMaxScaler")
        
        
        return X

class CustomMinMaxScalerAppTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model
        self.columns_order = None
    
    def fit(self, X):
        self.columns_order = self.model.feature_names_in_
        return self

    def transform(self, X, y=None):
        
        print("Start - CustomMinMaxScalerAppTransformer")
        
        
        # Reordering to transofrm with the normaliser (MaxMin required the same order)
        X_normalise = pd.DataFrame()
        for col in self.columns_order:
            X_normalise[col] = X[col]

        X = self.model.transform(X_normalise)
    
        print("End - CustomMinMaxScalerAppTransformer")
        
        
        return X
