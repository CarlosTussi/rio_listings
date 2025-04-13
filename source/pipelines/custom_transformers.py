'''

    This module contains all the custom transformers' implementation required for the pipeline.
    Each custom transformer implements methos fit() and transforms().

'''


import pandas as pd
import numpy as np
import operator as op
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from source.config import *
import joblib


'''
        class ColumnDroppersTransformer(BaseEstimator, TransformerMixin)
        ----------------------------------------------------------------

            Description:
            ************
                - This transformer drops unwanted features from the dataset.
        

            Attributes:
            ***********
                - drop_feat : List
            
            Fit:
            *****
                - Populate the list of features that will be dropped.

            Transform:
            ************
                - Drop features from the list.
               
  
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
        class DropNasTransformer(BaseEstimator, TransformerMixin) 
        ----------------------------------------------------------

            Description:
            ************
                - Remove NAs from a list of features.
        
            Attributes:
            ***********
                - features : list
            
            Fit:
            *****
                - None

            Transform:
            ************
                - Drop nas from the list
  
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
        class OutlierRemover(BaseEstimator, TransformerMixin) 
        ----------------------------------------------------
            Description:
            ************
                - Remove outliers from different types following different conditions:
                    - a <  x    ("lt")
                    - a <= x    ("le")
                    - a >  x    ("gt")
                    - a >= x    ("ge")
                    - a == x    ("eq")
                    - a != x    ("ne")
                
                - With different modes of operation:
                    - "replace": replace value (a) that respect a condition above with another value (b)
                    - "cap": cap value (a) that respect a condition above by assigned value (a)

                - Example 1: Replace all values bigger than 10 for "feat_a" with 10 and values bigger than 5 for "feat_b" with 5.

                        features_limit = [("feat_a", 10),
                                          ("feat_b", 5)] 
                        mode = "cap",
                        operatore = "lt"

                - Example 2: Replace all values that matches "Hotel Room" with string "Private Room".

                    features_limit = [("feat_a", "Hotel Room", "Private Room")]
                    mode = "replace",
                    operator = "eq"

                
            Attributes:
            ***********
                - features_limit : list[
                                        tuple(str, val_a),         #If only mode 'cap'
                                        tuple(str, val_a, val_b)   #If only mode 'replace'
                                        ]
                - mode: str

                - operator: str
                                        
            Fit:
            *****
                - None

            Transform:
            ************
                - If mode 'cap', replace all values that do not satisify the limit with the limit.
                - If mode 'replace', replace all values with some other value.
  
'''
class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, features_limit, mode = "replace", operator = "eq"): #st: smaller then
                    
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
        Class ClusterGeolocationTransformers 
        ------------------------------------
            Description:
            ************
                - Responsable for separating a set of coordinates (latitude/longitude) into different clusters using
                K-means method.
                - It saves the transformer model trained on the data for future use.
                - One-hot-encode the different clusters using One-Hot-Encoder
                
            Attributes:
            ***********
                - clusters: int,                            # Number of clusters
                - init: str,                                # Method for initialization for K-means algorithm
                - n_init: str,                              # Number of times the k-means algorithm
                - max_item: int                             # Max number of iteration of k-mean algorithm for each run
                - k_means: Kmeans()                         # KMeans model
                - onehot_encoder_cluster: OneHotEncoder()   # OneHoteEncoder encoder
            
            Fit:
            *****
                - Train KMeans on the dataset and retrieve the labels.
                - Creates new feature ("geo_cluster") with the lables.
                - Train OneHotEncoder with the new feature ("geo_cluster")
                - Save the trained encoder for future use.

            Transform:
            **********
                - OneHotEncode new feature ("geo_cluster").
                - Drop features "latitude", "longitude" and "geo_cluster"
  
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
        class PreprocessCorpus(BaseEstimator, TransformerMixin) 
        -------------------------------------------------------

            Description:
            ************
                - Preprocess some text with some predefined rules.
                
            Attributes:
            ***********
                - corpus_feature: str
            
            Fit:
            *****
                - None

            Transform:
            ************
                - Apply predefined rules transforming the text input.
  
'''
class PreprocessCorpus(BaseEstimator, TransformerMixin):
    def __init__(self, corpus_feature):
        self.corpus_feature = corpus_feature

    def fit(self, X):
        return self
    
    def transform(self, X, y=None):
        
        print("Start - PreprocessCorpus")
        
        
        def process_corpus(corpus):
            
            # Convert everything to lower case
            corpus = corpus.lower()
            # Remove eeverytyhing except letters and space
            corpus = re.sub(r'[^a-zA-Z ]', ' ', corpus)
            # Removing specific words
            corpus = re.sub(r'\b(the|are|and|in|with|to|a|of|br|is|from|for|on|this|you|it|has|all|at|de|by|br)\b', ' ', corpus)
            # Removing line breaker tag from HTML
            corpus = re.sub(r'br\b', '', corpus)
            # Removing extra white spaces
            corpus = re.sub(r' +', ' ', corpus)
            # Remove leading/trailing spaces
            corpus = corpus.strip()

            return corpus
    
        X[self.corpus_feature] = X[self.corpus_feature].apply(lambda x : process_corpus(x))
        
        print("End - PreprocessCorpus")
        
        return X
        



'''
        class ContainWordsTransformer(BaseEstimator, TransformerMixin) 
        --------------------------------------------------------------

            Description:
            ************
                - Creates a new feature that indicates weather some text contains some set of words or not.
                
            Attributes:
            ***********
                - words: list(str)
                - new_feature_name: str
                - corpus_target: str
            
            Fit:
            *****
                - None

            Transform:
            ************
                - Indicate if some text contains any of those words (1) or not (0) adding the result as a new columns.
                - Drop original corpus_target feature
  
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
        #X = X.drop(self.corpus_target, axis = 1)
        
        print("End - ContainWordsTransformer")
        
        return X

    
    


'''
        class ExtractAmenitiesTransformer(BaseEstimator, TransformerMixin) 
        ------------------------------------------------------------------

            Description:
            ************
                - Creates a new feature for each amenity being searched and populates it with 0 (not present) or 1 (present in the property)
                - It accomplishes it by using regex to parse through the amenity's list text from the raw input data.
                
            Attributes:
            ***********
                - amenities_dic: dict             # {amenity_name: [amenity_regex]}
                    Example:
                            {
                                "bathtub": ".*bathtub.*",
                                "seaview" : [".*beach view.*",".*sea view.*",".*ocean view.*"]
                            }
            
            Fit:
            *****
                - None

            Transform:
            ************
                - Identify the amenities and populate the new amenity's features.
                - Drop the original amenities feature.
  
'''
class ExtractAmenitiesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, amenities_dict):
        self.amenities_dict = amenities_dict
    

    def fit(self, X):
        return self

    def transform(self, X, y=None):
        
        print("Start - ExtractAmenitiesTransformer")
        
        '''
            def convert_amenities(amenities, index):
            ----------------------------------------
                - This function verifies the existance using regex of the searched amenities in the current amenitites list from the property,
                which is all concatenated in a large string of amenitites by default from the raw dataset.
                - It creates a new feature for each target amenitiy being searched "has_(amenity)"
                - For each amenity target identified, it indicated that it exists (1)

                input:
                ------
                    - amenitites: str
                    - index: int

                output:
                -------
                    - None
        '''
        def convert_amenities(amenities, index): 
            # For each amenity target
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
        class ExtractBathroom(BaseEstimator, TransformerMixin) 
        ------------------------------------------------------
            Description:
            ************
                - Check if the bathroom is shared or not based on the text information available.
                
            Attributes:
            ***********
                - None
            
            Fit:
            *****
                - None

            Transform:
            ************
                - Creates a new feature that indicaates the presence  (1) or not (0) of a shared toilet.
                - Drops the original text information of the bathrooms.
  
'''   
class ExtractBathroom(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X):
        return self

    def transform(self, X, y=None):

        print("Start - ExtractBathroom")
        
        def detect_bathroom_shared(a_bath):
            
            is_shared = 0 #default value

            # Case if it is a missing value
            if(pd.isna(a_bath)):
                is_shared = np.nan
                
            # Check if bathroom is shared using regex
            else:
                if(re.match(".*shared.*", str.lower(a_bath))):
                        is_shared = 1  


            return is_shared
            
    

        X[["is_bathroom_shared"]] = X["bathrooms_text"].apply(lambda x: pd.Series(detect_bathroom_shared(x)))
        
        X = X.drop("bathrooms_text", axis = 1)
        
        print("End - ExtractBathroom")
        
        return X
    
    





'''
        class ExtractScore(BaseEstimator, TransformerMixin)
        ---------------------------------------------------
            Description:
            ************
                - Creates a new feature that indicates weather property has not yet received any reviews for the location.
                
            Attributes:
            ***********
                - None
            
            Fit:
            *****
                - None

            Transform:
            ************
                - Creates a new feature that indicaates the presence  (0) or not (1) of a location review.
  
'''
class ExtractScore(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X):
        return self  
    
    def transform(self, X, y=None):
        
        print("Start - ExtractScore")
        
        X["is_score_empty"] = X.review_scores_value.apply(lambda x: 1 if x != 0 else 0)
      
        print("End - ExtractScore")
        
        return X
    



'''
        class CatImputer(BaseEstimator, TransformerMixin) 
        -------------------------------------------------
                Description:
                ************
                    - For each feature to be imputed, train a SimpleImputer and impute with a value.
                    
                Attributes:
                ***********
                    - features_limits: list(tuple(str, str))

                            - Exemple:
                                 features_limits =  [("bathrooms_text", "Private bath"),
                                                    ("description", "")]
                                                    
                
                
                Fit:
                *****
                    - Fit a SimpleImputer with a replace value for each feature from the list.
                    - Creates a dictionary with all the trained SimpleImputers

                Transform:
                ************
                    - Impute the features from the list with their respective traineed imputers.
  
'''
class CatImputer(BaseEstimator, TransformerMixin):
    def __init__(self, features_replace):
        self.features_replace = features_replace

        self.imputers = {}

       

    def fit(self, X):


        # For each feature in the list
        for feat_tuple in self.features_replace:
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
        for feat_tuple in self.features_replace:
            
            # Retrieve its name
            feature_name = feat_tuple[0]
            # Retrieve its fitted imputer
            imp = self.imputers[feature_name]
            # Transform
            X[[feature_name]] = imp.transform(X[[feature_name]])
        
        print("End - CatImputer")
        
        return X
    
    


'''
        class NumImputer(BaseEstimator, TransformerMixin) 
        --------------------------------------------------
                Description:
                ************
                    - Impute all numerical features with a value.
                    
                Attributes:
                ***********
                    - value: numerical
                
                Fit:
                *****
                    - None

                Transform:
                ************
                    - Fill all NAs with a value.
  
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
        class TypeConversionTransformer(BaseEstimator, TransformerMixin) 
        -----------------------------------------------------------------
                Description:
                ************
                    - Change a feature's type.
                    
                Attributes:
                ***********
                    - feature_to_type_list : list(tuple(str,dtype))
                
                Fit:
                *****
                    - None

                Transform:
                ************
                    - Convert the type for each (feature, dtype) tuple from the list.
  
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
        class FeatureEncoding(BaseEstimator, TransformerMixin)
        -------------------------------------------------------
                Description:
                ************
                    - Encode each feature from the list.
                    
                Attributes:
                ***********
                    - features_list: list(str)
                    - onehot_encoders: list(OneHotEncoder())
                
                Fit:
                *****
                    - Train each encoder with each feature of the features_list and 
                    add the trained encoder in a dictionary with respective feature.

                Transform:
                ************
                    - Encode each feature with respective encoder and drop original feature.
  
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
    
'''
        class GeoClusterPredict(BaseEstimator, TransformerMixin)
        --------------------------------------------------------
                Description:
                ************
                    - This class is mainly for an application where we need to make the input compliant with the price prediction model.
                    - Will predict a cluster membership for a coordinate (latitude/longitude)
                    
                Attributes:
                ***********
                    - model : KMeans()      # Already trained
                
                Fit:
                *****
                    - None

                Transform:
                ************
                    - Predict the clusted membership and create one-hot-encode features of all possible cluster categories to comply with 
                    the price prediction model.
                    - Drop latitude, longitude and geo_cluster.
'''
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

        # One hot-encode representation (manually encoding all NUMBER_OF_CLUSTERS_FT categories) and indicate the cluster which current geo-cluster number belongs to
        for i in range(0, NUMBER_OF_CLUSTERS_FT):
            X["geo_cluster_"+str(i)] = 0
        X["geo_cluster_"+str(cluster)] = 1
        X = X.drop(["latitude", "longitude", "geo_cluster"], axis = 1)
    
        print("End - GeoClusterPredict")
        
        
        return X

'''
        class CustomMinMaxScaler(BaseEstimator, TransformerMixin)
        ---------------------------------------------------------
                Description:
                ************
                    - Augmented vrsion of MinMaxScaler where the trained model is saved to be used by other applications later.
                    
                Attributes:
                ***********
                    - minmax: MinMaxScaler()
                
                Fit:
                *****
                    - Train the model
                    - Save the trained model

                Transform:
                ************
                    - Scale the dataset
'''
class CustomMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.minmax = MinMaxScaler()
        self.dfcolumns = []
        self.dfindex = []
    
    def fit(self, X):
        # Train the MinMaxScaler
        self.minmax.fit(X)

        # Save the trained MinMaxScaler to be used later
        joblib.dump(self.minmax, SCALER_MODEL_PATH)

        # Save the index and columns name order to re-create the dataframe after transformation
        self.dfcolumns = X.columns
        self.dfindex = X.index
        return self

    def transform(self, X, y=None):
        
        print("Start - CustomMinMaxScaler")
        
        X = self.minmax.transform(X)
    
        # Convert back into a Data Frame
        X = pd.DataFrame(X, index=self.dfindex, columns=self.dfcolumns)

        print("End - CustomMinMaxScaler")
        
        
        return X

'''
        class CustomTrainedMinMaxScalerTransformer(BaseEstimator, TransformerMixin)
        -----------------------------------------------------------------------
                Description:
                ************
                    - This class was made to be used by an application where we need to make sure that the order of the features are
                    the same as the one used when training the model.
                    
                Attributes:
                ***********
                    - model : MinMaxScaler()
                    - columns_order = list
                
                Fit:
                *****
                    - Retrieve the model columns' order.

                Transform:
                ************
                    - Reorder dataset to reflect the same order as the trained MinMaxScaler()
'''
class CustomTrainedMinMaxScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model
        self.columns_order = None
    
    def fit(self, X):
        self.columns_order = self.model.feature_names_in_

        return self

    def transform(self, X, y=None):
        
        print("Start - CustomMinMaxScalerAppTransformer")
        
        
        # Reordering to transofrm with the scaler (MaxMin required the same order)
        X_scaler = pd.DataFrame()
        for col in self.columns_order:
            X_scaler[col] = X[col]

        # To transform back into a dataframe
        dfcolumns = X_scaler.columns
        dfindex = X_scaler.index

        # Transform scale the data
        X = self.model.transform(X_scaler)
    
        # Convert back into dataframe
        X = pd.DataFrame(X, index=dfindex, columns=dfcolumns)

        print("End - CustomMinMaxScalerAppTransformer")
        
        
        return X