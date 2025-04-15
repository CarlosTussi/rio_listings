'''
    How to run:
    1) From rio_listing main folder
    2) python -m pytest -v .\test\

'''


from source.pipelines.custom_transformers import *
from literals import *
import pandas as pd
import numpy as np



'''
    Testing: ClusterGeolocationTransformer
'''
def test_ClusterGeolocationTransformer():
    X_geolocation
    geocluster1 = ClusterGeolocationTransformer()
    geocluster1.fit(X_geolocation)
    X_mod = geocluster1.transform(X_geolocation)
    X_mod_cols = [geo_col for geo_col in X_mod.columns if "geo_cluster" in geo_col]

    # TESTING: geo_cluster columns were created
    assert(len(X_mod_cols) > 0 )
    # TESTING: Absence of missing values in the columns generated
    assert(X_mod[X_mod_cols].isna().sum().all() == 0)
    # TESTING: Transformed dataset is a Pandas DataFrame 
    assert(isinstance(X_mod, pd.DataFrame))



'''
    Testing: ContainWordsTransformer
'''
def test_ContainWordsTransformer():
    NEW_FT_NAME = "contains_lux_description"
    TARGET = "description"
    WORDS = ["luxurious","fancy", "sophistication", "rooftop"]
    containwords = ContainWordsTransformer(new_feature_name = NEW_FT_NAME,
                                            corpus_target= TARGET,
                                            words = WORDS
    ) 
    containwords.fit(X_lux)
    X_mod = containwords.transform(X_lux)

    # TESTING: New feature was added to the dataset
    assert(NEW_FT_NAME in X_mod.columns)
    # TESTING: Words were correctly detected in the corpus
    assert(X_mod[NEW_FT_NAME].equals(X_processed_lux))


'''
    Testing: CustomMinMaxScaler
'''

def test_CustomMinMaxScaler():
    customminmax = CustomMinMaxScaler()
    customminmax.fit(X)
    X_mod = customminmax.transform(X)
    # TESTING: Transformed dataset is a Pandas DataFrame 
    assert(isinstance(X_mod, pd.DataFrame))


'''
    Testing: FeatureEncoding
'''
def test_FeatureEncoding():
    feature_encoding = FeatureEncoding(["house_types"])
    feature_encoding.fit(X_encoding)
    X_mod = feature_encoding.transform(X_encoding)

    # TESTING: Testing elements correctly encoded
    assert(X_mod.equals(X_processed_encoding))
    # TESTING: Transformed dataset is a Pandas DataFrame 
    assert(isinstance(X_mod, pd.DataFrame))
    


'''
    Testing: DropNasTransformer
'''
def test_DropNasTransformer():
    
    # Removing missing values from columns A
    drop_nas1 = DropNasTransformer(["A"])
    drop_nas1.fit(X_missing)
    # TESTING: No NAs removed from single column
    assert pd.isnull(drop_nas1.transform(X_missing)["A"]).sum() == 0


    # Removing missing values from columns A and C
    drop_nas2 = DropNasTransformer(["A", "C"])
    drop_nas2.fit(X_missing)
    # No empty columns after transformation
    X_mod = drop_nas1.transform(X_missing)
    # TESTING: NAs removed from multiple columns
    assert pd.isnull(X_mod[["A","C"]]).sum().all() == 0


    # Columns without any missing values #
    drop_nas3 = DropNasTransformer(["A", "C"])
    drop_nas3.fit(X_missing)
    X_mod = drop_nas3.transform(X_missing)
    # TESTING: Column has no missing values before transform
    assert pd.isnull(X_mod["D"]).sum() == 0
    
    # TESTING: Transformed dataset is a Pandas DataFrame 
    assert(isinstance(X_mod, pd.DataFrame))


'''
    Testing: PreprocessCorpus
'''

def test_PreprocessCorpus():

    preprocess_corpus1 = PreprocessCorpus("description")
    preprocess_corpus1.fit(X_corpus)
    X_mod = preprocess_corpus1.transform(X_corpus)

    # TESTING: Unwanted characters were correctly removed
    assert(X_mod.equals(X_processed_corpus))
    # TESTING Transformed dataset is a Pandas DataFrame 
    assert(isinstance(X_mod, pd.DataFrame))

