'''
    How to run:
    1) From rio_listing main folder
    2) python -m pytest .\test\

'''


from source.pipelines.custom_transformers import *
from literals import *
import pandas as pd
import numpy as np

'''
    Testing: DropNasTransformer
'''
def test_DropNasTransformer():
    
    ########################
    # Remove from 1 column #
    ########################
    # Removing missing values from columns A
    drop_nas1 = DropNasTransformer(["A"])
    drop_nas1.fit(X_missing)
    # No empty columns after transformation
    assert pd.isnull(drop_nas1.transform(X_missing)["A"]).sum() == 0

    ################################
    # Remove from multiple columns #
    ################################
    # Removing missing values from columns A and C
    drop_nas2 = DropNasTransformer(["A", "C"])
    drop_nas2.fit(X_missing)
    # No empty columns after transformation
    assert pd.isnull(drop_nas1.transform(X_missing)[["A","C"]]).sum().all() == 0

    ######################################
    # Columns without any missing values #
    ######################################
    rop_nas2 = DropNasTransformer(["A", "C"])
    drop_nas2.fit(X_missing)
    # No empty columns after transformation
    assert pd.isnull(drop_nas1.transform(X_missing)["D"]).sum() == 0


'''
    Testing: PreprocessCorpus
'''

def test_PreprocessCorpus():
    X_corpus

    ################################
    # Removing unwanted characters #
    ################################
    preprocess_corpus1 = PreprocessCorpus("description")
    preprocess_corpus1.fit(X_corpus)
    assert(preprocess_corpus1.transform(X_corpus).equals(X_processed_corpus))


    
