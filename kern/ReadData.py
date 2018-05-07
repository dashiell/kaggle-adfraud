#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 05:23:24 2018

@author: computer
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import Dataset as lgb_ds
from xgboost import DMatrix as xgb_ds
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

import gc


class ReadData(object):

    CATEGORICAL_FEATURES = ['app', 'device', 'os', 'channel', 'day', 'hour', 'second']
    
    def __init__(self):
        
        self.X_train = pd.read_pickle('../input/preproc/X_train.pkl')
        self.y_train = pd.read_pickle('../input/preproc/y_train.pkl')
        self.X_test = pd.read_pickle('../input/preproc/X_test.pkl')
                
    
class LGBData(ReadData):

    def __init__(self):
        super().__init__()

        drop_first_nrows = np.int(.7e8)
        
        self.X_train.drop(self.X_train.index[:drop_first_nrows], inplace=True)
        self.y_train.drop(self.y_train.index[:drop_first_nrows], inplace=True)
        gc.collect()


    def split_train_valid(self, validation_size):
        
        # portion of training sample read
        X_train, X_valid, y_train, y_valid = train_test_split(self.X_train, self.y_train, test_size=validation_size)
        
        self.train = lgb_ds(X_train, y_train, categorical_feature=self.CATEGORICAL_FEATURES)
        self.valid = lgb_ds(X_valid, y_valid, reference=self.train)
        
        del self.X_train; del self.y_train
        del X_train; del X_valid; del y_train; del y_valid
        
        
        gc.collect()
        
        
class XGBData(ReadData):
    
    def __init__(self):
        super().__init__()
                
        drop_first_nrows = np.int(1.5e8)
        
        self.X_train.drop(self.X_train.index[:drop_first_nrows], inplace=True)
        self.y_train.drop(self.y_train.index[:drop_first_nrows], inplace=True)
        
        train_n = self.X_train.shape[0]
        
        all_X = pd.concat([self.X_train, self.X_test], axis=0)
        
        del self.X_train; del self.X_test
        gc.collect()
        
        encoder = OneHotEncoder(sparse=True)
        
        oh_cat_feats = encoder.fit_transform(all_X[self.CATEGORICAL_FEATURES])
        
        oh_numeric = all_X.drop(self.CATEGORICAL_FEATURES, axis=1).as_matrix()
        
        self.X_train = sparse.hstack([oh_cat_feats[:train_n], oh_numeric[:train_n]], format='csr')
        self.X_test = sparse.hstack([oh_cat_feats[train_n:], oh_numeric[train_n:]], format='csr')
        
        del oh_cat_feats; del oh_numeric; del all_X
        
        gc.collect()
        
    
    def split_train_valid(self, validation_size):
        print("pre splitted dtype: ", self.X_train.dtype)
        
        X_train, X_valid, y_train, y_valid = train_test_split(self.X_train, self.y_train, test_size=validation_size)
        
        print("splitted dtype: ", X_train.dtype)
        
        self.train = xgb_ds(X_train, y_train)
        self.valid = xgb_ds(X_valid, y_valid)
        
        
        del self.X_train; del self.y_train
        del X_train; del y_train; del X_valid; del y_valid
        gc.collect()
        
        

class KerasData(ReadData):
    def __init__(self):
        super().__init__()
        print("LGBDATA::init")
        self.max_dict = self._calc_max_cat_feats()
        
    
    def split_train_valid(self, validation_size):
        super().split_train_valid(validation_size)
        
        self.X_train = self._get_keras_dataset(self.X_train)
        self.X_valid = self._get_keras_dataset(self.X_valid)
        self.X_test = self._get_keras_dataset(self.X_test)
    
        gc.collect()
        
    # get the maximum value for each categorical feature in train, test
    def _calc_max_cat_feats(self):
        max_dict = {}
        for feat in self.CATEGORICAL_FEATURES:
            max_dict[feat] = np.max( [ self.X_train[feat].max(), self.X_test[feat].max() ] ) + 1
            
        return max_dict
    
    
    def _get_keras_dataset(self, df):
        keras_dict = {}
        
        # all the categorical features
        for col in self.CATEGORICAL_FEATURES:
            keras_dict[col] = np.array(df[col])

        #
        keras_dict['X_non_categorical'] = df.drop(self.CATEGORICAL_FEATURES, axis=1).as_matrix()
        #print(keras_dict['X_non_categorical'].shape)
        return keras_dict
    
