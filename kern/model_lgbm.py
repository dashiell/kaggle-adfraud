#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 02:41:15 2018

@author: computer
"""

''' keras model '''

from utils import submit as submit_utils

import lightgbm as lgb
from ReadData import LGBData
    
    
params = {
        'boosting_type' : 'gbdt',
        'objective' :  'binary',
        'metric' : 'auc',
        'max_depth' : -1, # unlimited
        'num_leaves' : 31, # 100 must be smaller than 2^max_depth 
        'min_child_samples' : 20, # min data in leaf
        'max_bin' : 255, # 250
        'min_child_weight' : 0,
        'learning_rate' : 0.05,
        'colsample_bytree' : 0.9, # subsample ratio of columns when constructing each tree
        'subsample' : 0.6,
        #'feature_fraction' : 0.9,
        #'bagging_fraction' : 0.8,
        #'bagging_freq' : 5,
        'scale_pos_weight' : 200,
        'verbose' : 0,
        #'subsample_for_bin' : 200000, # number of sumbsamples for constructing bins
        'min_split_gain' : 0,
        'reg_alpha' : 0, # l1 regularization on weights
        'reg_lambda' : 0, # l2 regularization on weights
        'device' : 'gpu',

        }

data = LGBData()
data.split_train_valid(validation_size=0.1)


gbm = lgb.train(params, data.train, early_stopping_rounds=20, num_boost_round=5000, valid_sets=data.valid)

preds = gbm.predict(data.X_test)
submit_utils.output_submit('lgbm-1.csv', preds)
    
    
