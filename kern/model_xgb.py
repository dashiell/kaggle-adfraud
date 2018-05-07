#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 20:07:21 2018

@author: computer
"""

import xgboost as xgb
from ReadData import XGBData
from utils import submit as submit_utils

data = XGBData()
data.split_train_valid(validation_size=0.1)

params = {
        'eval_metric' : 'auc',
        'objective' : 'binary:logistic',
        #'tree_method' : 'gpu_hist', # crashes
        
        #'max_bin' : 16, # Maximum number of discrete bins to bucket continuous features.
        'learning_rate' : 0.1,
        #'gpu_id' : 0,
        }

# evals=data.valid,         early_stopping_rounds=50

xgb.train(params=params, 
          dtrain=data.train, 
          num_boost_round=3000, 
          evals=[(data.train, 'train'), (data.valid, 'validation')],
          early_stopping_rounds=20,
          )
preds = xgb.predict(data.X_test)
submit_utils.output_submit('xgb-0.csv', preds)