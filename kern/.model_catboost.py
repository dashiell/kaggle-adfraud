#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 01:20:32 2018

@author: computer
"""

from ReadData import ReadData
from catboost import Pool, CatBoostRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import gc 

data = ReadData()

data.X_train.drop(data.X_train.index[:int( 1e+8 )], inplace=True)
data.y_train.drop(data.y_train.index[:int( 1e+8 )], inplace=True)



X_train, X_valid, y_train, y_valid = train_test_split(data.X_train, data.y_train, test_size=0.1)

cat_features = X_train.columns.get_indexer_for(data.CATEGORICAL_FEATURES)

del data; gc.collect()

train_pool = Pool(X_train, y_train, cat_features=cat_features)

del X_train; del y_train; gc.collect()

val_pool = Pool(X_valid, y_valid, cat_features=cat_features)

del X_valid; del y_valid; gc.collect()


model = CatBoostRegressor(iterations=2, learning_rate=0.1, task_type='CPU')
model.fit(train_pool)