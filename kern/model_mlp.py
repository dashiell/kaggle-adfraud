#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 06:50:10 2018

@author: computer
"""

from keras.models import Model
from keras.utils import multi_gpu_model
import tensorflow as tf
from keras.layers import Input, Dense, Dropout, Embedding, concatenate, Flatten
from keras.layers import SpatialDropout1D
from utils import train as train_utils
from utils import submit as submit_utils
from ReadData import KerasData
from lightgbm import Dataset as lgb_ds
import numpy as np

def mlp_model(emb_size_dict, non_categorical_size):

    emb_out = 20
    inps = []
    emb_layers = []
    
    # The categorical features
    for cat_feat, size in emb_size_dict.items():
        cat_inp = Input(shape=[1], name=cat_feat)
        emb_layer = Embedding(size, emb_out) (cat_inp)
        
        inps.append(cat_inp)
        emb_layers.append(emb_layer)
    
    # The non-categorical features
    in_X_non_categorical = Input(shape=(non_categorical_size,), name='X_non_categorical')    
    inps.append(in_X_non_categorical)
        
    oh_embedded = concatenate(emb_layers)
    oh_embedded = SpatialDropout1D(0.3) (oh_embedded)
    x_emb = Flatten() (oh_embedded)

    x_emb = Dense(128, activation='relu') (x_emb)    
    
    x_non_categorical = Dense(64, activation='relu') (in_X_non_categorical)
    
    x = concatenate([x_emb, x_non_categorical]) 
    
    x = Dense(128, activation='relu') (x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu') (x)
    
    out = Dense(1, activation='sigmoid') (x)
    
    #with tf.device('/cpu:0'):
    model = Model(inputs = inps , outputs = out)
    #model = multi_gpu_model(model, gpus=2)
    

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model


data = KerasData()
data.split_train_valid(validation_size=0.1)


model = mlp_model(data.max_dict, data.X_train['X_non_categorical'].shape[1])
train_utils.fit_on_val(model, 40000, data.X_train, data.X_valid, data.y_train, data.y_valid)
preds = model.predict(data.X_test)
submit_utils.output_submit('mlp-0.csv', preds)


