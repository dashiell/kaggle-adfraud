#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn import preprocessing
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import gc
import utils.aggregate as agg

''' read train, test '''

def read_tt():
    dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint8',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32',
        }
    use_cols = ['ip','app','device','os','channel','click_time','is_attributed']
    
    train = pd.read_csv('../input/train.csv', dtype=dtypes, usecols=use_cols)#, skiprows=np.arange(1,100000000))
    test = pd.read_csv('../input/test.csv', dtype=dtypes, usecols=use_cols[:-1])
    
    train_n = train.shape[0]
    
    all_df = pd.concat( (train, test), axis=0, ignore_index=True)
    
    
    all_df['day'] = pd.to_datetime(all_df.click_time).dt.day.astype('uint8')
    all_df['hour'] = pd.to_datetime(all_df.click_time).dt.hour.astype('uint8')
    all_df['second'] = pd.to_datetime(all_df.click_time).dt.second.astype('uint8')
    return all_df, train_n

def agg_summary(all_df):
    all_df = agg.cum_count( all_df, ['ip', 'device', 'os'], 'app', 'Xcc0' ); gc.collect()
    all_df = agg.cum_count( all_df, ['ip'], 'os', 'Xcc1' ); gc.collect()
    all_df = agg.cum_count( all_df, ['ip', 'day', 'hour'], 'os', 'Xcc2' ); gc.collect()
    all_df = agg.cum_count( all_df, ['ip', 'day', 'hour'], 'second', 'Xcc3' ); gc.collect()
    all_df = agg.cum_count( all_df, ['ip'], 'device', 'Xcc4' ); gc.collect()
    
    all_df = agg.count_unique( all_df, ['ip'], 'channel', 'Xcu0' ); gc.collect()
    all_df = agg.count_unique( all_df, ['ip', 'day'], 'hour', 'Xcu1' ); gc.collect()
    all_df = agg.count_unique( all_df, ['ip'], 'app', 'Xcu2' ); gc.collect()
    all_df = agg.count_unique( all_df, ['ip', 'app'], 'os', 'Xcu3' ); gc.collect()
    all_df = agg.count_unique( all_df, ['ip'], 'device', 'Xcu4' ); gc.collect()
    all_df = agg.count_unique( all_df, ['app'], 'channel', 'Xcu5' ); gc.collect()
    all_df = agg.count_unique( all_df, ['device'], 'channel', 'Xcu6' ); gc.collect()
    all_df = agg.count_unique( all_df, ['ip', 'device', 'os'], 'app', 'Xcu7' ); gc.collect()
    
    all_df = agg.count( all_df, ['ip', 'device', 'day', 'hour'], 'Xc0'); gc.collect()
    all_df = agg.count( all_df, ['ip', 'device', 'day', 'hour', 'second'], 'Xc1'); gc.collect()
    all_df = agg.count( all_df, ['ip', 'day', 'hour'], 'Xc2' ); gc.collect()
    all_df = agg.count( all_df, ['ip', 'app'], 'Xc3' ); gc.collect()
    all_df = agg.count( all_df, ['ip', 'app', 'os'], 'Xc4' ); gc.collect()
    
    all_df = agg.variance( all_df, ['ip', 'day', 'channel'], 'hour', 'Xv0' ); gc.collect()
    all_df = agg.variance( all_df, ['ip', 'app', 'os'], 'hour', 'Xv1' ); gc.collect()
    all_df = agg.variance( all_df, ['ip', 'app', 'channel'], 'day', 'Xv2' ); gc.collect()
    all_df = agg.variance( all_df, ['ip', 'day', 'hour'], 'second', 'Xv3' ); gc.collect()
    
    all_df = agg.mean( all_df, ['ip', 'app', 'channel'], 'hour', 'Xv4' ); gc.collect()
    
    return all_df


all_df, train_n = read_tt()

all_df = agg_summary(all_df)

y_train = all_df[:train_n]['is_attributed']

all_df.drop(['click_time', 'is_attributed'], axis=1, inplace=True)

X_train = all_df[:train_n]
X_test = all_df[train_n:]

del all_df; gc.collect()

X_train.to_pickle('../input/preproc/X_train.pkl.gz', compression='gzip')
X_test.to_pickle('../input/preproc/X_test.pkl.gz', compression='gzip')
y_train.to_pickle('../input/preproc/y_train.pkl.gz', compression='gzip')






