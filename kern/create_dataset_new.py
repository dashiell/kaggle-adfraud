#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
#from scipy import sparse
#from sklearn import preprocessing
from multiprocessing import Pool

import gc
from utils.Aggregate import Aggregate



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



def agg_wrapper(x):
        agg_func = x['func']
        params = x['params']
        
        ctr += 1
        outfile = '../input/preproc/X'+ctr+'.npy'
        
        output = agg_func(*params)
        
        np.save(outfile, output)
        
        return outfile
        #return agg_func(*params)

def agg_summary(all_df):

    
    agg = Aggregate(all_df)
    
    calls = [
            { 'func' : agg.count, 'params' : [ ['ip', 'device', 'day', 'hour'] ] } ,
            { 'func' : agg.count, 'params' :  [ ['ip', 'device', 'day', 'hour', 'second'] ] },
            { 'func' : agg.count, 'params' : [ ['ip', 'app'] ] },
            { 'func' : agg.count, 'params': [ ['ip', 'app', 'os'] ] },
            { 'func' : agg.cum_count, 'params' : [ ['ip', 'device', 'os'], 'app' ] },
            { 'func' : agg.cum_count, 'params' : [ ['ip'], 'os'] },
            { 'func' : agg.cum_count, 'params' : [ ['ip', 'day', 'hour'], 'os' ] },
            { 'func' : agg.cum_count, 'params' : [ ['ip', 'day', 'hour'], 'second' ] },
            { 'func' : agg.cum_count, 'params' : [ ['ip'], 'device' ] },
            { 'func' : agg.count_unique, 'params' : [ ['ip'], 'channel' ] } , 
            { 'func' : agg.count_unique, 'params' : [ ['ip', 'day'], 'hour' ] } , 
            { 'func' : agg.count_unique, 'params' : [ ['ip'], 'app' ] } , 
            { 'func' : agg.count_unique, 'params' : [ ['ip', 'app'], 'os' ] } , 
            { 'func' : agg.count_unique, 'params' : [ ['ip'], 'device' ] } , 
            { 'func' : agg.count_unique, 'params' : [ ['app'], 'channel' ] } , 
            { 'func' : agg.count_unique, 'params' : [ ['device'], 'channel' ] } , 
            { 'func' : agg.count_unique, 'params' : [ ['ip', 'device', 'os'], 'app' ] } , 
            { 'func' : agg.variance, 'params' : [ ['ip', 'day', 'channel'], 'hour' ] },
            { 'func' : agg.variance, 'params' : [ ['ip', 'app', 'os'], 'hour' ] },
            { 'func' : agg.variance, 'params' : [ ['ip', 'app', 'channel'], 'day' ] },
            { 'func' : agg.variance, 'params' : [ ['ip', 'day', 'hour'], 'second' ] },
            { 'func' : agg.mean, 'params' : [ ['ip', 'app', 'channel'], 'hour' ] },
    ]
    
    pool = Pool(processes=4)
    
    
    
    results = pool.imap_unordered(agg_wrapper, calls)
    #results = pool.starmap(agg.count, zip(all_df[count_groups], count_groups))
    pool.close()
    #pool.join()
    
    results = np.asarray(results).T
    
    return pd.DataFrame(results).add_prefix('agg_')

ctr = 0

all_df, train_n = read_tt()

temp = agg_summary(all_df)
'''
y_train = all_df[:train_n]['is_attributed']

all_df.drop(['click_time', 'is_attributed'], axis=1, inplace=True)

X_train = all_df[:train_n]
X_test = all_df[train_n:]

del all_df; gc.collect()

X_train.to_pickle('../input/preproc/X_train.pkl.gz', compression='gzip')
X_test.to_pickle('../input/preproc/X_test.pkl.gz', compression='gzip')
y_train.to_pickle('../input/preproc/y_train.pkl.gz', compression='gzip')


'''



