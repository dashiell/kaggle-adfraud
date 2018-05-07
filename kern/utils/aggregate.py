#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 00:57:57 2018

@author: computer
"""

import pandas as pd

def count( df, group_cols, agg_name ):
    print( "Count by ", group_cols , '...' )
    
    gp = df[group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    df[agg_name] = pd.to_numeric(df[agg_name], downcast='integer')

    return df 

def count_unique( df, group_cols, count, agg_name):
    print( "Counting unqiue ", count, " by ", group_cols , '...' )
    
    gp = df[group_cols+[count]].groupby(group_cols)[count].nunique().reset_index().rename(columns={count:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    df[agg_name] = pd.to_numeric(df[agg_name], downcast='integer')
    
    return df
    
def cum_count( df, group_cols, count, agg_name):
    print( "Cumulative count by ", group_cols , '...' )
    
    gp = df[group_cols+[count]].groupby(group_cols)[count].cumcount()
    df[agg_name]=gp.values
    df[agg_name] = pd.to_numeric(df[agg_name], downcast='integer')
    
    return df 

def mean( df, group_cols, counted, agg_name):
    
    print( "Calculating mean of ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].mean().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    df[agg_name] = pd.to_numeric(df[agg_name], downcast='float')

    return df

def variance( df, group_cols, counted, agg_name):
    
    print( "Calculating variance of ", counted, " by ", group_cols , '...' )
    
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].var().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    df[agg_name] = pd.to_numeric(df[agg_name], downcast='float')

    return df


''' create summary data for each unique ip addr 
def add_agg_by_ip(df):
    
    df['click_time'] = pd.to_datetime(df['click_time'])
    
    ip_group = df.groupby('ip')
    
    ip_summary_data = { 
            'ip' : { 'agg__ip_count' : 'count'},
            'click_time' : { 'agg__time_first_click' : 'min',
                             'agg__time_last_click' : 'max',
                            },
            'app' : { 'agg__unique_app' : 'nunique' },
            'device' : { 'agg__unique_device' : 'nunique' },
            'os' : { 'agg__unique_os' : 'nunique' },
            'channel' : { 'agg__unique_channel' : 'nunique' }
    
    }
    summary_stats = ip_group.agg(ip_summary_data)
    summary_stats.columns = summary_stats.columns.droplevel(0)
    
    time_delta = summary_stats['time_last_click'] - summary_stats['time_first_click']
    
    ## TODO: Get the time order of the current click.. like 1/398, 2/398, 395/398
    # because estimate that downloads are at beginning / end of clicks
    summary_stats['time_delta_mins'] = time_delta / pd.Timedelta(1, unit='m') 
    summary_stats['mean_clicks_per_min'] = summary_stats['ip_count'] / summary_stats['time_delta_mins']
    summary_stats['mean_clicks_per_min'].replace(np.inf, 1, inplace=True)
    
    summary_stats.drop(['time_first_click', 'time_last_click'], axis=1, inplace=True)
    
    cols = summary_stats.columns.values

    # index is ip, changing this to a column.    
    summary_stats.reset_index(drop=False, inplace=True)
    
    summary_stats[cols] = preprocessing.scale(summary_stats[cols], axis = 0, with_mean=True, with_std=True)


    return summary_stats
    
'''